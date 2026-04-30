import re
import shutil
import shlex
import subprocess
import warnings

from datetime import datetime

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Optional

from rdkit import Chem
from rdkit.Chem import AllChem, rdDetermineBonds

from .protomer import Protomer, Species, Tautomer

HARTREE_TO_KCAL_MOL = 627.5094740631


def _append_log(log_path: Path, message: str) -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {message}\n")


def _log_status(log_paths: list[Path], status: str, message: str) -> None:
    for log_path in log_paths:
        _append_log(log_path, f"{status}: {message}")


def _run(
    cmd: str | list[str],
    *,
    cwd: Path,
    timeout_s: Optional[int] = None,
    dry_run: bool = False,
) -> subprocess.CompletedProcess[str]:
    """
    Run an external command and capture stdout/stderr.
    """
    if dry_run:
        # Keep behavior consistent with CompletedProcess enough for callers.
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        shell=isinstance(cmd, str),
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )


def _float_regex() -> str:
    # Matches:
    #  - 1.23, -1.23
    #  - 1, -1
    #  - 1.23e-4, -1e3
    return r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"


def _parse_first_float(patterns: list[str], text: str) -> Optional[float]:
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue
    return None
    
def _parse_last_float(patterns: list[str], text: str) -> Optional[float]:
    """
    Parse the last matching float for each pattern (useful when logs contain
    iterative/intermediate energies and a final summary value).
    """
    for pat in patterns:
        matches = list(re.finditer(pat, text, flags=re.IGNORECASE | re.MULTILINE))
        if not matches:
            continue
        for m in reversed(matches):
            try:
                return float(m.group(1))
            except ValueError:
                continue
    return None


def _parse_last_float(patterns: list[str], text: str) -> Optional[float]:
    """
    Parse the last matching float for each pattern (useful when logs contain
    iterative/intermediate energies and a final summary value).
    """
    for pat in patterns:
        matches = list(re.finditer(pat, text, flags=re.IGNORECASE | re.MULTILINE))
        if not matches:
            continue
        for m in reversed(matches):
            try:
                return float(m.group(1))
            except ValueError:
                continue
    return None


def _set_mol_prop_str(mol: Chem.Mol, key: str, value: Optional[str]) -> None:
    if value is None:
        return
    mol.SetProp(key, value)


def _set_mol_prop_double(mol: Chem.Mol, key: str, value: Optional[float]) -> None:
    if value is None:
        return
    mol.SetDoubleProp(key, float(value))


def _formal_charge(mol: Chem.Mol) -> int:
    # RDKit formal charge is guaranteed to be an integer (sum of formal charges).
    return int(Chem.GetFormalCharge(mol))


def _embed_conformers_rdkit_mmff94(
    mol: Chem.Mol,
    *,
    random_seed: int = 42,
    mmff_max_iters: int = 1000,
) -> tuple[Chem.Mol, float, int]:
    """
    Generate conformers with RDKit ETKDG and optimize with MMFF94.

    Returns: (mol_no_h, best_energy_kcal_mol, best_conf_id)
    """
    if mol is None:
        raise ValueError("mol is None")

    mol_in = Chem.Mol(mol)
    mol_h = Chem.AddHs(mol_in, addCoords=False)

    # KDGv3 minus ET
    params = AllChem.ETKDGv3()
    params.randomSeed = int(random_seed)
    params.numThreads = 0
    params.useExpTorsionAnglePrefs = False # better for liquid phase

    n_rotatable_bonds = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol_h)
    n_confs = min(100, 2 ** n_rotatable_bonds, 2000)
    conf_ids = list(AllChem.EmbedMultipleConfs(mol_h, int(n_confs), params))
    if not conf_ids:
        raise RuntimeError("RDKit conformer embedding produced no conformers.")

    best_energy = float("inf")
    best_conf_id = conf_ids[0]

    # optimize confs
    for conf_id in conf_ids:
        status = AllChem.MMFFOptimizeMolecule(
            mol_h,
            mmffVariant="MMFF94",
            maxIters=int(mmff_max_iters),
            confId=int(conf_id),
        )
        if status == -1:
            # Force field setup failed for this conformer. Skip it.
            continue

        # Compute MMFF energy for selection (lowest MMFF energy).
        mmff_props = AllChem.MMFFGetMoleculeProperties(mol_h, mmffVariant="MMFF94")
        if mmff_props is None:
            raise RuntimeError("MMFF94 molecule properties could not be created.")
        ff = AllChem.MMFFGetMoleculeForceField(mol_h, mmff_props, confId=int(conf_id))
        ff.Initialize()
        energy = float(ff.CalcEnergy())
        if energy < best_energy:
            best_energy = energy
            best_conf_id = int(conf_id)

    # Drop explicit Hs for storage; keep best heavy-atom coordinates.
    mol_best_h = Chem.Mol(mol_h)
    mol_best_h.RemoveAllConformers()
    mol_best_h.AddConformer(mol_h.GetConformer(best_conf_id), assignId=True)
    mol_no_h = Chem.RemoveHs(mol_best_h)

    return mol_no_h, best_energy, best_conf_id


def _mol_to_xyz_block(mol: Chem.Mol, *, conf_id: int = 0) -> str:
    if mol.GetNumConformers() == 0:
        raise ValueError("Molecule has no conformers; cannot write 3D xyz.")
    # xyz coordinates need  explicit hydrogens in xyz blocks
    mol_h = Chem.AddHs(Chem.Mol(mol), addCoords=True)
    return Chem.MolToXYZBlock(mol_h, confId=conf_id)


def _write_xyz(mol: Chem.Mol, path: Path, *, conf_id: int = 0) -> None:
    xyz = _mol_to_xyz_block(mol, conf_id=conf_id)
    path.write_text(xyz)


def _build_fix_xcontrol(mol: Chem.Mol, xcontrol_path: Path, *, fixed_distance: float = 1.05) -> int:
    """
    Build xTB xcontrol constraints for positively charged zwitterionic X-H sites.

    For each heavy atom with formal charge +1 that is bonded to at least one
    hydrogen, write:
      $constrain
       distance: X_idx, H_idx, <fixed_distance>
      $end

    Bond distances are selected by heavy-atom element symbol where available.
    If no element-specific entry exists, ``fixed_distance`` is used.

    Returns number of generated distance constraints.
    """
    mol_h = Chem.AddHs(Chem.Mol(mol), addCoords=True)
    constraints: list[str] = []
    zwitterion_xh_distance_by_symbol = {
        "N": 1.03,
    }

    for atom in mol_h.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        if atom.GetFormalCharge() != 1:
            continue

        h_neighbors = [nbr for nbr in atom.GetNeighbors() if nbr.GetAtomicNum() == 1]
        if not h_neighbors:
            continue

        heavy_idx = atom.GetIdx() + 1  # xTB uses 1-based indexing.
        constrained_distance = zwitterion_xh_distance_by_symbol.get(
            atom.GetSymbol(),
            fixed_distance,
        )
        for h_atom in h_neighbors:
            h_idx = h_atom.GetIdx() + 1
            constraints.append(f" distance: {heavy_idx},{h_idx},{constrained_distance:.2f}")

    if constraints:
        content = "$constrain\n" + "\n".join(constraints) + "\n$end\n"
        xcontrol_path.write_text(content)
    elif xcontrol_path.exists():
        xcontrol_path.unlink()

    return len(constraints)


def _parse_xtb_total_energy_hartree(text: str) -> Optional[float]:
    float_re = _float_regex()
    patterns = [
        rf"total energy[^\S\r\n]*[:=]?[^\S\r\n]*({float_re})",
    ]
    return _parse_last_float(patterns, text)


def _parse_xtb_solvent_free_energy_hartree(
    text: str,
    *,
    mode: Literal["g", "e"] = "g",
) -> Optional[float]:
    float_re = _float_regex()
    if mode == "g":
        patterns = [
            rf"solvation free energy \(dG_solv\):[\ \t]*({float_re})",
        ]
    elif mode == "e":
        patterns = [
            rf"Gsolv[^\S\r\n]*[:=]?[^\S\r\n]*({float_re})",
        ]
    else:
        raise ValueError(f"Unknown solvation parse mode: {mode}")
    return _parse_last_float(patterns, text)


def _parse_xtb_rrho_contrib(text: str) -> Optional[float]:
    float_re = _float_regex()
    patterns = [
        # xTB RRHO summary line, e.g.:
        # :: G(RRHO) contrib.            0.047744476512 Eh   ::
        rf"G\(RRHO\)[^\S\r\n]*contrib\.?[^\S\r\n]*[:=]?[^\S\r\n]*({float_re})",
    ]
    return _parse_last_float(patterns, text)


@dataclass(frozen=True)
class SolvationWorkflowResult:
    conformer_energy_kcal_mol: Optional[float]
    xtb_optimized_xyz: Optional[Path]
    solvation_free_energy_kcal_mol: Optional[float]
    gas_sp_energy_kcal_mol: Optional[float]
    rrho_contribution_kcal_mol: Optional[float]
    solution_phase_free_energy_kcal_mol: Optional[float]
    stdout_tail: str


@dataclass(frozen=True)
class ScreeningWorkflowResult:
    conformer_energy_kcal_mol: Optional[float]
    solvation_free_energy_kcal_mol: Optional[float]
    gas_sp_energy_kcal_mol: Optional[float]
    rrho_contribution_kcal_mol: Optional[float]
    solution_phase_free_energy_kcal_mol: Optional[float]
    stdout_tail: str


@dataclass
class SolvationScratchContext:
    scratch_root: Path
    scratch_dir: Path
    workflow_log: Path
    run_log: Path
    log_paths: list[Path]


def _create_scratch_context(scratch_root: str | Path, protomer_id: int | str) -> SolvationScratchContext:
    scratch_root_path = Path(scratch_root)
    scratch_root_path.mkdir(parents=True, exist_ok=True)
    workflow_log = scratch_root_path / "peace.out"

    scratch_dir = scratch_root_path / f"protomer_{protomer_id}"
    scratch_dir.mkdir(parents=True, exist_ok=False)
    run_log = scratch_dir / "run.out"

    return SolvationScratchContext(
        scratch_root=scratch_root_path,
        scratch_dir=scratch_dir,
        workflow_log=workflow_log,
        run_log=run_log,
        log_paths=[workflow_log, run_log],
    )


def _prepare_protomer_conformer(
    protomer: Protomer,
    *,
    conformer_mode: Literal["mmff94", "external_xyz", "skip_search"],
    external_xyz_path: Optional[str | Path],
    log_paths: list[Path],
) -> tuple[Chem.Mol, Optional[float]]:
    mol = protomer.mol
    if mol is None:
        raise ValueError("Protomer.mol is None; cannot run workflow.")
    if getattr(protomer, "input_mol", None) is None:
        protomer.input_mol = Chem.Mol(mol)

    _log_status(log_paths, "STEP", "starting conformer preparation")
    conformer_energy_kcal_mol: Optional[float] = None

    if conformer_mode == "mmff94":
        mol_best, best_energy, best_conf_id = _embed_conformers_rdkit_mmff94(mol)
        conformer_energy_kcal_mol = best_energy
        protomer.mol = mol_best
        protomer.mol.SetProp("peace_conformer_mode", "mmff94")
        _log_status(
            log_paths,
            "OK",
            f"rdkit conformer search complete best_conf_id={best_conf_id} best_energy_kcal_mol={best_energy}",
        )
        return protomer.mol, conformer_energy_kcal_mol

    if conformer_mode == "external_xyz":
        if external_xyz_path is None:
            raise ValueError("external_xyz_path must be provided for conformer_mode='external_xyz'.")
        mol = Chem.MolFromXYZFile(str(external_xyz_path))
        if mol is None:
            raise ValueError(f"RDKit failed to read xyz: {external_xyz_path}")
        if mol.GetNumConformers() == 0:
            raise ValueError("External xyz produced a molecule without conformers.")
        protomer.mol = mol
        if getattr(protomer, "input_mol", None) is None:
            protomer.input_mol = Chem.Mol(mol)
        protomer.mol.SetProp("peace_conformer_mode", "external_xyz")
        _log_status(log_paths, "OK", f"loaded external xyz from {external_xyz_path}")
        return protomer.mol, conformer_energy_kcal_mol

    if conformer_mode == "skip_search":
        if mol.GetNumConformers() == 0:
            raise ValueError("conformer_mode='skip_search' but molecule has no conformers.")
        protomer.mol = mol
        if getattr(protomer, "input_mol", None) is None:
            protomer.input_mol = Chem.Mol(mol)
        protomer.mol.SetProp("peace_conformer_mode", "skip_search")
        _log_status(log_paths, "OK", "using existing conformer on protomer.mol")
        return protomer.mol, conformer_energy_kcal_mol

    raise ValueError(f"Unknown conformer_mode: {conformer_mode}")


def _write_workflow_inputs(mol: Chem.Mol, scratch_dir: Path, charge: int, log_paths: list[Path]) -> Path:
    input_xyz_path = scratch_dir / "input.xyz"
    _write_xyz(mol, input_xyz_path, conf_id=0)
    _log_status(log_paths, "OK", f"wrote input geometry to {input_xyz_path.name}")
    return input_xyz_path


def _run_xtb_optimization(
    *,
    mol: Chem.Mol,
    scratch_dir: Path,
    input_xyz_path: Path,
    xtb_executable: str,
    opt_level: str,
    solvent: str,
    timeout_s: Optional[int],
    dry_run: bool,
    log_paths: list[Path],
) -> tuple[Path, Optional[float], Optional[float]]:
    xtbopt_xyz_path = scratch_dir / "xtbopt.xyz"
    xcontrol_path = scratch_dir / "xcontrol.inp"
    n_constraints = _build_fix_xcontrol(
        mol,
        xcontrol_path,
        fixed_distance=1.01,
    )

    input_flag = ""
    if n_constraints > 0:
        input_flag = f" --input {shlex.quote(xcontrol_path.name)}"
        _log_status(
            log_paths,
            "OK",
            "generated xcontrol constraints for positively charged heavy-atom X-H bonds: "
            f"n_constraints={n_constraints}",
        )
    else:
        _log_status(log_paths, "OK", "no positively charged heavy-atom X-H constraints generated")

    cmd_opt = (
        f"{shlex.quote(xtb_executable)} {shlex.quote(input_xyz_path.name)} "
        f"{input_flag} "
        f"--opt {shlex.quote(opt_level)} --alpb {shlex.quote(solvent)}"
    )
    _log_status(log_paths, "STEP", f"running optimization: {cmd_opt}")
    cp_opt = _run(cmd_opt, cwd=scratch_dir, timeout_s=timeout_s, dry_run=dry_run)
    if not dry_run:
        optimization_log_path = scratch_dir / "xtbopt_run.log"
        optimization_log_path.write_text(cp_opt.stdout)
        _log_status(log_paths, "OK", f"saved optimization stdout to {optimization_log_path.name}")
    if cp_opt.returncode != 0:
        _log_status(
            log_paths,
            "FAIL",
            f"optimization failed returncode={cp_opt.returncode} stdout_tail={cp_opt.stdout[-1000:]} stderr_tail={cp_opt.stderr[-1000:]}",
        )
        raise RuntimeError(
            f"xTB optimization failed with code {cp_opt.returncode}.\n"
            f"stdout:\n{cp_opt.stdout[-4000:]}\n"
            f"stderr:\n{cp_opt.stderr[-4000:]}\n"
        )
    if not xtbopt_xyz_path.exists():
        _log_status(log_paths, "FAIL", "optimization finished but xtbopt.xyz was not produced")
        raise FileNotFoundError("Expected output xtbopt.xyz was not produced by xTB.")

    _log_status(log_paths, "OK", f"optimization produced {xtbopt_xyz_path.name}")

    # Parse gas-phase SP energy directly from optimization output/log.
    parse_text = cp_opt.stdout
    xtbopt_log_path = scratch_dir / "xtbopt.log"
    if xtbopt_log_path.exists():
        parse_text += "\n" + xtbopt_log_path.read_text()

    gas_sp_energy_h = _parse_xtb_total_energy_hartree(parse_text)
    gas_sp_energy_kcal_mol = None
    if gas_sp_energy_h is not None:
        gas_sp_energy_kcal_mol = gas_sp_energy_h * HARTREE_TO_KCAL_MOL
    else:
        warnings.warn(
            "Could not parse TOTAL ENERGY from optimization output/log. "
            "Gas-phase energy will remain None.",
            RuntimeWarning,
        )
        _log_status(log_paths, "WARN", "failed to parse gas-phase SP energy from optimization output/log")

    return xtbopt_xyz_path, gas_sp_energy_kcal_mol, gas_sp_energy_h


def _all_atom_connectivity_signature(mol: Chem.Mol) -> set[tuple[int, int]]:
    """
    Return all-atom connectivity as undirected atom-index pairs.
    Used to check whether an opt. geom matches input geom.
    """
    edges: set[tuple[int, int]] = set()
    if mol is None:
        return edges
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtom()
        end = bond.GetEndAtom()
        i, j = begin.GetIdx(), end.GetIdx()
        edges.add((min(i, j), max(i, j)))
    return edges


def _all_atom_connectivity_signature(mol: Chem.Mol) -> set[tuple[int, int]]:
    """
    Return all-atom connectivity as undirected atom-index pairs.
    Used to check whether an opt. geom matches input geom.
    """
    edges: set[tuple[int, int]] = set()
    if mol is None:
        return edges
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtom()
        end = bond.GetEndAtom()
        i, j = begin.GetIdx(), end.GetIdx()
        edges.add((min(i, j), max(i, j)))
    return edges

def _update_protomer_geometry_from_xyz(protomer: Protomer, xyz_path: Path, log_paths: list[Path]) -> Optional[str]:
  # build a bonded graph from optimized xyz coordinates for sanity checking.
    mol_opt = Chem.MolFromXYZFile(str(xyz_path))
    if mol_opt is not None and mol_opt.GetNumConformers() > 0:
        rdDetermineBonds.DetermineConnectivity(mol_opt)

        # sanity check: optimized connectivity should match input connectivity.
        input_mol = protomer.input_mol if getattr(protomer, "input_mol", None) is not None else protomer.mol
        # make a copy of the input mol that has explicit Hydrogens
        input_mol_with_hydrogens = Chem.AddHs(input_mol)
        
        input_edges = _all_atom_connectivity_signature(input_mol_with_hydrogens)
        opt_edges = _all_atom_connectivity_signature(mol_opt)
        if input_edges != opt_edges:
            mol_opt.SetProp("peace_connectivity_mismatch", "true")
            mol_opt.SetProp(
                "peace_connectivity_mismatch_error",
                (
                    "Optimized structure connectivity differs from input mol. "
                    f"Got: {sorted(opt_edges)}, Expected: {sorted(input_edges)}"
                )[:4000],
            )
            warnings.warn(
                f"Optimized structure connectivity differs from input mol!! Please double-check the optimized structure. Got: {opt_edges}, Expected: {input_edges}",
                RuntimeWarning,
            )
            _log_status(
                log_paths,
                "WARN",
                "optimized connectivity does not match input connectivity!!",
            )
        else:
            mol_opt.SetProp("peace_connectivity_mismatch", "false")
        protomer.mol = mol_opt
        _log_status(log_paths, "OK", f"updated protomer geometry from {xyz_path.name}")
        return xyz_path.read_text()
    return None

def _run_cpcmx_single_point(
    *,
    scratch_dir: Path,
    xtbopt_xyz_path: Path,
    xtb_executable: str,
    solvent: str,
    charge: int,
    gfn: int,
    parse_solvation: Literal["g", "e"],
    timeout_s: Optional[int],
    dry_run: bool,
    log_paths: list[Path],
) -> Optional[float]:
    cmd_sp = [
        xtb_executable,
        str(xtbopt_xyz_path.name),
        "--cpcmx",
        solvent,
        "--chrg",
        str(charge),
        "--gfn",
        str(gfn),
    ]
    _log_status(log_paths, "STEP", f"running CPCM-X SP: {' '.join(shlex.quote(x) for x in cmd_sp)}")
    cp_sp = _run(cmd_sp, cwd=scratch_dir, timeout_s=timeout_s, dry_run=dry_run)
    if not dry_run:
        solvation_log_path = scratch_dir / "xtbsolv_run.log"
        solvation_log_path.write_text(cp_sp.stdout)
        _log_status(log_paths, "OK", f"saved CPCM-X stdout to {solvation_log_path.name}")
    if cp_sp.returncode != 0:
        _log_status(
            log_paths,
            "FAIL",
            f"CPCM-X SP failed returncode={cp_sp.returncode} stdout_tail={cp_sp.stdout[-1000:]} stderr_tail={cp_sp.stderr[-1000:]}",
        )
        raise RuntimeError(
            f"xTB CPCM-X SP calculation failed with code {cp_sp.returncode}.\n"
            f"stdout:\n{cp_sp.stdout[-4000:]}\n"
            f"stderr:\n{cp_sp.stderr[-4000:]}\n"
        )

    solvation_free_energy_h = _parse_xtb_solvent_free_energy_hartree(cp_sp.stdout, mode=parse_solvation)

    solvation_free_energy_kcal_mol = None
    if solvation_free_energy_h is not None:
        solvation_free_energy_kcal_mol = solvation_free_energy_h * HARTREE_TO_KCAL_MOL

    _log_status(
        log_paths,
        "OK",
        f"parsed solvation_free_energy_kcal_mol={solvation_free_energy_kcal_mol}",
    )
    return solvation_free_energy_kcal_mol


def _run_gxtb_single_point_energy(
    *,
    scratch_dir: Path,
    xtbopt_xyz_path: Path,
    xtb_executable: str,
    charge: int,
    timeout_s: Optional[int],
    dry_run: bool,
    log_paths: list[Path],
) -> tuple[Optional[float], Optional[float]]:
    cmd_sp = (
        f"{shlex.quote(xtb_executable)} {shlex.quote(xtbopt_xyz_path.name)} "
        f'--driver "gxtb -grad -c xtbdriver.xyz" '
        f"--chrg {shlex.quote(str(charge))}"
    )
    _log_status(log_paths, "STEP", f"running g-xTB gas-phase SP via driver: {cmd_sp}")
    cp_sp = _run(cmd_sp, cwd=scratch_dir, timeout_s=timeout_s, dry_run=dry_run)
    if not dry_run:
        gxtbsp_log_path = scratch_dir / "gxtbsp_run.log"
        gxtbsp_log_path.write_text(cp_sp.stdout)
        _log_status(log_paths, "OK", f"saved g-xTB SP stdout to {gxtbsp_log_path.name}")
    if cp_sp.returncode != 0:
        _log_status(
            log_paths,
            "FAIL",
            f"g-xTB SP failed returncode={cp_sp.returncode} stdout_tail={cp_sp.stdout[-1000:]} stderr_tail={cp_sp.stderr[-1000:]}",
        )
        raise RuntimeError(
            f"g-xTB gas-phase SP calculation failed with code {cp_sp.returncode}.\n"
            f"stdout:\n{cp_sp.stdout[-4000:]}\n"
            f"stderr:\n{cp_sp.stderr[-4000:]}\n"
        )

    gas_sp_energy_h = _parse_xtb_total_energy_hartree(cp_sp.stdout)
    gas_sp_energy_kcal_mol = None
    if gas_sp_energy_h is not None:
        gas_sp_energy_kcal_mol = gas_sp_energy_h * HARTREE_TO_KCAL_MOL
    else:
        warnings.warn(
            "Could not parse TOTAL ENERGY from g-xTB SP output. "
            "Gas-phase energy will remain None.",
            RuntimeWarning,
        )
        _log_status(log_paths, "WARN", "failed to parse gas-phase SP energy from g-xTB SP output")

    return gas_sp_energy_kcal_mol, gas_sp_energy_h


def _run_hessian_and_parse_energies(
    *,
    scratch_dir: Path,
    xtbopt_xyz_path: Path,
    xtb_executable: str,
    charge: int,
    gfn: int,
    timeout_s: Optional[int],
    dry_run: bool,
    log_paths: list[Path],
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    cmd_hess = [
        xtb_executable,
        str(xtbopt_xyz_path.name),
        "--hess",
        "--chrg",
        str(charge),
        "--gfn",
        str(gfn),
    ]
    _log_status(log_paths, "STEP", f"running hessian: {' '.join(shlex.quote(x) for x in cmd_hess)}")
    cp_hess = _run(cmd_hess, cwd=scratch_dir, timeout_s=timeout_s, dry_run=dry_run)
    if not dry_run:
        frequency_log_path = scratch_dir / "xtbfreq_run.log"
        frequency_log_path.write_text(cp_hess.stdout)
        _log_status(log_paths, "OK", f"saved frequency stdout to {frequency_log_path.name}")
    if cp_hess.returncode != 0:
        _log_status(
            log_paths,
            "FAIL",
            f"hessian failed returncode={cp_hess.returncode} stdout_tail={cp_hess.stdout[-1000:]} stderr_tail={cp_hess.stderr[-1000:]}",
        )
        raise RuntimeError(
            f"xTB hess calculation failed with code {cp_hess.returncode}.\n"
            f"stdout:\n{cp_hess.stdout[-4000:]}\n"
            f"stderr:\n{cp_hess.stderr[-4000:]}\n"
        )

    gas_sp_energy_h = _parse_xtb_total_energy_hartree(cp_hess.stdout)
    rrho_contrib_h = _parse_xtb_rrho_contrib(cp_hess.stdout)

    gas_sp_energy_kcal_mol = None

    if gas_sp_energy_h is not None:
        gas_sp_energy_kcal_mol = gas_sp_energy_h * HARTREE_TO_KCAL_MOL

    rrho_contrib_kcal_mol = None
    if rrho_contrib_h is not None:
        rrho_contrib_kcal_mol = rrho_contrib_h * HARTREE_TO_KCAL_MOL

    return gas_sp_energy_kcal_mol, rrho_contrib_kcal_mol, gas_sp_energy_h


def _compute_solution_phase_energy(
    gas_sp_energy_kcal_mol: Optional[float],
    solvation_free_energy_kcal_mol: Optional[float],
    rrho_contribution_kcal_mol: Optional[float],
    log_paths: list[Path],
) -> Optional[float]:
    solution_phase_free_energy_kcal_mol = None
    if (
        gas_sp_energy_kcal_mol is not None
        and solvation_free_energy_kcal_mol is not None
        and rrho_contribution_kcal_mol is not None
    ):
        # see doi:10.1021/acs.jpca.3c04382.
        # Gtotal = Egas,0K + DG(RRHO) + DGSolv

        solution_phase_free_energy_kcal_mol = (
            gas_sp_energy_kcal_mol
            + rrho_contribution_kcal_mol
            + solvation_free_energy_kcal_mol
        )

    _log_status(
        log_paths,
        "OK",
        "parsed energies "
        f"gas_sp_energy_kcal_mol={gas_sp_energy_kcal_mol} "
        f"rrho_contribution_kcal_mol={rrho_contribution_kcal_mol} "
        f"solution_phase_free_energy_kcal_mol={solution_phase_free_energy_kcal_mol}",
    )
    return solution_phase_free_energy_kcal_mol


def _persist_protomer_results(
    protomer: Protomer,
    *,
    charge: int,
    conformer_energy_kcal_mol: Optional[float],
    solvation_free_energy_kcal_mol: Optional[float],
    gas_sp_energy_kcal_mol: Optional[float],
    rrho_contribution_kcal_mol: Optional[float],
    solution_phase_free_energy_kcal_mol: Optional[float],
) -> None:
    protomer.mol.SetProp("peace_charge", str(charge))
    _set_mol_prop_double(protomer.mol, "peace_conformer_energy_kcal_mol", conformer_energy_kcal_mol)
    _set_mol_prop_double(protomer.mol, "peace_solvation_free_energy_kcal_mol", solvation_free_energy_kcal_mol)
    _set_mol_prop_double(protomer.mol, "peace_gas_sp_energy_kcal_mol", gas_sp_energy_kcal_mol)
    _set_mol_prop_double(
        protomer.mol,
        "peace_rrho_contribution_kcal_mol",
        rrho_contribution_kcal_mol,
    )
    _set_mol_prop_double(
        protomer.mol,
        "peace_solution_phase_free_energy_kcal_mol",
        solution_phase_free_energy_kcal_mol,
    )


def _preserve_output_files(
    scratch_dir: Path,
    *,
    keep_logs: bool = False,
) -> Optional[Path]:
    if not scratch_dir.exists():
        return None

    log_paths = [scratch_dir.parent / "peace.out", scratch_dir / "run.out"]
    preserved_dir = scratch_dir.parent / "xyz"
    preserved_dir.mkdir(parents=True, exist_ok=True)

    preserved_xtbopt_path: Optional[Path] = None
    files_to_preserve = [
        "input.xyz",
        "xtbopt.xyz",
        "xtbopt.log"
    ]
    for file_name in files_to_preserve:
        src = scratch_dir / file_name
        if not src.exists():
            continue
        dst = preserved_dir / f"{scratch_dir.name}_{file_name}"
        shutil.copy2(src, dst)
        _log_status(log_paths, "KEEP", f"preserved {file_name} at {dst}")
        if file_name == "xtbopt.xyz":
            preserved_xtbopt_path = dst
    if keep_logs:
        preserved_log_dir = scratch_dir.parent / "log"
        preserved_log_dir.mkdir(parents=True, exist_ok=True)
        log_files_to_preserve = [
            "xtbopt_run.log",
            "gxtbsp_run.log",
            "xtbsolv_run.log",
            "xtbfreq_run.log",
        ]
        for file_name in log_files_to_preserve:
            src = scratch_dir / file_name
            if not src.exists():
                continue
            dst = preserved_log_dir / f"{scratch_dir.name}_{file_name}"
            shutil.copy2(src, dst)
            _log_status(log_paths, "KEEP", f"preserved {file_name} at {dst}")

    return preserved_xtbopt_path


def _cleanup_scratch_dir(
    scratch_dir: Path,
    *,
    keep_scratch: bool,
    dry_run: bool,
    log_paths: list[Path],
    success: bool,
) -> None:
    if keep_scratch or dry_run:
        return

    if scratch_dir.exists() and not dry_run:
        phase = "successful" if success else "failed"
        _log_status(log_paths, "CLEANUP", f"removing scratch directory after {phase} run")
        shutil.rmtree(scratch_dir, ignore_errors=True) # TODO: for some reason this is not actually working


def run_protomer_screening(
    protomer: Protomer,
    *,
    protomer_id: int | str = 0,
    scratch_root: str | Path = "./peace_scratch_solvation",
    conformer_mode: Literal["mmff94", "external_xyz", "skip_search"] = "mmff94",
    external_xyz_path: Optional[str | Path] = None,
    charge_override: Optional[int] = None,
    solvent: Literal["water"] = "water",
    gfn: int = 2,
    parse_solvation: Literal["g", "e"] = "g",
    opt_level: Literal["loose", "tight", "vtight"] = "loose",
    xtb_executable: str = "xtb",
    keep_scratch: bool = False,
    keep_logs: bool = False,
    keep_scratch_on_failure: bool = False,
    dry_run: bool = False,
    timeout_s: Optional[int] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> ScreeningWorkflowResult:
    """
    Lightweight pre-screening workflow for protomer pruning.

    Steps:
    1) Build/choose conformer geometry.
    2) Run xTB geometry optimization in solvent.
    3) Run xTB CPCM-X single-point solvation on optimized geometry.
    4) Run xTB Hessian on optimized geometry to get gas SP + RRHO term.
    5) Compute screening solution-phase free energy (without g-xTB).
    """
    def _progress(message: str) -> None:
        if progress_callback is not None:
            progress_callback(message)

    scratch_context = _create_scratch_context(scratch_root, protomer_id)
    scratch_dir = scratch_context.scratch_dir
    log_paths = scratch_context.log_paths

    if protomer.mol is None:
        raise ValueError("Protomer does not have mol; cannot run screening workflow.")
    charge = int(charge_override) if charge_override is not None else _formal_charge(protomer.mol)
    _log_status(
        log_paths,
        "START",
        f"screening protomer_id={protomer_id} scratch_dir={scratch_dir.name} charge={charge} conformer_mode={conformer_mode}",
    )
    _progress("preparing conformer")

    conformer_energy_kcal_mol: Optional[float] = None
    solvation_free_energy_kcal_mol: Optional[float] = None
    gas_sp_energy_kcal_mol: Optional[float] = None
    rrho_contribution_kcal_mol: Optional[float] = None
    solution_phase_free_energy_kcal_mol: Optional[float] = None

    try:
        if dry_run:
            _progress("dry run enabled; skipping screening workflow")
            _log_status(log_paths, "SKIP", "dry_run enabled; skipping screening steps")
            return ScreeningWorkflowResult(
                conformer_energy_kcal_mol=None,
                solvation_free_energy_kcal_mol=None,
                gas_sp_energy_kcal_mol=None,
                rrho_contribution_kcal_mol=None,
                solution_phase_free_energy_kcal_mol=None,
                stdout_tail="dry_run; skipped screening steps.",
            )

        mol, conformer_energy_kcal_mol = _prepare_protomer_conformer(
            protomer,
            conformer_mode=conformer_mode,
            external_xyz_path=external_xyz_path,
            log_paths=log_paths,
        )
        input_xyz_path = _write_workflow_inputs(mol, scratch_dir, charge, log_paths)
        _progress("optimizing screening geometry")
        xtbopt_xyz_path, _opt_gas_sp_kcal_mol, _opt_gas_sp_h = _run_xtb_optimization(
            mol=mol,
            scratch_dir=scratch_dir,
            input_xyz_path=input_xyz_path,
            xtb_executable=xtb_executable,
            solvent=solvent,
            opt_level=opt_level,
            timeout_s=timeout_s,
            dry_run=dry_run,
            log_paths=log_paths,
        )
        _update_protomer_geometry_from_xyz(protomer, xtbopt_xyz_path, log_paths)

        _progress("computing screening solvation single point")
        solvation_free_energy_kcal_mol = _run_cpcmx_single_point(
            scratch_dir=scratch_dir,
            xtbopt_xyz_path=xtbopt_xyz_path,
            xtb_executable=xtb_executable,
            solvent=solvent,
            charge=charge,
            gfn=gfn,
            parse_solvation=parse_solvation,
            timeout_s=timeout_s,
            dry_run=dry_run,
            log_paths=log_paths,
        )

        _progress("computing screening frequencies")
        gas_sp_energy_kcal_mol, rrho_contribution_kcal_mol, _ = _run_hessian_and_parse_energies(
            scratch_dir=scratch_dir,
            xtbopt_xyz_path=xtbopt_xyz_path,
            xtb_executable=xtb_executable,
            charge=charge,
            gfn=gfn,
            timeout_s=timeout_s,
            dry_run=dry_run,
            log_paths=log_paths,
        )

        solution_phase_free_energy_kcal_mol = _compute_solution_phase_energy(
            gas_sp_energy_kcal_mol,
            solvation_free_energy_kcal_mol,
            rrho_contribution_kcal_mol,
            log_paths,
        )
        _set_mol_prop_double(protomer.mol, "peace_screening_conformer_energy_kcal_mol", conformer_energy_kcal_mol)
        _set_mol_prop_double(protomer.mol, "peace_screening_solvation_free_energy_kcal_mol", solvation_free_energy_kcal_mol)
        _set_mol_prop_double(protomer.mol, "peace_screening_gas_sp_energy_kcal_mol", gas_sp_energy_kcal_mol)
        _set_mol_prop_double(protomer.mol, "peace_screening_rrho_contribution_kcal_mol", rrho_contribution_kcal_mol)
        _set_mol_prop_double(protomer.mol, "peace_screening_solution_phase_free_energy_kcal_mol", solution_phase_free_energy_kcal_mol)
        _progress("finished screening workflow")

    except Exception as e:
        _progress(f"failed screening: {e}")
        _log_status(log_paths, "FAIL", f"screening exception for protomer_id={protomer_id}: {e}")
        warnings.warn(
            f"Screening workflow failed for protomer_id={protomer_id}: {e}",
            RuntimeWarning,
        )
        if protomer.mol is not None:
            _set_mol_prop_str(protomer.mol, "peace_screening_error", str(e)[:4000])
        _preserve_output_files(scratch_dir, keep_logs=keep_logs)
        keep = keep_scratch or keep_scratch_on_failure
        _cleanup_scratch_dir(
            scratch_dir,
            keep_scratch=keep,
            dry_run=dry_run,
            log_paths=log_paths,
            success=False,
        )
        return ScreeningWorkflowResult(
            conformer_energy_kcal_mol=conformer_energy_kcal_mol,
            solvation_free_energy_kcal_mol=None,
            gas_sp_energy_kcal_mol=None,
            rrho_contribution_kcal_mol=None,
            solution_phase_free_energy_kcal_mol=None,
            stdout_tail=str(e)[-4000:],
        )

    _preserve_output_files(scratch_dir, keep_logs=keep_logs)
    _cleanup_scratch_dir(
        scratch_dir,
        keep_scratch=keep_scratch,
        dry_run=dry_run,
        log_paths=log_paths,
        success=True,
    )
    _log_status(
        log_paths,
        "SUCCESS",
        "screening complete "
        f"gas={gas_sp_energy_kcal_mol} solv={solvation_free_energy_kcal_mol} "
        f"freq={rrho_contribution_kcal_mol} solution={solution_phase_free_energy_kcal_mol}",
    )
    _progress("screening success")

    return ScreeningWorkflowResult(
        conformer_energy_kcal_mol=conformer_energy_kcal_mol,
        solvation_free_energy_kcal_mol=solvation_free_energy_kcal_mol,
        gas_sp_energy_kcal_mol=gas_sp_energy_kcal_mol,
        rrho_contribution_kcal_mol=rrho_contribution_kcal_mol,
        solution_phase_free_energy_kcal_mol=solution_phase_free_energy_kcal_mol,
        stdout_tail=(
            f"screening ok; parsed values: "
            f"solv={solvation_free_energy_kcal_mol}, gas={gas_sp_energy_kcal_mol}"
        )[-4000:],
    )


def run_protomer_solvation(
    protomer: Protomer,
    *,
    protomer_id: int | str = 0,
    scratch_root: str | Path = "./peace_scratch_solvation",
    conformer_mode: Literal["mmff94", "external_xyz", "skip_search"] = "mmff94",
    external_xyz_path: Optional[str | Path] = None,
    charge_override: Optional[int] = None,
    solvent: Literal["water"] = "water",
    gfn: int = 2,
    parse_solvation: Literal["g", "e"] = "g",
    opt_level: Literal["loose", "tight", "vtight"] = "loose",
    xtb_executable: str = "xtb",
    sp_energy: Literal["gxtb", "xtb"] = "gxtb",
    run_geometry_optimization: bool = False,
    recompute_solvation: bool = False,
    recompute_frequencies: bool = False,
    reuse_screening_terms: bool = True,
    keep_scratch: bool = False,
    keep_logs: bool = False,
    keep_scratch_on_failure: bool = False,
    dry_run: bool = False,
    timeout_s: Optional[int] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> SolvationWorkflowResult:
    """
    Solvates a protomer and gets the solution-phase energy

    Default staged behavior:
    1) Reuse existing pre-screened geometry and terms if present.
    2) Run g-xTB gas-phase SP on current geometry.
    3) Combine gas SP + RRHO + solvation into final solution-phase free energy.
    Optional flags allow geometry optimization and/or recomputing solvation and frequency terms.
    """
    def _progress(message: str) -> None:
        if progress_callback is not None:
            progress_callback(message)

    scratch_context = _create_scratch_context(scratch_root, protomer_id)
    scratch_dir = scratch_context.scratch_dir
    log_paths = scratch_context.log_paths

    if protomer.mol is None:
        raise ValueError("Protomer does not have mol; cannot run workflow. Either this charge type is not suppported, or your SMILES input is invalid.")
    charge = int(charge_override) if charge_override is not None else _formal_charge(protomer.mol)
    _log_status(
        log_paths,
        "START",
        f"protomer_id={protomer_id} scratch_dir={scratch_dir.name} charge={charge} conformer_mode={conformer_mode}",
    )
    _progress("preparing conformer")

    conformer_energy_kcal_mol: Optional[float] = None
    xtbopt_xyz_path: Optional[Path] = None
    xtbopt_xyz_block: Optional[str] = None
    solvation_free_energy_kcal_mol: Optional[float] = None
    gas_sp_energy_kcal_mol: Optional[float] = None
    rrho_contribution_kcal_mol: Optional[float] = None
    solution_phase_free_energy_kcal_mol: Optional[float] = None

    try:

        if dry_run:
            _progress("dry run enabled; skipping optimization workflow")
            _log_status(log_paths, "SKIP", "dry_run enabled; skipping xTB/g-xTB steps")
            _set_mol_prop_double(
                protomer.mol,
                "peace_conformer_energy_kcal_mol",
                conformer_energy_kcal_mol,
            )
            protomer.mol.SetProp("peace_charge", str(charge))
            return SolvationWorkflowResult(
                conformer_energy_kcal_mol=conformer_energy_kcal_mol,
                xtb_optimized_xyz=None,
                solvation_free_energy_kcal_mol=None,
                gas_sp_energy_kcal_mol=None,
                rrho_contribution_kcal_mol=None,
                solution_phase_free_energy_kcal_mol=None,
                stdout_tail="dry_run; skipped confgen/opt/energy steps.",
            )

        mol, conformer_energy_kcal_mol = _prepare_protomer_conformer(
            protomer,
            conformer_mode=conformer_mode,
            external_xyz_path=external_xyz_path,
            log_paths=log_paths,
        )

        input_xyz_path = _write_workflow_inputs(mol, scratch_dir, charge, log_paths)
        active_xyz_path = input_xyz_path
        gas_sp_hess_kcal_mol: Optional[float] = None

        if run_geometry_optimization:
            _progress("optimizing geometry")
            xtbopt_xyz_path, _opt_gas_sp_kcal_mol, _opt_gas_sp_h = _run_xtb_optimization(
                mol=mol,
                scratch_dir=scratch_dir,
                input_xyz_path=input_xyz_path,
                xtb_executable=xtb_executable,
                solvent=solvent,
                opt_level=opt_level,
                timeout_s=timeout_s,
                dry_run=dry_run,
                log_paths=log_paths,
            )
            active_xyz_path = xtbopt_xyz_path
            xtbopt_xyz_block = _update_protomer_geometry_from_xyz(protomer, xtbopt_xyz_path, log_paths)

        if reuse_screening_terms and protomer.mol.HasProp("peace_screening_solvation_free_energy_kcal_mol"):
            solvation_free_energy_kcal_mol = protomer.mol.GetDoubleProp("peace_screening_solvation_free_energy_kcal_mol")
        if reuse_screening_terms and protomer.mol.HasProp("peace_screening_rrho_contribution_kcal_mol"):
            rrho_contribution_kcal_mol = protomer.mol.GetDoubleProp("peace_screening_rrho_contribution_kcal_mol")
        if reuse_screening_terms and protomer.mol.HasProp("peace_screening_gas_sp_energy_kcal_mol"):
            gas_sp_hess_kcal_mol = protomer.mol.GetDoubleProp("peace_screening_gas_sp_energy_kcal_mol")

        if recompute_solvation or solvation_free_energy_kcal_mol is None:
            _progress("computing solvation single point")
            solvation_free_energy_kcal_mol = _run_cpcmx_single_point(
                scratch_dir=scratch_dir,
                xtbopt_xyz_path=active_xyz_path,
                xtb_executable=xtb_executable,
                solvent=solvent,
                charge=charge,
                gfn=gfn,
                parse_solvation=parse_solvation,
                timeout_s=timeout_s,
                dry_run=dry_run,
                log_paths=log_paths,
            )

        if recompute_frequencies or rrho_contribution_kcal_mol is None or gas_sp_hess_kcal_mol is None:
            _progress("computing frequencies")
            gas_sp_hess_kcal_mol, rrho_contribution_kcal_mol, _gas_sp_hess_h = _run_hessian_and_parse_energies(
                scratch_dir=scratch_dir,
                xtbopt_xyz_path=active_xyz_path,
                xtb_executable=xtb_executable,
                charge=charge,
                gfn=gfn,
                timeout_s=timeout_s,
                dry_run=dry_run,
                log_paths=log_paths,
            )
        if sp_energy == "gxtb":
            _progress("computing gas-phase single point")
            gas_sp_energy_kcal_mol, _ = _run_gxtb_single_point_energy(
                scratch_dir=scratch_dir,
                xtbopt_xyz_path=active_xyz_path,
                xtb_executable=xtb_executable,
                charge=charge,
                timeout_s=timeout_s,
                dry_run=dry_run,
                log_paths=log_paths,
            )
        elif sp_energy == "xtb":
            gas_sp_energy_kcal_mol = gas_sp_hess_kcal_mol
        else:
            raise ValueError(f"Unknown sp_energy mode: {sp_energy}")

        solution_phase_free_energy_kcal_mol = _compute_solution_phase_energy(
            gas_sp_energy_kcal_mol,
            solvation_free_energy_kcal_mol,
            rrho_contribution_kcal_mol,
            log_paths,
        )
        _progress("finished optimization workflow")

    except Exception as e:
        _progress(f"failed: {e}")
        _log_status(log_paths, "FAIL", f"workflow exception for protomer_id={protomer_id}: {e}")
        warnings.warn(
            f"Solvation workflow failed for protomer_id={protomer_id}: {e}",
            RuntimeWarning,
        )
        _set_mol_prop_str(protomer.mol, "peace_workflow_error", str(e)[:4000] if protomer.mol is not None else str(e))

        solvation_free_energy_kcal_mol = None
        gas_sp_energy_kcal_mol = None
        rrho_contribution_kcal_mol = None
        solution_phase_free_energy_kcal_mol = None

        stdout_tail = str(e)

        preserved_xtbopt_path = _preserve_output_files(scratch_dir, keep_logs=keep_logs)
        keep = keep_scratch or keep_scratch_on_failure
        _cleanup_scratch_dir(
            scratch_dir,
            keep_scratch=keep,
            dry_run=dry_run,
            log_paths=log_paths,
            success=False,
        )

        return SolvationWorkflowResult(
            conformer_energy_kcal_mol=conformer_energy_kcal_mol,
            xtb_optimized_xyz=preserved_xtbopt_path if preserved_xtbopt_path is not None else xtbopt_xyz_path,
            solvation_free_energy_kcal_mol=None,
            gas_sp_energy_kcal_mol=None,
            rrho_contribution_kcal_mol=None,
            solution_phase_free_energy_kcal_mol=None,
            stdout_tail=stdout_tail[-4000:],
        )

    _persist_protomer_results(
        protomer,
        charge=charge,
        conformer_energy_kcal_mol=conformer_energy_kcal_mol,
        solvation_free_energy_kcal_mol=solvation_free_energy_kcal_mol,
        gas_sp_energy_kcal_mol=gas_sp_energy_kcal_mol,
        rrho_contribution_kcal_mol=rrho_contribution_kcal_mol,
        solution_phase_free_energy_kcal_mol=solution_phase_free_energy_kcal_mol,
    )

    final_xtbopt_path = _preserve_output_files(scratch_dir, keep_logs=keep_logs)
    _cleanup_scratch_dir(
        scratch_dir,
        keep_scratch=keep_scratch,
        dry_run=dry_run,
        log_paths=log_paths,
        success=True,
    )

    stdout_tail = f"ok; parsed values: solv={solvation_free_energy_kcal_mol}, gas={gas_sp_energy_kcal_mol}"
    _log_status(
        log_paths,
        "SUCCESS",
        f"protomer_id={protomer_id} gas={gas_sp_energy_kcal_mol} solv={solvation_free_energy_kcal_mol} "
        f"freq={rrho_contribution_kcal_mol} solution={solution_phase_free_energy_kcal_mol}",
    )
    _progress("success")

    return SolvationWorkflowResult(
        conformer_energy_kcal_mol=conformer_energy_kcal_mol,
        xtb_optimized_xyz=final_xtbopt_path if final_xtbopt_path is not None else xtbopt_xyz_path,
        solvation_free_energy_kcal_mol=solvation_free_energy_kcal_mol,
        gas_sp_energy_kcal_mol=gas_sp_energy_kcal_mol,
        rrho_contribution_kcal_mol=rrho_contribution_kcal_mol,
        solution_phase_free_energy_kcal_mol=solution_phase_free_energy_kcal_mol,
        stdout_tail=stdout_tail[-4000:],
    )


def run_tautomer_solvation(
    tautomer: Tautomer,
    *,
    tautomer_id: int | str = 0,
    species_key: Optional[str] = None,
    scratch_root: str | Path = "./peace_scratch_solvation",
    **kwargs,
) -> dict[int | str, SolvationWorkflowResult]:
    """
    Run the solvation workflow for all protomers contained in a tautomer.
    """
    results: dict[int | str, SolvationWorkflowResult] = {}
    # Namespace scratch per tautomer to keep debuggability.
    per_taut_scratch = Path(scratch_root) / f"tautomer_{tautomer_id}"
    per_taut_scratch.mkdir(parents=True, exist_ok=True)
    for prot_idx, protomer in tautomer.protomers.items():
        results[prot_idx] = run_protomer_solvation(
            protomer,
            protomer_id=str(prot_idx),
            scratch_root=per_taut_scratch,
            **kwargs,
        )
    return results


def run_species_solvation(
    species: Species,
    *,
    scratch_root: str | Path = "./peace_scratch_solvation",
    override_solvation: bool = False,
    **kwargs,
) -> dict[int | str, dict[int | str, SolvationWorkflowResult]]:
    """
    Run the solvation workflow for all tautomers, and thus their protomers.
    """
    results: dict[int | str, dict[int | str, SolvationWorkflowResult]] = {}
    scratch_root_path = Path(scratch_root)
    scratch_root_path.mkdir(parents=True, exist_ok=True)
    per_species_scratch = scratch_root_path / f"species_{species.key}"
    workflow_log = scratch_root_path / "peace.out"

    if per_species_scratch.exists():
        if override_solvation:
            warnings.warn(f"OVERRIDE: removing existing species folder before rerun: {per_species_scratch}")
            shutil.rmtree(per_species_scratch, ignore_errors=True)
        else:
            msg = (
                "Existing solvation results detected. Refusing to rerun by default. "
                f"Found existing species folder: {per_species_scratch}. "
                "Use --override-solvation to delete prior results and rerun."
            )
            _append_log(workflow_log, f"SKIP: {msg}")
            raise FileExistsError(msg)

    per_species_scratch.mkdir(parents=True, exist_ok=True)
    for taut_idx, tautomer in species.tautomers.items():
        results[taut_idx] = run_tautomer_solvation(
            tautomer,
            tautomer_id=str(taut_idx),
            scratch_root=per_species_scratch,
            **kwargs,
        )
    return results


def _build_cli_parser():
    import argparse

    p = argparse.ArgumentParser(description="PEACE solvation energy workflow (xTB + g-xTB).")
    p.add_argument("--smiles", type=str, default=None, help="Run the workflow on this single SMILES.")
    p.add_argument("--external-xyz", type=str, default=None, help="Use external xyz for conformer mode.")
    p.add_argument(
        "--conformer-mode",
        type=str,
        default="mmff94",
        choices=["mmff94", "external_xyz", "skip_search"],
    )
    p.add_argument("--charge", type=int, default=None, help="Override formal charge.")
    p.add_argument("--scratch-root", type=str, default="./peace_scratch_solvation")
    p.add_argument(
        "--xtb-executable",
        type=str,
        default="xtb",
        help="xTB executable used for optimization (with g-xTB driver), CPCM-X, and Hessian.",
    )
    p.add_argument("--keep-scratch", action="store_true", help="Keep xTB scratch directories.")
    p.add_argument(
        "--keep-logs",
        action="store_true",
        help="Preserve run stdout logs into a separate log folder after each run.",
    )
    p.add_argument(
        "--override-solvation",
        action="store_true",
        help="Override any existing species solvation folder results.",
    )
    p.add_argument("--dry-run", action="store_true", help="Print nothing; just skip execution.")
    p.add_argument(
        "--parse-solvation",
        type=str,
        default="g",
        choices=["g", "e"],
        help="Solvation energy parser mode: 'g' for dG_solv (default), 'e' for Gsolv.",
    )
    p.add_argument(
        "--sp-energy",
        type=str,
        default="gxtb",
        choices=["gxtb", "xtb"],
        help="Gas-phase SP source: 'gxtb' (driver SP, default) or 'xtb' (from Hessian output).",
    )
    p.add_argument("--hess-charge-mode", type=str, default="negate", choices=["as_is", "negate"])
    return p


def main_cli(argv: Optional[list[str]] = None) -> int:
    parser = _build_cli_parser()
    args = parser.parse_args(argv)

    if not args.smiles:
        parser.error("Please provide --smiles for the CLI.")

    protomer = Protomer.from_smiles(args.smiles)
    run_protomer_solvation(
        protomer,
        protomer_id="cli",
        scratch_root=args.scratch_root,
        conformer_mode=args.conformer_mode,
        external_xyz_path=args.external_xyz,
        charge_override=args.charge,
        parse_solvation=args.parse_solvation,
        sp_energy=args.sp_energy,
        keep_scratch=args.keep_scratch,
        keep_logs=args.keep_logs,
        dry_run=bool(args.dry_run),
        hess_charge_mode=args.hess_charge_mode,
    )

    # Print a minimal summary of what got attached to the mol.
    # RDKit may not have all props if parsing failed.
    summary_keys = [
        "peace_solution_phase_free_energy_kcal_mol",
        "peace_solvation_free_energy_kcal_mol",
        "peace_gas_sp_energy_kcal_mol",
        "peace_rrho_contribution_kcal_mol",
        "peace_conformer_energy_kcal_mol",
    ]
    for k in summary_keys:
        if protomer.mol.HasProp(k):
            print(f"{k}={protomer.mol.GetProp(k)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())

