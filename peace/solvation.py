import shutil
import subprocess
import warnings

from datetime import datetime

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Optional

from rdkit import Chem
from rdkit.Chem import AllChem, rdDetermineBonds, Descriptors

from .calculators import (
    cleanup_orca_refine_scratch_keep_log,
    run_aimnet2_optimization,
    run_aimnet2_single_point_energy,
    run_orca_cosmo_rs,
    run_cpcmx_single_point,
    run_gxtb_single_point_energy,
    run_hessian_and_parse_energies,
    run_skala_single_point_energy,
    run_xtb_optimization,
)
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


def _default_spin_multiplicity(mol: Chem.Mol) -> int:
    """
    Closed-shell default: 2S+1 with S inferred from radical electrons (RDKit).
    """
    r = int(Descriptors.NumRadicalElectrons(mol))
    return max(1, r + 1)


def refine_protomer_solvation_with_orca_cosmors(
    protomer: Protomer,
    *,
    scratch_dir: str | Path,
    solvent: str = "water",
    keep_scratch: bool = False,
    charge_override: Optional[int] = None,
    orca_executable: str = "orca",
    dry_run: bool = False,
    timeout_s: Optional[int] = None,
    log_paths: Optional[list[Path]] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Optional[float]:
    """
    Run ORCA openCOSMO-RS on the current 3D geometry (e.g. post–xTB optimization),
    update solvation and solution-phase free energies, and keep the prior
    CPCM-X value as ``solvation_free_energy_cpcmx_kcal_mol`` when replacing
    ``solvation_free_energy_kcal_mol``.

    Coordinates in the ORCA input are taken from RDKit mol conformer coordinates.
    """
    def _progress(message: str) -> None:
        if progress_callback is not None:
            progress_callback(message)

    if protomer.mol is None:
        raise ValueError("Protomer.mol is None; cannot run ORCA COSMO-RS refine.")
    mol = protomer.mol
    scratch = Path(scratch_dir)
    scratch.mkdir(parents=True, exist_ok=True)
    lp = log_paths if log_paths is not None else [_species_workflow_log_path(scratch)]

    charge = int(charge_override) if charge_override is not None else _formal_charge(mol)
    mult = _default_spin_multiplicity(mol)
    _progress(f"ORCA COSMO-RS (charge={charge}, mult={mult})")

    if mol.HasProp("solvation_free_energy_kcal_mol") and not mol.HasProp("solvation_free_energy_cpcmx_kcal_mol"):
        try:
            prev = float(mol.GetProp("solvation_free_energy_kcal_mol"))
            _set_mol_prop_double(mol, "solvation_free_energy_cpcmx_kcal_mol", prev)
        except ValueError:
            pass

    dgsolv, gas_sp_orca_h, _merged = run_orca_cosmo_rs(
        mol=mol,
        scratch_dir=scratch,
        charge=charge,
        multiplicity=mult,
        solvent=solvent,
        orca_executable=orca_executable,
        timeout_s=timeout_s,
        dry_run=dry_run,
        log_paths=lp,
        run_command=_run,
        log_status=_log_status,
    )
    cleanup_orca_refine_scratch_keep_log(
        scratch,
        keep_scratch=keep_scratch,
        log_paths=lp,
        log_status=_log_status,
    )
    if dgsolv is None:
        return None

    _set_mol_prop_double(mol, "solvation_free_energy_cosmors_kcal_mol", dgsolv)
    _set_mol_prop_double(mol, "solvation_free_energy_kcal_mol", dgsolv)
    mol.SetProp("solvation_refined_cosmors", "true")

    gas_sp_energy_kcal_mol: Optional[float] = None
    if mol.HasProp("gas_sp_energy_kcal_mol"):
        gas_sp_energy_kcal_mol = float(mol.GetDoubleProp("gas_sp_energy_kcal_mol"))
        _set_mol_prop_double(mol, "gas_sp_energy_gxtb_kcal_mol", gas_sp_energy_kcal_mol)

    gas_sp_energy_bp86_kcal_mol: Optional[float] = None
    if gas_sp_orca_h is not None:
        gas_sp_energy_bp86_kcal_mol = gas_sp_orca_h * HARTREE_TO_KCAL_MOL
        _set_mol_prop_double(mol, "gas_sp_energy_bp86_kcal_mol", gas_sp_energy_bp86_kcal_mol)
        # Replace the active gas SP term used in final solution free energy.
        gas_sp_energy_kcal_mol = gas_sp_energy_bp86_kcal_mol
        _set_mol_prop_double(mol, "gas_sp_energy_kcal_mol", gas_sp_energy_kcal_mol)
        _log_status(lp, "OK", f"using BP86 gas-phase SP for refined solution energy: {gas_sp_energy_bp86_kcal_mol}")
    else:
        _log_status(
            lp,
            "WARN",
            "ORCA gas-phase SP unavailable; keeping existing gas_sp_energy_kcal_mol value",
        )
    rrho_contribution_kcal_mol: Optional[float] = None
    if mol.HasProp("rrho_contribution_kcal_mol"):
        rrho_contribution_kcal_mol = float(mol.GetDoubleProp("rrho_contribution_kcal_mol"))

    solution_phase = _compute_solution_phase_energy(
        gas_sp_energy_kcal_mol,
        dgsolv,
        rrho_contribution_kcal_mol,
        lp,
    )
    _set_mol_prop_double(mol, "solution_phase_free_energy_kcal_mol", solution_phase)
    _progress("ORCA COSMO-RS refine done")
    return solution_phase


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
    log_paths: list[Path]


def _species_workflow_log_path(path: Path) -> Path:
    """
    Return canonical species-level workflow log path.
    If an ancestor directory named 'species_*' exists, use that folder;
    otherwise fall back to the provided path.
    """
    p = Path(path)
    for anc in [p] + list(p.parents):
        if anc.name.startswith("species_"):
            return anc / "peace.out"
    return p / "peace.out"


def _create_scratch_context(scratch_root: str | Path, protomer_id: int | str) -> SolvationScratchContext:
    scratch_root_path = Path(scratch_root)
    scratch_root_path.mkdir(parents=True, exist_ok=True)
    workflow_log = _species_workflow_log_path(scratch_root_path)

    scratch_dir = scratch_root_path / f"protomer_{protomer_id}"
    scratch_dir.mkdir(parents=True, exist_ok=False)

    return SolvationScratchContext(
        scratch_root=scratch_root_path,
        scratch_dir=scratch_dir,
        workflow_log=workflow_log,
        log_paths=[workflow_log],
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
        protomer.mol.SetProp("conformer_mode", "mmff94")
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
        protomer.mol.SetProp("conformer_mode", "external_xyz")
        _log_status(log_paths, "OK", f"loaded external xyz from {external_xyz_path}")
        return protomer.mol, conformer_energy_kcal_mol

    if conformer_mode == "skip_search":
        if mol.GetNumConformers() == 0:
            raise ValueError("conformer_mode='skip_search' but molecule has no conformers.")
        protomer.mol = mol
        if getattr(protomer, "input_mol", None) is None:
            protomer.input_mol = Chem.Mol(mol)
        protomer.mol.SetProp("conformer_mode", "skip_search")
        _log_status(log_paths, "OK", "using existing conformer on protomer.mol")
        return protomer.mol, conformer_energy_kcal_mol

    raise ValueError(f"Unknown conformer_mode: {conformer_mode}")


def _write_workflow_inputs(mol: Chem.Mol, scratch_dir: Path, charge: int, log_paths: list[Path]) -> Path:
    input_xyz_path = scratch_dir / "input.xyz"
    _write_xyz(mol, input_xyz_path, conf_id=0)
    _log_status(log_paths, "OK", f"wrote input geometry to {input_xyz_path.name}")
    return input_xyz_path


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


def _update_protomer_geometry_from_xyz(
    protomer: Protomer,
    xyz_path: Path,
    log_paths: list[Path],
) -> tuple[Optional[str], bool]:
  # build a bonded graph from optimized xyz coordinates for sanity checking.
  # tuple[Optional[str], bool]: the first element is the xyz text, the second element is a boolean indicating whether there was a connectivity mismatch.
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
            mol_opt.SetProp("connectivity_mismatch", "true")
            mol_opt.SetProp(
                "connectivity_mismatch_error",
                (
                    "Optimized structure connectivity differs from input mol. "
                    f"Got: {sorted(opt_edges)}, Expected: {sorted(input_edges)}"
                )[:4000],
            )
            warnings.warn(
                f"Optimized structure connectivity differs from input mol! Using unoptimized geometry instead (this result is less trustworthy!) Got: {opt_edges}, Expected: {input_edges}",
                RuntimeWarning,
            )
            _log_status(
                log_paths,
                "WARN",
                "optimized connectivity does not match input connectivity -- using unoptimized geometry. This could cause errors!",
            )
            if protomer.mol is not None:
                protomer.mol.SetProp("connectivity_mismatch", "true")
                protomer.mol.SetProp(
                    "connectivity_mismatch_error",
                    (
                        "Optimized structure connectivity differs from input mol. "
                        f"Got: {sorted(opt_edges)}, Expected: {sorted(input_edges)}"
                    )[:4000],
                )
            # fallback: keep pre-optimization graph/geometry attached to protomer.
            return xyz_path.read_text(), True
        else:
            mol_opt.SetProp("connectivity_mismatch", "false")
            if protomer.mol is not None:
                protomer.mol.SetProp("connectivity_mismatch", "false")
        protomer.mol = mol_opt
        _log_status(log_paths, "OK", f"updated protomer geometry from {xyz_path.name}")
        return xyz_path.read_text(), False
    _log_status(
        log_paths,
        "WARN",
        f"RDKit MolFromXYZFile did not yield a conformer for {xyz_path}; "
        "mol coordinates may be stale — downstream xTB/ORCA still use the xyz file on disk when provided.",
    )
    return None, False

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
    protomer.mol.SetProp("charge", str(charge))
    _set_mol_prop_double(protomer.mol, "conformer_energy_kcal_mol", conformer_energy_kcal_mol)
    _set_mol_prop_double(protomer.mol, "solvation_free_energy_kcal_mol", solvation_free_energy_kcal_mol)
    _set_mol_prop_double(protomer.mol, "gas_sp_energy_kcal_mol", gas_sp_energy_kcal_mol)
    _set_mol_prop_double(
        protomer.mol,
        "rrho_contribution_kcal_mol",
        rrho_contribution_kcal_mol,
    )
    _set_mol_prop_double(
        protomer.mol,
        "solution_phase_free_energy_kcal_mol",
        solution_phase_free_energy_kcal_mol,
    )


def _preserve_output_files(
    scratch_dir: Path,
    *,
    keep_logs: bool = False,
) -> Optional[Path]:
    if not scratch_dir.exists():
        return None

    log_paths = [_species_workflow_log_path(scratch_dir.parent)]
    preserved_dir = scratch_dir.parent / "xyz"
    preserved_dir.mkdir(parents=True, exist_ok=True)

    preserved_opt_path: Optional[Path] = None
    files_to_preserve = [
        "input.xyz",
        "xtbopt.xyz",
        "aimnet2opt.xyz",
        "xtbopt.log",
    ]
    for file_name in files_to_preserve:
        src = scratch_dir / file_name
        if not src.exists():
            continue
        dst = preserved_dir / f"{scratch_dir.name}_{file_name}"
        shutil.copy2(src, dst)
        _log_status(log_paths, "KEEP", f"preserved {file_name} at {dst}")
        if file_name in ("xtbopt.xyz", "aimnet2opt.xyz"):
            preserved_opt_path = dst
    if keep_logs:
        preserved_log_dir = scratch_dir.parent / "log"
        preserved_log_dir.mkdir(parents=True, exist_ok=True)
        log_files_to_preserve = [
            "xtbopt_run.log",
            "gxtbsp_run.log",
            "xtbsolv_run.log",
            "xtbfreq_run.log",
            "aimnet2opt_run.log",
            "aimnet2sp_run.log",
            "skalasp_run.log",
            "orca_cosmo_run.log",
        ]
        for file_name in log_files_to_preserve:
            src = scratch_dir / file_name
            if not src.exists():
                continue
            dst = preserved_log_dir / f"{scratch_dir.name}_{file_name}"
            shutil.copy2(src, dst)
            _log_status(log_paths, "KEEP", f"preserved {file_name} at {dst}")

    return preserved_opt_path


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
    scratch_root: str | Path = "./scratch_solvation",
    conformer_mode: Literal["mmff94", "external_xyz", "skip_search"] = "mmff94",
    external_xyz_path: Optional[str | Path] = None,
    charge_override: Optional[int] = None,
    solvent: Literal["water"] = "water",
    gfn: int = 2,
    optimization_engine: Literal["xtb", "aimnet2"] = "xtb",
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
    optimization_energy_kcal_mol: Optional[float] = None
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
        if optimization_engine == "xtb":
            opt_xyz_path, _opt_gas_sp_kcal_mol, _opt_gas_sp_h = run_xtb_optimization(
                mol=mol,
                scratch_dir=scratch_dir,
                input_xyz_path=input_xyz_path,
                xtb_executable=xtb_executable,
                solvent=solvent,
                opt_level=opt_level,
                charge=charge,
                timeout_s=timeout_s,
                dry_run=dry_run,
                log_paths=log_paths,
                run_command=_run,
                log_status=_log_status,
            )
            optimization_energy_kcal_mol = _opt_gas_sp_kcal_mol
        elif optimization_engine == "aimnet2":
            opt_xyz_path, _opt_gas_sp_kcal_mol, _opt_gas_sp_h = run_aimnet2_optimization(
                scratch_dir=scratch_dir,
                input_xyz_path=input_xyz_path,
                charge=charge,
                dry_run=dry_run,
                log_paths=log_paths,
                log_status=_log_status,
            )
            optimization_energy_kcal_mol = _opt_gas_sp_kcal_mol

        _set_mol_prop_str(protomer.mol, "screening_optimization_engine", optimization_engine)
        _set_mol_prop_double(
            protomer.mol,
            "screening_optimization_energy_kcal_mol",
            optimization_energy_kcal_mol,
        )
        _, has_connectivity_mismatch = _update_protomer_geometry_from_xyz(
            protomer,
            opt_xyz_path,
            log_paths,
        )
        active_xyz_path = input_xyz_path if has_connectivity_mismatch else opt_xyz_path
        if has_connectivity_mismatch:
            _log_status(
                log_paths,
                "WARN",
                f"connectivity mismatch detected; running subsequent screening steps with pre-optimized geometry {input_xyz_path.name}",
            )

        _progress("computing screening solvation single point")
        solvation_free_energy_kcal_mol = run_cpcmx_single_point(
            scratch_dir=scratch_dir,
            xyz_path=active_xyz_path,
            xtb_executable=xtb_executable,
            solvent=solvent,
            charge=charge,
            gfn=gfn,
            timeout_s=timeout_s,
            dry_run=dry_run,
            log_paths=log_paths,
            run_command=_run,
            log_status=_log_status,
        )

        _progress("computing screening frequencies")
        gas_sp_energy_kcal_mol, rrho_contribution_kcal_mol, _ = run_hessian_and_parse_energies(
            scratch_dir=scratch_dir,
            xyz_path=active_xyz_path,
            xtb_executable=xtb_executable,
            charge=charge,
            gfn=gfn,
            timeout_s=timeout_s,
            dry_run=dry_run,
            log_paths=log_paths,
            run_command=_run,
            log_status=_log_status,
        )

        solution_phase_free_energy_kcal_mol = _compute_solution_phase_energy(
            gas_sp_energy_kcal_mol,
            solvation_free_energy_kcal_mol,
            rrho_contribution_kcal_mol,
            log_paths,
        )
        _set_mol_prop_double(protomer.mol, "screening_conformer_energy_kcal_mol", conformer_energy_kcal_mol)
        _set_mol_prop_double(protomer.mol, "screening_solvation_free_energy_kcal_mol", solvation_free_energy_kcal_mol)
        _set_mol_prop_double(protomer.mol, "screening_gas_sp_energy_kcal_mol", gas_sp_energy_kcal_mol)
        _set_mol_prop_double(protomer.mol, "screening_rrho_contribution_kcal_mol", rrho_contribution_kcal_mol)
        _set_mol_prop_double(protomer.mol, "screening_solution_phase_free_energy_kcal_mol", solution_phase_free_energy_kcal_mol)
        _progress("finished screening workflow")

    except Exception as e:
        _progress(f"failed screening: {e}")
        _log_status(log_paths, "FAIL", f"screening exception for protomer_id={protomer_id}: {e}")
        warnings.warn(
            f"Screening workflow failed for protomer_id={protomer_id}: {e}",
            RuntimeWarning,
        )
        if protomer.mol is not None:
            _set_mol_prop_str(protomer.mol, "screening_error", str(e)[:4000])
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

    _preserved_xtbopt = _preserve_output_files(scratch_dir, keep_logs=keep_logs)
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
    scratch_root: str | Path = "./scratch_solvation",
    conformer_mode: Literal["mmff94", "external_xyz", "skip_search"] = "skip_search",
    external_xyz_path: Optional[str | Path] = None,
    optimization_engine: Literal["xtb", "aimnet2"] = "xtb",
    charge_override: Optional[int] = None,
    solvent: Literal["water"] = "water",
    gfn: int = 2,
    opt_level: Literal["loose", "tight", "vtight"] = "loose",
    xtb_executable: str = "xtb",
    sp_energy: Literal["gxtb", "xtb", "skala", "aimnet2"] = "gxtb",
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
    2) Run 'refined' gas-phase SP (depending on provided engine) on current geometry.
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
    _progress(f"preparing conformer: {conformer_mode}")

    conformer_energy_kcal_mol: Optional[float] = None
    opt_xyz_path: Optional[Path] = None
    solvation_free_energy_kcal_mol: Optional[float] = None
    gas_sp_energy_kcal_mol: Optional[float] = None
    rrho_contribution_kcal_mol: Optional[float] = None
    solution_phase_free_energy_kcal_mol: Optional[float] = None

    try:

        if dry_run:
            _progress("dry run enabled; skipping solvation workflow")
            _log_status(log_paths, "SKIP", "dry_run enabled; skipping xTB/g-xTB steps")
            _set_mol_prop_double(
                protomer.mol,
                "conformer_energy_kcal_mol",
                conformer_energy_kcal_mol,
            )
            protomer.mol.SetProp("charge", str(charge))
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

        input_xyz_path: Optional[Path] = None
        gas_sp_hess_kcal_mol: Optional[float] = None

        input_xyz_path = _write_workflow_inputs(mol, scratch_dir, charge, log_paths)
        active_xyz_path = input_xyz_path

        # reuse the energies from the previous screening step if applicable
        if reuse_screening_terms:
            solvation_free_energy_kcal_mol = (
                protomer.mol.GetDoubleProp("screening_solvation_free_energy_kcal_mol")
                if protomer.mol.HasProp("screening_solvation_free_energy_kcal_mol") else solvation_free_energy_kcal_mol
            )
            rrho_contribution_kcal_mol = (
                protomer.mol.GetDoubleProp("screening_rrho_contribution_kcal_mol")
                if protomer.mol.HasProp("screening_rrho_contribution_kcal_mol") else rrho_contribution_kcal_mol
            )
            gas_sp_hess_kcal_mol = (
                protomer.mol.GetDoubleProp("screening_gas_sp_energy_kcal_mol")
                if protomer.mol.HasProp("screening_gas_sp_energy_kcal_mol") else gas_sp_hess_kcal_mol
            )
       
        # compute DGsolv only when requested or unavailable from screening.
        if recompute_solvation or solvation_free_energy_kcal_mol is None:
            _progress("computing solvation single point")
            solvation_free_energy_kcal_mol = run_cpcmx_single_point(
                scratch_dir=scratch_dir,
                xyz_path=active_xyz_path,
                xtb_executable=xtb_executable,
                solvent=solvent,
                charge=charge,
                gfn=gfn,
                timeout_s=timeout_s,
                dry_run=dry_run,
                log_paths=log_paths,
                run_command=_run,
                log_status=_log_status,
            )
        else:
            _log_status(
                log_paths,
                "OK",
                f"reusing screening solvation energy (kcal/mol)={solvation_free_energy_kcal_mol}",
            )
            _progress("reusing screening solvation single point")

        # compute RRHO term
        if recompute_frequencies or rrho_contribution_kcal_mol is None or gas_sp_hess_kcal_mol is None:
            _progress("computing frequencies")
            gas_sp_hess_kcal_mol, rrho_contribution_kcal_mol, _gas_sp_hess_h = run_hessian_and_parse_energies(
                scratch_dir=scratch_dir,
                xyz_path=active_xyz_path,
                xtb_executable=xtb_executable,
                charge=charge,
                gfn=gfn,
                timeout_s=timeout_s,
                dry_run=dry_run,
                log_paths=log_paths,
                run_command=_run,
                log_status=_log_status,
            )

        # If screening optimization and requested SP use the same engine,
        # reuse the optimization energy as SP when available.
        reused_optimization_sp = False
        if sp_energy == optimization_engine and reuse_screening_terms:
            if (
                protomer.mol.HasProp("screening_optimization_engine")
                and protomer.mol.GetProp("screening_optimization_engine") == optimization_engine
                and protomer.mol.HasProp("screening_optimization_energy_kcal_mol")
            ):
                gas_sp_energy_kcal_mol = protomer.mol.GetDoubleProp("screening_optimization_energy_kcal_mol")
                reused_optimization_sp = True
                _log_status(
                    log_paths,
                    "OK",
                    f"reusing screening optimization energy as SP (engine={optimization_engine}) "
                    f"gas_sp_energy_kcal_mol={gas_sp_energy_kcal_mol}",
                )
                _progress(f"reusing optimization energy for {optimization_engine} SP")

        # compute SP energy using a pluggable strategy map so additional
        # calculators can be added with minimal wiring.
        def _sp_energy_from_gxtb() -> Optional[float]:
            _progress("computing gas-phase single point")
            value_kcal_mol, _ = run_gxtb_single_point_energy(
                scratch_dir=scratch_dir,
                xyz_path=active_xyz_path,
                xtb_executable=xtb_executable,
                charge=charge,
                timeout_s=timeout_s,
                dry_run=dry_run,
                log_paths=log_paths,
                run_command=_run,
                log_status=_log_status,
            )
            return value_kcal_mol

        def _sp_energy_from_xtb_hessian() -> Optional[float]:
            return gas_sp_hess_kcal_mol

        def _sp_energy_from_skala() -> Optional[float]:
            _progress("computing gas-phase single point (Skala)")
            multiplicity = _default_spin_multiplicity(protomer.mol)
            value_kcal_mol, _ = run_skala_single_point_energy(
                scratch_dir=scratch_dir,
                xyz_path=active_xyz_path,
                charge=charge,
                multiplicity=multiplicity,
                dry_run=dry_run,
                log_paths=log_paths,
                log_status=_log_status,
            )
            return value_kcal_mol

        def _sp_energy_from_aimnet2() -> Optional[float]:
            _progress("computing gas-phase single point (AIMNet2)")
            value_kcal_mol, _ = run_aimnet2_single_point_energy(
                scratch_dir=scratch_dir,
                xyz_path=active_xyz_path,
                charge=charge,
                dry_run=dry_run,
                log_paths=log_paths,
                log_status=_log_status,
            )
            return value_kcal_mol

        sp_energy_strategies: dict[str, Callable[[], Optional[float]]] = {
            "gxtb": _sp_energy_from_gxtb,
            "xtb": _sp_energy_from_xtb_hessian,
            "skala": _sp_energy_from_skala,
            "aimnet2": _sp_energy_from_aimnet2,
        }
        if not reused_optimization_sp:
            try:
                gas_sp_energy_kcal_mol = sp_energy_strategies[sp_energy]()
            except KeyError as exc:
                raise ValueError(f"Unknown sp_energy mode: {sp_energy}") from exc

        # compute solution-phase free energy by adding everything together
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
        _set_mol_prop_str(protomer.mol, "workflow_error", str(e)[:4000] if protomer.mol is not None else str(e))

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
            xtb_optimized_xyz=preserved_xtbopt_path if preserved_xtbopt_path is not None else opt_xyz_path,
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
        xtb_optimized_xyz=final_xtbopt_path if final_xtbopt_path is not None else opt_xyz_path,
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
    scratch_root: str | Path = "./scratch_solvation",
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
    scratch_root: str | Path = "./scratch_solvation",
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
