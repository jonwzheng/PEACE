import re
import shutil
import shlex
import subprocess
import warnings
import uuid
from datetime import datetime

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from rdkit import Chem
from rdkit.Chem import AllChem

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
    n_confs: int = 100,
    random_seed: int = 0,
    max_attempts: int = 2000,
    mmff_max_iters: int = 500,
) -> tuple[Chem.Mol, float, int]:
    """
    Generate conformers with RDKit ETKDG and optimize with MMFF94.

    Returns: (mol_no_h, best_energy_kcal_mol, best_conf_id)
    """
    if mol is None:
        raise ValueError("mol is None")

    mol_in = Chem.Mol(mol)
    mol_h = Chem.AddHs(mol_in, addCoords=False)

    # Use RDKit's ETKDGv3 parameters for robust conformer generation.
    params = AllChem.ETKDGv3()
    params.randomSeed = int(random_seed)
    params.numThreads = 0
    params.maxAttempts = int(max_attempts)
    params.useExpTorsionAnglePrefs = False

    # Use the (mol, numConfs, params) overload; kwargs not supported
    conf_ids = list(AllChem.EmbedMultipleConfs(mol_h, int(n_confs), params))
    if not conf_ids:
        raise RuntimeError("RDKit conformer embedding produced no conformers.")

    best_energy = float("inf")
    best_conf_id = conf_ids[0]

    for conf_id in conf_ids:
        # Optimize this conformer in-place using RDKit's built-in MMFF optimizer.
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


def _parse_xtb_total_energy_hartree(text: str) -> Optional[float]:
    float_re = _float_regex()
    patterns = [
        rf"TOTAL ENERGY[^\S\r\n]*[:=]?[^\S\r\n]*({float_re})",
        rf"Total energy[^\S\r\n]*[:=]?[^\S\r\n]*({float_re})",
        rf"total energy[^\S\r\n]*[:=]?[^\S\r\n]*({float_re})",
    ]
    return _parse_first_float(patterns, text)


def _parse_xtb_total_free_energy_hartree(text: str) -> Optional[float]:
    float_re = _float_regex()
    patterns = [
        rf"TOTAL FREE ENERGY[^\S\r\n]*[:=]?[^\S\r\n]*({float_re})",
        rf"Total free energy[^\S\r\n]*[:=]?[^\S\r\n]*({float_re})",
        rf"total free energy[^\S\r\n]*[:=]?[^\S\r\n]*({float_re})",
    ]
    return _parse_first_float(patterns, text)


def _parse_xtb_solvent_free_energy_hartree(text: str) -> Optional[float]:
    float_re = _float_regex()
    patterns = [
        # Common phrasing variants in xTB-like outputs.
        rf"free energy of solvation[^\S\r\n]*[:=]?[^\S\r\n]*({float_re})",
        rf"Free energy of solvation[^\S\r\n]*[:=]?[^\S\r\n]*({float_re})",
        rf"solvation free energy[^\S\r\n]*[:=]?[^\S\r\n]*({float_re})",
        rf"cpcm[xX][^\S\r\n]*.*free energy[^\S\r\n]*[:=]?[^\S\r\n]*({float_re})",
        rf"Delta\s*G.*solv[^\S\r\n]*[:=]?[^\S\r\n]*({float_re})",
        rf"Gsolv[^\S\r\n]*[:=]?[^\S\r\n]*({float_re})",
    ]
    return _parse_first_float(patterns, text)


def _parse_xtb_zpe_hartree(text: str) -> Optional[float]:
    float_re = _float_regex()
    patterns = [
        rf"Zero[-\s]*point energy[^\S\r\n]*[:=]?[^\S\r\n]*({float_re})",
        rf"Zero point energy[^\S\r\n]*[:=]?[^\S\r\n]*({float_re})",
        rf"ZPE[^\S\r\n]*[:=]?[^\S\r\n]*({float_re})",
    ]
    return _parse_first_float(patterns, text)


def _parse_xtb_thermal_gibbs_correction_hartree(text: str) -> Optional[float]:
    float_re = _float_regex()
    patterns = [
        rf"Thermal correction to Gibbs Free Energy[^\S\r\n]*[:=]?[^\S\r\n]*({float_re})",
        rf"Thermal correction to Gibbs free energy[^\S\r\n]*[:=]?[^\S\r\n]*({float_re})",
        rf"thermal correction to Gibbs free energy[^\S\r\n]*[:=]?[^\S\r\n]*({float_re})",
        rf"Gibbs free energy[^\S\r\n]*correction[^\S\r\n]*[:=]?[^\S\r\n]*({float_re})",
    ]
    return _parse_first_float(patterns, text)


@dataclass(frozen=True)
class SolvationWorkflowResult:
    conformer_energy_kcal_mol: Optional[float]
    gxtb_optimized_xyz: Optional[Path]
    solvation_free_energy_kcal_mol: Optional[float]
    gas_sp_energy_kcal_mol: Optional[float]
    frequency_contribution_kcal_mol: Optional[float]
    solution_phase_free_energy_kcal_mol: Optional[float]
    stdout_tail: str


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
    opt_level: Literal["loose", "tight"] = "loose",
    xtb_executable: str = "xtb",
    keep_scratch: bool = False,
    keep_scratch_on_failure: bool = False,
    dry_run: bool = False,
    timeout_s: Optional[int] = None,
) -> SolvationWorkflowResult:
    """
    Implements the README "Solvated energy quick calculation workflow" for one Protomer.

    Steps:
    1) Generate conformers (RDKit/MMFF94 by default) and keep the lowest MMFF energy.
    2) g-xTB optimization with implicit solvent (xTB + gxtb driver).
    3) CPCM-X SP solvation energy using GFN2-xTB.
    4) Gas-phase frequency calculation using GFN2-xTB (--hess).
    """
    scratch_root = Path(scratch_root)
    scratch_root.mkdir(parents=True, exist_ok=True)
    workflow_log = scratch_root / "peace.out"

    # Naming: unique per protomer run to prevent collisions/overwrites
    # even if caller uses the same scratch_root for multiple protomers.
    unique_suffix = uuid.uuid4().hex[:8]
    scratch_dir = scratch_root / f"protomer_{protomer_id}_{unique_suffix}"
    scratch_dir.mkdir(parents=True, exist_ok=False)
    run_log = scratch_dir / "run.out"
    log_paths = [workflow_log, run_log]

    mol = protomer.mol
    if mol is None:
        raise ValueError("Protomer.mol is None; cannot run workflow.")

    charge = int(charge_override) if charge_override is not None else _formal_charge(mol)
    _log_status(
        log_paths,
        "START",
        f"protomer_id={protomer_id} scratch_dir={scratch_dir.name} charge={charge} conformer_mode={conformer_mode}",
    )

    conformer_energy_kcal_mol: Optional[float] = None
    xtbopt_xyz_path: Optional[Path] = None
    solvation_free_energy_kcal_mol: Optional[float] = None
    gas_sp_energy_kcal_mol: Optional[float] = None
    frequency_contribution_kcal_mol: Optional[float] = None
    solution_phase_free_energy_kcal_mol: Optional[float] = None

    try:
        # 1) Conformer search / input geometry selection.
        _log_status(log_paths, "STEP", "starting conformer preparation")
        if conformer_mode == "mmff94":
            mol_best, best_energy, best_conf_id = _embed_conformers_rdkit_mmff94(mol)
            conformer_energy_kcal_mol = best_energy
            mol = mol_best
            protomer.mol = mol
            protomer.mol.SetProp("peace_conformer_mode", "mmff94")
            _log_status(
                log_paths,
                "OK",
                f"rdkit conformer search complete best_conf_id={best_conf_id} best_energy_kcal_mol={best_energy}",
            )
        elif conformer_mode == "external_xyz":
            if external_xyz_path is None:
                raise ValueError("external_xyz_path must be provided for conformer_mode='external_xyz'.")
            mol = Chem.MolFromXYZFile(str(external_xyz_path))
            if mol is None:
                raise ValueError(f"RDKit failed to read xyz: {external_xyz_path}")
            if mol.GetNumConformers() == 0:
                raise ValueError("External xyz produced a molecule without conformers.")
            protomer.mol = mol
            protomer.mol.SetProp("peace_conformer_mode", "external_xyz")
            _log_status(log_paths, "OK", f"loaded external xyz from {external_xyz_path}")
        elif conformer_mode == "skip_search":
            if mol.GetNumConformers() == 0:
                raise ValueError("conformer_mode='skip_search' but molecule has no conformers.")
            protomer.mol = mol
            protomer.mol.SetProp("peace_conformer_mode", "skip_search")
            _log_status(log_paths, "OK", "using existing conformer on protomer.mol")
        else:
            raise ValueError(f"Unknown conformer_mode: {conformer_mode}")

        # Write xyz for xTB.
        input_xyz_path = scratch_dir / "input.xyz"
        _write_xyz(mol, input_xyz_path, conf_id=0)
        _log_status(log_paths, "OK", f"wrote input geometry to {input_xyz_path.name}")

        # README: write .CHRG with just contents of the integer charge.
        # xTB can also accept --chrg btw, but g-xTB can't righ tnow
        (scratch_dir / ".CHRG").write_text(str(charge))
        _log_status(log_paths, "OK", f"wrote .CHRG with charge={charge}")

        # In dry-run mode, do not attempt to execute xTB/g-xTB (no outputs will exist).
        if dry_run:
            _log_status(log_paths, "SKIP", "dry_run enabled; skipping xTB/g-xTB steps")
            _set_mol_prop_double(
                protomer.mol,
                "peace_conformer_energy_kcal_mol",
                conformer_energy_kcal_mol,
            )
            protomer.mol.SetProp("peace_charge", str(charge))
            return SolvationWorkflowResult(
                conformer_energy_kcal_mol=conformer_energy_kcal_mol,
                gxtb_optimized_xyz=None,
                solvation_free_energy_kcal_mol=None,
                gas_sp_energy_kcal_mol=None,
                frequency_contribution_kcal_mol=None,
                solution_phase_free_energy_kcal_mol=None,
                stdout_tail="dry_run; skipped xTB/g-xTB steps.",
            )

        # 2) g-xTB optimization with implicit solvent and loose coordinates.
        xtbopt_xyz_path = scratch_dir / "xtbopt.xyz"
        # Run this exactly in the shell form documented in the README:
        # xtb input.xyz --driver "gxtb -grad -c xtbdriver.xyz" --opt loose --alpb water
        cmd_opt = (
            f"{shlex.quote(xtb_executable)} {shlex.quote(input_xyz_path.name)} "
            f'--driver "gxtb -grad -c xtbdriver.xyz" '
            f"--opt {shlex.quote(opt_level)} --alpb {shlex.quote(solvent)}"
        )
        _log_status(log_paths, "STEP", f"running optimization: {cmd_opt}")
        cp_opt = _run(cmd_opt, cwd=scratch_dir, timeout_s=timeout_s, dry_run=dry_run)
        if cp_opt.returncode != 0:
            _log_status(
                log_paths,
                "FAIL",
                f"optimization failed returncode={cp_opt.returncode} stdout_tail={cp_opt.stdout[-1000:]} stderr_tail={cp_opt.stderr[-1000:]}",
            )
            raise RuntimeError(
                f"xTB optimization (gxtb driver) failed with code {cp_opt.returncode}.\n"
                f"stdout:\n{cp_opt.stdout[-4000:]}\n"
                f"stderr:\n{cp_opt.stderr[-4000:]}\n"
            )
        if not xtbopt_xyz_path.exists():
            _log_status(log_paths, "FAIL", "optimization finished but xtbopt.xyz was not produced")
            raise FileNotFoundError("Expected output xtbopt.xyz was not produced by xTB.")
        _log_status(log_paths, "OK", f"optimization produced {xtbopt_xyz_path.name}")

        # Update protomer geometry to the optimized geometry (keepable for debugging).
        mol_opt = Chem.MolFromXYZFile(str(xtbopt_xyz_path))
        if mol_opt is not None and mol_opt.GetNumConformers() > 0:
            protomer.mol = mol_opt
            _log_status(log_paths, "OK", "updated protomer geometry from xtbopt.xyz")

        # 3) CPCM-X SP solvation energy calculation using GFN2-xTB.
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

        solvation_free_energy_h = _parse_xtb_solvent_free_energy_hartree(cp_sp.stdout)
        if solvation_free_energy_h is None:
            # Try a fallback: sometimes xTB prints a "Total free energy" - treat it as solvation-related.
            # This is best-effort parsing; if it fails, we keep results as None.
            solvation_free_energy_h = _parse_xtb_total_free_energy_hartree(cp_sp.stdout)
        if solvation_free_energy_h is not None:
            solvation_free_energy_kcal_mol = solvation_free_energy_h * HARTREE_TO_KCAL_MOL
        _log_status(
            log_paths,
            "OK",
            f"parsed solvation_free_energy_kcal_mol={solvation_free_energy_kcal_mol}",
        )

        # Gas-phase SP energy is derived from the hess output (step 4).

        # 4) Gas-phase frequency calculation using GFN2-xTB.

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
        total_free_energy_h = _parse_xtb_total_free_energy_hartree(cp_hess.stdout)
        zpe_h = _parse_xtb_zpe_hartree(cp_hess.stdout)
        thermal_gibbs_h = _parse_xtb_thermal_gibbs_correction_hartree(cp_hess.stdout)

        if gas_sp_energy_h is not None:
            gas_sp_energy_kcal_mol = gas_sp_energy_h * HARTREE_TO_KCAL_MOL

        # Frequency contribution:
        # - Primary: ZPE + thermal Gibbs correction
        # - Fallback: (TOTAL FREE ENERGY - TOTAL ENERGY)
        freq_contrib_h: Optional[float] = None
        if zpe_h is not None and thermal_gibbs_h is not None:
            freq_contrib_h = zpe_h + thermal_gibbs_h
        elif gas_sp_energy_h is not None and total_free_energy_h is not None:
            freq_contrib_h = total_free_energy_h - gas_sp_energy_h

        if freq_contrib_h is not None:
            frequency_contribution_kcal_mol = freq_contrib_h * HARTREE_TO_KCAL_MOL

        if (
            gas_sp_energy_h is not None
            and solvation_free_energy_kcal_mol is not None
            and frequency_contribution_kcal_mol is not None
        ):
            solution_phase_free_energy_kcal_mol = (
                gas_sp_energy_kcal_mol
                + frequency_contribution_kcal_mol
                + solvation_free_energy_kcal_mol
            )
        _log_status(
            log_paths,
            "OK",
            "parsed energies "
            f"gas_sp_energy_kcal_mol={gas_sp_energy_kcal_mol} "
            f"frequency_contribution_kcal_mol={frequency_contribution_kcal_mol} "
            f"solution_phase_free_energy_kcal_mol={solution_phase_free_energy_kcal_mol}",
        )

    except Exception as e:
        _log_status(log_paths, "FAIL", f"workflow exception for protomer_id={protomer_id}: {e}")
        warnings.warn(
            f"Solvation workflow failed for protomer_id={protomer_id}: {e}",
            RuntimeWarning,
        )
        _set_mol_prop_str(protomer.mol, "peace_workflow_error", str(e)[:4000] if protomer.mol is not None else str(e))
        # If anything fails, follow README behavior: energies are None.
        solvation_free_energy_kcal_mol = None
        gas_sp_energy_kcal_mol = None
        frequency_contribution_kcal_mol = None
        solution_phase_free_energy_kcal_mol = None

        stdout_tail = str(e)

        keep = keep_scratch or keep_scratch_on_failure
        if not keep and scratch_dir.exists() and not dry_run:
            _log_status(log_paths, "CLEANUP", "removing scratch directory after failed run")
            shutil.rmtree(scratch_dir, ignore_errors=True)

        return SolvationWorkflowResult(
            conformer_energy_kcal_mol=conformer_energy_kcal_mol,
            gxtb_optimized_xyz=None,
            solvation_free_energy_kcal_mol=None,
            gas_sp_energy_kcal_mol=None,
            frequency_contribution_kcal_mol=None,
            solution_phase_free_energy_kcal_mol=None,
            stdout_tail=stdout_tail[-4000:],
        )

    # Persist energies on the protomer for later inspection.
    protomer.mol.SetProp("peace_charge", str(charge))
    _set_mol_prop_double(protomer.mol, "peace_conformer_energy_kcal_mol", conformer_energy_kcal_mol)
    _set_mol_prop_double(protomer.mol, "peace_solvation_free_energy_kcal_mol", solvation_free_energy_kcal_mol)
    _set_mol_prop_double(protomer.mol, "peace_gas_sp_energy_kcal_mol", gas_sp_energy_kcal_mol)
    _set_mol_prop_double(
        protomer.mol,
        "peace_frequency_contribution_kcal_mol",
        frequency_contribution_kcal_mol,
    )
    _set_mol_prop_double(
        protomer.mol,
        "peace_solution_phase_free_energy_kcal_mol",
        solution_phase_free_energy_kcal_mol,
    )

    # Cleanup.
    gxtb_optimized_xyz_to_report: Optional[Path]
    if keep_scratch or dry_run:
        gxtb_optimized_xyz_to_report = xtbopt_xyz_path
    else:
        gxtb_optimized_xyz_to_report = None

    if not keep_scratch and scratch_dir.exists() and not dry_run:
        _log_status(log_paths, "CLEANUP", "removing scratch directory after successful run")
        shutil.rmtree(scratch_dir, ignore_errors=True)

    stdout_tail = f"ok; parsed values: solv={solvation_free_energy_kcal_mol}, gas={gas_sp_energy_kcal_mol}"
    _log_status(
        log_paths,
        "SUCCESS",
        f"protomer_id={protomer_id} solv={solvation_free_energy_kcal_mol} gas={gas_sp_energy_kcal_mol} "
        f"freq={frequency_contribution_kcal_mol} solution={solution_phase_free_energy_kcal_mol}",
    )

    return SolvationWorkflowResult(
        conformer_energy_kcal_mol=conformer_energy_kcal_mol,
        gxtb_optimized_xyz=gxtb_optimized_xyz_to_report,
        solvation_free_energy_kcal_mol=solvation_free_energy_kcal_mol,
        gas_sp_energy_kcal_mol=gas_sp_energy_kcal_mol,
        frequency_contribution_kcal_mol=frequency_contribution_kcal_mol,
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
    **kwargs,
) -> dict[int | str, dict[int | str, SolvationWorkflowResult]]:
    """
    Run the solvation workflow for all tautomers, and thus their protomers.
    """
    results: dict[int | str, dict[int | str, SolvationWorkflowResult]] = {}
    per_species_scratch = Path(scratch_root) / f"species_{species.key}"
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
    p.add_argument("--keep-scratch", action="store_true", help="Keep xTB scratch directories.")
    p.add_argument("--dry-run", action="store_true", help="Print nothing; just skip execution.")
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
        keep_scratch=args.keep_scratch,
        dry_run=bool(args.dry_run),
        hess_charge_mode=args.hess_charge_mode,
    )

    # Print a minimal summary of what got attached to the mol.
    # RDKit may not have all props if parsing failed.
    summary_keys = [
        "peace_solution_phase_free_energy_kcal_mol",
        "peace_solvation_free_energy_kcal_mol",
        "peace_gas_sp_energy_kcal_mol",
        "peace_frequency_contribution_kcal_mol",
        "peace_conformer_energy_kcal_mol",
    ]
    for k in summary_keys:
        if protomer.mol.HasProp(k):
            print(f"{k}={protomer.mol.GetProp(k)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())

