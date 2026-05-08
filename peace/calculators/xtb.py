from __future__ import annotations

import shlex
import subprocess
import warnings
from pathlib import Path
from typing import Callable, Optional

from rdkit import Chem

from .common import HARTREE_TO_KCAL_MOL, float_regex, parse_last_float


def parse_xtb_total_energy_hartree(text: str) -> Optional[float]:
    float_re = float_regex()
    patterns = [
        rf"total energy[^\S\r\n]*[:=]?[^\S\r\n]*({float_re})",
    ]
    return parse_last_float(patterns, text)


def parse_xtb_solvent_free_energy_hartree(text: str) -> Optional[float]:
    float_re = float_regex()
    patterns = [
        rf"solvation free energy \(dG_solv\):[\ \t]*({float_re})",
    ]
    return parse_last_float(patterns, text)


def parse_xtb_rrho_contrib_hartree(text: str) -> Optional[float]:
    float_re = float_regex()
    patterns = [
        rf"G\(RRHO\)[^\S\r\n]*contrib\.?[^\S\r\n]*[:=]?[^\S\r\n]*({float_re})",
    ]
    return parse_last_float(patterns, text)


def run_xtb_optimization(
    *,
    mol: Chem.Mol,
    scratch_dir: Path,
    input_xyz_path: Path,
    xtb_executable: str,
    opt_level: str,
    charge: int,
    solvent: str,
    timeout_s: Optional[int],
    dry_run: bool,
    log_paths: list[Path],
    run_command: Callable[..., subprocess.CompletedProcess[str]],
    log_status: Callable[[list[Path], str, str], None],
) -> tuple[Path, Optional[float], Optional[float]]:
    xtbopt_xyz_path = scratch_dir / "xtbopt.xyz"
    xcontrol_path = scratch_dir / "xcontrol.inp"
    n_constraints = build_fix_xcontrol(
        mol,
        xcontrol_path,
        fixed_distance=1.01,
    )

    input_flag = ""
    if n_constraints > 0:
        input_flag = f" --input {shlex.quote(xcontrol_path.name)}"
        log_status(
            log_paths,
            "OK",
            "generated xcontrol constraints for positively charged heavy-atom X-H bonds: "
            f"n_constraints={n_constraints}",
        )
    else:
        log_status(log_paths, "OK", "no positively charged heavy-atom X-H constraints generated")

    cmd_opt = (
        f"{shlex.quote(xtb_executable)} {shlex.quote(input_xyz_path.name)} "
        f"{input_flag} "
        f"--opt {shlex.quote(opt_level)} --alpb {shlex.quote(solvent)} "
        f"--charge {shlex.quote(str(charge))}"
    )
    log_status(log_paths, "STEP", f"running optimization: {cmd_opt}")
    cp_opt = run_command(cmd_opt, cwd=scratch_dir, timeout_s=timeout_s, dry_run=dry_run)
    if not dry_run:
        optimization_log_path = scratch_dir / "xtbopt_run.log"
        optimization_log_path.write_text(cp_opt.stdout)
        log_status(log_paths, "OK", f"saved optimization stdout to {optimization_log_path.name}")
    if cp_opt.returncode != 0:
        log_status(
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
        log_status(log_paths, "FAIL", "optimization finished but xtbopt.xyz was not produced")
        raise FileNotFoundError("Expected output xtbopt.xyz was not produced by xTB.")

    log_status(log_paths, "OK", f"optimization produced {xtbopt_xyz_path.name}")

    parse_text = cp_opt.stdout
    xtbopt_log_path = scratch_dir / "xtbopt.log"
    if xtbopt_log_path.exists():
        parse_text += "\n" + xtbopt_log_path.read_text()

    gas_sp_energy_h = parse_xtb_total_energy_hartree(parse_text)
    gas_sp_energy_kcal_mol = None
    if gas_sp_energy_h is not None:
        gas_sp_energy_kcal_mol = gas_sp_energy_h * HARTREE_TO_KCAL_MOL
    else:
        warnings.warn(
            "Could not parse TOTAL ENERGY from optimization output/log. "
            "Gas-phase energy will remain None.",
            RuntimeWarning,
        )
        log_status(log_paths, "WARN", "failed to parse gas-phase SP energy from optimization output/log")

    return xtbopt_xyz_path, gas_sp_energy_kcal_mol, gas_sp_energy_h


def build_fix_xcontrol(mol: Chem.Mol, xcontrol_path: Path, *, fixed_distance: float = 1.05) -> int:
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


def run_cpcmx_single_point(
    *,
    scratch_dir: Path,
    xyz_path: Path,
    xtb_executable: str,
    solvent: str,
    charge: int,
    gfn: int,
    timeout_s: Optional[int],
    dry_run: bool,
    log_paths: list[Path],
    run_command: Callable[..., subprocess.CompletedProcess[str]],
    log_status: Callable[[list[Path], str, str], None],
    parse_solvation_energy_hartree: Callable[[str], Optional[float]] = parse_xtb_solvent_free_energy_hartree,
) -> Optional[float]:
    cmd_sp = [
        xtb_executable,
        str(xyz_path.name),
        "--cpcmx",
        solvent,
        "--chrg",
        str(charge),
        "--gfn",
        str(gfn),
    ]
    log_status(log_paths, "STEP", f"running CPCM-X SP: {' '.join(shlex.quote(x) for x in cmd_sp)}")
    cp_sp = run_command(cmd_sp, cwd=scratch_dir, timeout_s=timeout_s, dry_run=dry_run)
    if not dry_run:
        solvation_log_path = scratch_dir / "xtbsolv_run.log"
        solvation_log_path.write_text(cp_sp.stdout)
        log_status(log_paths, "OK", f"saved CPCM-X stdout to {solvation_log_path.name}")
    if cp_sp.returncode != 0:
        log_status(
            log_paths,
            "FAIL",
            f"CPCM-X SP failed returncode={cp_sp.returncode} stdout_tail={cp_sp.stdout[-1000:]} stderr_tail={cp_sp.stderr[-1000:]}",
        )
        raise RuntimeError(
            f"xTB CPCM-X SP calculation failed with code {cp_sp.returncode}.\n"
            f"stdout:\n{cp_sp.stdout[-4000:]}\n"
            f"stderr:\n{cp_sp.stderr[-4000:]}\n"
        )

    solvation_free_energy_h = parse_solvation_energy_hartree(cp_sp.stdout)
    solvation_free_energy_kcal_mol = None
    if solvation_free_energy_h is not None:
        solvation_free_energy_kcal_mol = solvation_free_energy_h * HARTREE_TO_KCAL_MOL

    log_status(
        log_paths,
        "OK",
        f"parsed solvation_free_energy_kcal_mol={solvation_free_energy_kcal_mol}",
    )
    return solvation_free_energy_kcal_mol


def run_gxtb_single_point_energy(
    *,
    scratch_dir: Path,
    xyz_path: Path,
    xtb_executable: str,
    charge: int,
    timeout_s: Optional[int],
    dry_run: bool,
    log_paths: list[Path],
    run_command: Callable[..., subprocess.CompletedProcess[str]],
    log_status: Callable[[list[Path], str, str], None],
) -> tuple[Optional[float], Optional[float]]:
    cmd_sp = (
        f"{shlex.quote(xtb_executable)} {shlex.quote(xyz_path.name)} "
        f'--driver "gxtb -grad -c xtbdriver.xyz" '
        f"--chrg {shlex.quote(str(charge))}"
    )
    log_status(log_paths, "STEP", f"running g-xTB gas-phase SP via driver: {cmd_sp}")
    cp_sp = run_command(cmd_sp, cwd=scratch_dir, timeout_s=timeout_s, dry_run=dry_run)
    if not dry_run:
        gxtbsp_log_path = scratch_dir / "gxtbsp_run.log"
        gxtbsp_log_path.write_text(cp_sp.stdout)
        log_status(log_paths, "OK", f"saved g-xTB SP stdout to {gxtbsp_log_path.name}")
    if cp_sp.returncode != 0:
        log_status(
            log_paths,
            "FAIL",
            f"g-xTB SP failed returncode={cp_sp.returncode} stdout_tail={cp_sp.stdout[-1000:]} stderr_tail={cp_sp.stderr[-1000:]}",
        )
        raise RuntimeError(
            f"g-xTB gas-phase SP calculation failed with code {cp_sp.returncode}.\n"
            f"stdout:\n{cp_sp.stdout[-4000:]}\n"
            f"stderr:\n{cp_sp.stderr[-4000:]}\n"
        )

    gas_sp_energy_h = parse_xtb_total_energy_hartree(cp_sp.stdout)
    gas_sp_energy_kcal_mol = None
    if gas_sp_energy_h is not None:
        gas_sp_energy_kcal_mol = gas_sp_energy_h * HARTREE_TO_KCAL_MOL
    else:
        warnings.warn(
            "Could not parse TOTAL ENERGY from g-xTB SP output. "
            "Gas-phase energy will remain None.",
            RuntimeWarning,
        )
        log_status(log_paths, "WARN", "failed to parse gas-phase SP energy from g-xTB SP output")

    return gas_sp_energy_kcal_mol, gas_sp_energy_h


def run_hessian_and_parse_energies(
    *,
    scratch_dir: Path,
    xyz_path: Path,
    xtb_executable: str,
    charge: int,
    gfn: int,
    timeout_s: Optional[int],
    dry_run: bool,
    log_paths: list[Path],
    run_command: Callable[..., subprocess.CompletedProcess[str]],
    log_status: Callable[[list[Path], str, str], None],
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    cmd_hess = [
        xtb_executable,
        str(xyz_path.name),
        "--hess",
        "--chrg",
        str(charge),
        "--gfn",
        str(gfn),
    ]
    log_status(log_paths, "STEP", f"running hessian: {' '.join(shlex.quote(x) for x in cmd_hess)}")
    cp_hess = run_command(cmd_hess, cwd=scratch_dir, timeout_s=timeout_s, dry_run=dry_run)
    if not dry_run:
        frequency_log_path = scratch_dir / "xtbfreq_run.log"
        frequency_log_path.write_text(cp_hess.stdout)
        log_status(log_paths, "OK", f"saved frequency stdout to {frequency_log_path.name}")
    if cp_hess.returncode != 0:
        log_status(
            log_paths,
            "FAIL",
            f"hessian failed returncode={cp_hess.returncode} stdout_tail={cp_hess.stdout[-1000:]} stderr_tail={cp_hess.stderr[-1000:]}",
        )
        raise RuntimeError(
            f"xTB hess calculation failed with code {cp_hess.returncode}.\n"
            f"stdout:\n{cp_hess.stdout[-4000:]}\n"
            f"stderr:\n{cp_hess.stderr[-4000:]}\n"
        )

    gas_sp_energy_h = parse_xtb_total_energy_hartree(cp_hess.stdout)
    rrho_contrib_h = parse_xtb_rrho_contrib_hartree(cp_hess.stdout)

    gas_sp_energy_kcal_mol = None
    if gas_sp_energy_h is not None:
        gas_sp_energy_kcal_mol = gas_sp_energy_h * HARTREE_TO_KCAL_MOL

    rrho_contrib_kcal_mol = None
    if rrho_contrib_h is not None:
        rrho_contrib_kcal_mol = rrho_contrib_h * HARTREE_TO_KCAL_MOL

    return gas_sp_energy_kcal_mol, rrho_contrib_kcal_mol, gas_sp_energy_h