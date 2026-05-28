# Compatibility layer for g-xTB 2.0.0+

from __future__ import annotations

import shlex
import subprocess
import warnings
from pathlib import Path
from typing import Callable, Optional

from rdkit import Chem

from .common import HARTREE_TO_KCAL_MOL
from .xtb import (
    build_fix_xcontrol,
    parse_xtb_rrho_contrib_hartree,
    parse_xtb_solvent_free_energy_hartree,
    parse_xtb_total_energy_hartree,
    run_cpcmx_single_point,
    run_hessian_and_parse_energies,
    run_xtb_optimization,
)

__all__ = [
    "parse_xtb_rrho_contrib_hartree",
    "parse_xtb_solvent_free_energy_hartree",
    "parse_xtb_total_energy_hartree",
    "run_cpcmx_single_point",
    "run_gxtb_optimization",
    "run_gxtb_single_point_energy",
    "run_hessian_and_parse_energies",
    "run_xtb_optimization",
]

def run_gxtb_optimization(
    *,
    scratch_dir: Path,
    xyz_path: Path,
    input_mol: Optional[Chem.Mol],
    xtb_executable: str,
    charge: int,
    timeout_s: Optional[int],
    dry_run: bool,
    log_paths: list[Path],
    run_command: Callable[..., subprocess.CompletedProcess[str]],
    log_status: Callable[[list[Path], str, str], None],
) -> tuple[Path, Optional[float], Optional[float]]:
    xcontrol_path = scratch_dir / "xcontrol.inp"
    input_flag = ""
    if xcontrol_path.exists():
        input_flag = f" --input {shlex.quote(xcontrol_path.name)}"
        log_status(log_paths, "OK", f"reusing constraints input for g-xTB optimization: {xcontrol_path.name}")
    elif input_mol is not None:
        n_constraints = build_fix_xcontrol(input_mol, xcontrol_path, fixed_distance=1.01)
        if n_constraints > 0:
            input_flag = f" --input {shlex.quote(xcontrol_path.name)}"
            log_status(
                log_paths,
                "OK",
                "regenerated xcontrol constraints for g-xTB optimization: "
                f"n_constraints={n_constraints}",
            )
        else:
            log_status(
                log_paths,
                "OK",
                "xcontrol.inp missing and no positively charged heavy-atom X-H constraints were generated",
            )
    else:
        log_status(log_paths, "OK", "xcontrol.inp missing and no input_mol available; running unconstrained")

    cmd_opt = (
        f"{shlex.quote(xtb_executable)} {shlex.quote(xyz_path.name)} "
        f"--gxtb{input_flag} --opt --grad --chrg {shlex.quote(str(charge))}"
    )
    log_status(log_paths, "STEP", f"running g-xTB optimization: {cmd_opt}")
    cp_opt = run_command(cmd_opt, cwd=scratch_dir, timeout_s=timeout_s, dry_run=dry_run)
    if not dry_run:
        gxtbopt_log_path = scratch_dir / "gxtbopt_run.log"
        gxtbopt_log_path.write_text(cp_opt.stdout)
        log_status(log_paths, "OK", f"saved g-xTB optimization stdout to {gxtbopt_log_path.name}")
    if cp_opt.returncode != 0:
        log_status(
            log_paths,
            "FAIL",
            f"g-xTB optimization failed returncode={cp_opt.returncode} stdout_tail={cp_opt.stdout[-1000:]} stderr_tail={cp_opt.stderr[-1000:]}",
        )
        raise RuntimeError(
            f"g-xTB optimization failed with code {cp_opt.returncode}.\n"
            f"stdout:\n{cp_opt.stdout[-4000:]}\n"
            f"stderr:\n{cp_opt.stderr[-4000:]}\n"
        )

    gxtbopt_xyz_path = scratch_dir / "xtbopt.xyz"
    if not gxtbopt_xyz_path.exists():
        log_status(log_paths, "FAIL", "g-xTB optimization finished but xtbopt.xyz was not produced")
        raise FileNotFoundError("Expected output xtbopt.xyz was not produced by g-xTB optimization.")

    gas_sp_energy_h = parse_xtb_total_energy_hartree(cp_opt.stdout)
    gas_sp_energy_kcal_mol = None
    if gas_sp_energy_h is not None:
        gas_sp_energy_kcal_mol = gas_sp_energy_h * HARTREE_TO_KCAL_MOL
    else:
        warnings.warn(
            "Could not parse TOTAL ENERGY from g-xTB optimization output. "
            "Gas-phase energy will remain None.",
            RuntimeWarning,
        )
        log_status(log_paths, "WARN", "failed to parse gas-phase energy from g-xTB optimization output")

    return gxtbopt_xyz_path, gas_sp_energy_kcal_mol, gas_sp_energy_h


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
    cmd_sp = [
        xtb_executable,
        str(xyz_path.name),
        "--gxtb",
        "--chrg",
        str(charge),
    ]
    log_status(log_paths, "STEP", f"running g-xTB gas-phase SP: {' '.join(shlex.quote(x) for x in cmd_sp)}")
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
