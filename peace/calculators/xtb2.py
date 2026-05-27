from __future__ import annotations

import shlex
import subprocess
import warnings
from pathlib import Path
from typing import Callable, Optional

from .common import HARTREE_TO_KCAL_MOL
from .xtb import (
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
    "run_gxtb_single_point_energy",
    "run_hessian_and_parse_energies",
    "run_xtb_optimization",
]


def run_gxtb_single_point_energy(
    *,
    scratch_dir: Path,
    xyz_path: Path,
    xtb_executable: str = "xtb2",
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
