from __future__ import annotations

import traceback
from pathlib import Path
from typing import Callable, Optional

from .common import EV_TO_KCAL_MOL


def run_aimnet2_single_point_energy(
    *,
    scratch_dir: Path,
    xyz_path: Path,
    charge: int,
    dry_run: bool,
    log_paths: list[Path],
    log_status: Callable[[list[Path], str, str], None],
    model: str = "aimnet2",
) -> tuple[Optional[float], Optional[float]]:
    if dry_run:
        log_status(log_paths, "SKIP", "dry_run; skipping AIMNet2 SP")
        return None, None
    try:
        from ase.io import read
        from aimnet.calculators import AIMNet2ASE
    except Exception as exc:
        raise RuntimeError(
            "AIMNet2 dependencies are unavailable. Install 'aimnet[ase]' for AIMNet2 support."
        ) from exc

    log_status(log_paths, "STEP", f"running AIMNet2 SP on {xyz_path.name} model={model} charge={charge}")
    try:
        atoms = read(str(xyz_path))
        atoms.calc = AIMNet2ASE(model, charge=int(charge))
        energy_ev = float(atoms.get_potential_energy())
        energy_kcal_mol = energy_ev * EV_TO_KCAL_MOL
    except Exception as exc:
        err = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        log_status(log_paths, "FAIL", f"AIMNet2 SP failed: {err}")
        raise RuntimeError(f"AIMNet2 SP calculation failed: {err}") from exc

    log_path = scratch_dir / "aimnet2sp_run.log"
    log_path.write_text(
        f"xyz_path={xyz_path}\nmodel={model}\ncharge={charge}\nenergy_ev={energy_ev}\nenergy_kcal_mol={energy_kcal_mol}\n",
        encoding="utf-8",
    )
    log_status(log_paths, "OK", f"saved AIMNet2 SP summary to {log_path.name}")
    return energy_kcal_mol, energy_ev


def run_aimnet2_optimization(
    *,
    scratch_dir: Path,
    input_xyz_path: Path,
    charge: int,
    dry_run: bool,
    log_paths: list[Path],
    log_status: Callable[[list[Path], str, str], None],
    model: str = "aimnet2",
    fmax: float = 0.01,
    max_steps: int = 200,
) -> tuple[Path, Optional[float], Optional[float]]:
    optimized_xyz_path = scratch_dir / "aimnet2opt.xyz"
    if dry_run:
        log_status(log_paths, "SKIP", "dry_run; skipping AIMNet2 optimization")
        return optimized_xyz_path, None, None
    try:
        from ase.io import read, write
        from ase.optimize import LBFGS
        from aimnet.calculators import AIMNet2ASE
    except Exception as exc:
        raise RuntimeError(
            "AIMNet2 dependencies are unavailable. Install 'aimnet[ase]' for AIMNet2 support."
        ) from exc

    log_status(
        log_paths,
        "STEP",
        f"running AIMNet2 optimization on {input_xyz_path.name} model={model} charge={charge} fmax={fmax}",
    )
    try:
        atoms = read(str(input_xyz_path))
        atoms.calc = AIMNet2ASE(model, charge=int(charge))
        opt = LBFGS(atoms, logfile=str(scratch_dir / "aimnet2opt_run.log"))
        opt.run(fmax=float(fmax), steps=int(max_steps))
        write(str(optimized_xyz_path), atoms)
        energy_ev = float(atoms.get_potential_energy())
        energy_kcal_mol = energy_ev * EV_TO_KCAL_MOL
    except Exception as exc:
        err = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        log_status(log_paths, "FAIL", f"AIMNet2 optimization failed: {err}")
        raise RuntimeError(f"AIMNet2 optimization failed: {err}") from exc

    log_status(log_paths, "OK", f"AIMNet2 optimization produced {optimized_xyz_path.name}")
    return optimized_xyz_path, energy_kcal_mol, energy_ev
