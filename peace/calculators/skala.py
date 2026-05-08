import traceback
from pathlib import Path
from typing import Callable, Optional

from .common import HARTREE_TO_KCAL_MOL


def run_skala_single_point_energy(
    *,
    scratch_dir: Path,
    xyz_path: Path,
    charge: int,
    multiplicity: int,
    dry_run: bool,
    log_paths: list[Path],
    log_status: Callable[[list[Path], str, str], None],
    basis: str = "ma-def2-QZVP",
    xc: str = "skala-1.1",
) -> tuple[Optional[float], Optional[float]]:
    """
    Run Skala single-point energy via GPU4PySCF and return:
    (energy_kcal_mol, energy_hartree).
    """
    if dry_run:
        log_status(log_paths, "SKIP", "dry_run; skipping Skala SP")
        return None, None

    spin = int(multiplicity) - 1
    if spin < 0:
        raise ValueError(f"Invalid multiplicity={multiplicity}; expected >= 1.")

    log_status(
        log_paths,
        "STEP",
        f"running Skala SP on {xyz_path.name} with basis={basis} xc={xc} charge={charge} spin={spin}",
    )
    try:
        from pyscf import gto
        from skala.gpu4pyscf import SkalaKS
    except Exception as exc:
        raise RuntimeError(
            "Skala/GPU4PySCF dependencies are unavailable. "
            "Install PySCF + skala + gpu4pyscf for sp_energy='skala'."
        ) from exc

    try:
        # PySCF gto can parse XYZ files directly from path-like strings.
        mol = gto.M(
            atom=str(xyz_path),
            basis=basis,
            charge=int(charge),
            spin=spin,
        )
        ks = SkalaKS(mol, xc=xc)
        energy_h = ks.kernel()
        if energy_h is None:
            raise RuntimeError("Skala kernel() returned None.")
        energy_h = float(energy_h)
        energy_kcal_mol = energy_h * HARTREE_TO_KCAL_MOL
    except Exception as exc:
        err = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        log_status(log_paths, "FAIL", f"Skala SP failed: {err}")
        raise RuntimeError(f"Skala SP calculation failed: {err}") from exc

    log_path = scratch_dir / "skalasp_run.log"
    log_path.write_text(
        f"xyz_path={xyz_path}\n"
        f"basis={basis}\n"
        f"xc={xc}\n"
        f"charge={charge}\n"
        f"spin={spin}\n"
        f"energy_hartree={energy_h}\n"
        f"energy_kcal_mol={energy_kcal_mol}\n",
        encoding="utf-8",
    )
    log_status(log_paths, "OK", f"saved Skala SP summary to {log_path.name}")
    return energy_kcal_mol, energy_h
