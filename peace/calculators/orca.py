"""ORCA calculator: arbitrary-keyword single points, optimizations, frequencies, and solvation."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import warnings
from pathlib import Path
from typing import Callable, Optional

from rdkit import Chem

from .common import DEFAULT_TEMPERATURE_K, HARTREE_TO_KCAL_MOL, float_regex, parse_last_float
from .settings import OrcaSettings
from .xtb import zwitterion_xh_constraints

ORCA_NORMAL_TERMINATION_MARKER = "ORCA TERMINATED NORMALLY"
ORCA_OPT_CONVERGED_MARKER = "THE OPTIMIZATION HAS CONVERGED"

# xTB --opt presets mapped onto ORCA's optimization convergence keywords.
ORCA_OPT_KEYWORD_BY_LEVEL: dict[str, str] = {
    "crude": "LooseOpt",
    "sloppy": "LooseOpt",
    "loose": "LooseOpt",
    "lax": "LooseOpt",
    "normal": "Opt",
    "tight": "TightOpt",
    "vtight": "VeryTightOpt",
    "extreme": "VeryTightOpt",
}

_ORCA_OPT_TOKENS = frozenset(
    {"OPT", "LOOSEOPT", "NORMALOPT", "TIGHTOPT", "VERYTIGHTOPT", "COPT", "ZOPT", "GDIISOPT", "L-OPT"}
)
_ORCA_FREQ_TOKENS = frozenset({"FREQ", "NUMFREQ", "ANFREQ"})

RunCommand = Callable[..., subprocess.CompletedProcess[str]]
LogStatus = Callable[[list[Path], str, str], None]


def verify_orca_executable(orca_executable: str) -> str:
    """Return the absolute path to the ORCA binary (ORCA requires a full path for parallel runs)."""
    exe = orca_executable.strip()
    if not exe:
        raise ValueError("--orca-executable must not be empty.")
    if os.sep in exe or (os.altsep and os.altsep in exe):
        path = Path(exe)
        if not path.is_file():
            raise ValueError(f"ORCA executable not found: {exe}")
        if not os.access(path, os.X_OK):
            raise ValueError(f"ORCA executable is not executable: {exe}")
        return str(path.resolve())
    found = shutil.which(exe)
    if found is None:
        raise ValueError(
            f"ORCA executable {exe!r} not found on PATH. Provide a full path via --orca-executable."
        )
    return str(Path(found).resolve())


def spin_multiplicity(mol: Chem.Mol) -> int:
    n_radical = sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms())
    return int(n_radical) + 1


def orca_opt_keyword(opt_level: str) -> str:
    return ORCA_OPT_KEYWORD_BY_LEVEL.get(opt_level, "Opt")


def _keyword_tokens(keywords: str) -> set[str]:
    return {token.upper() for token in keywords.replace("!", " ").split()}


def parse_orca_final_energy_hartree(text: str) -> Optional[float]:
    return parse_last_float([rf"FINAL SINGLE POINT ENERGY\s+({float_regex()})"], text)


def parse_orca_rrho_contrib_hartree(text: str) -> Optional[float]:
    """Parse ``G-E(el)``: the thermal free-energy correction (ZPE + thermal + -TS)."""
    return parse_last_float([rf"G-E\(el\)\s*\.\.\.\s*({float_regex()})\s*Eh"], text)


def parse_orca_cosmo_rs_dgsolv_kcal_mol(text: str) -> Optional[float]:
    return parse_last_float(
        [rf"Free energy of solvation \(dGsolv\)\s*:.*?({float_regex()})\s*kcal/mol"],
        text,
    )


def build_orca_input(
    *,
    keywords: str,
    xyz_name: str,
    charge: int,
    multiplicity: int,
    settings: OrcaSettings,
    blocks: Optional[list[str]] = None,
) -> str:
    lines = [f"! {keywords.strip()}"]
    if settings.nprocs > 1:
        lines.append(f"%pal nprocs {int(settings.nprocs)} end")
    if settings.maxcore_mb is not None:
        lines.append(f"%maxcore {int(settings.maxcore_mb)}")
    for block in blocks or []:
        lines.append(block.rstrip())
    if settings.extra_input:
        lines.append(settings.extra_input.rstrip())
    lines.append(f"* xyzfile {int(charge)} {int(multiplicity)} {xyz_name}")
    return "\n".join(lines) + "\n"


def run_orca_job(
    *,
    job_name: str,
    keywords: str,
    scratch_dir: Path,
    xyz_path: Path,
    charge: int,
    multiplicity: int,
    settings: OrcaSettings,
    dry_run: bool,
    log_paths: list[Path],
    run_command: RunCommand,
    log_status: LogStatus,
    blocks: Optional[list[str]] = None,
    description: str = "ORCA job",
) -> Optional[str]:
    """
    Write ``<job_name>.inp``, run ORCA, and return the output text.

    Returns None on dry runs. Raises RuntimeError when ORCA does not terminate normally.
    """
    scratch_dir.mkdir(parents=True, exist_ok=True)
    if xyz_path.parent.resolve() != scratch_dir.resolve():
        shutil.copy2(xyz_path, scratch_dir / xyz_path.name)
    inp_name = f"{job_name}.inp"
    (scratch_dir / inp_name).write_text(
        build_orca_input(
            keywords=keywords,
            xyz_name=xyz_path.name,
            charge=charge,
            multiplicity=multiplicity,
            settings=settings,
            blocks=blocks,
        ),
        encoding="utf-8",
    )
    cmd = [settings.executable, inp_name]
    log_status(
        log_paths,
        "STEP",
        f"running {description}: {' '.join(shlex.quote(x) for x in cmd)} keywords='! {keywords}'",
    )
    if dry_run:
        log_status(log_paths, "SKIP", f"dry_run; skipping {description}")
        return None

    completed = run_command(cmd, cwd=scratch_dir, dry_run=False)
    output = completed.stdout or ""
    log_path = scratch_dir / f"{job_name}_run.log"
    log_path.write_text(output + (f"\n{completed.stderr}" if completed.stderr else ""), encoding="utf-8")
    log_status(log_paths, "OK", f"saved {description} output to {log_path.name}")

    if completed.returncode != 0 or ORCA_NORMAL_TERMINATION_MARKER not in output:
        log_status(
            log_paths,
            "FAIL",
            f"{description} failed returncode={completed.returncode} "
            f"stdout_tail={output[-1000:]} stderr_tail={(completed.stderr or '')[-1000:]}",
        )
        raise RuntimeError(
            f"{description} did not terminate normally (returncode={completed.returncode}).\n"
            f"stdout:\n{output[-4000:]}\n"
            f"stderr:\n{(completed.stderr or '')[-4000:]}\n"
        )
    return output


def _energy_or_warn(
    energy_h: Optional[float],
    *,
    description: str,
    log_paths: list[Path],
    log_status: LogStatus,
) -> tuple[Optional[float], Optional[float]]:
    if energy_h is None:
        warnings.warn(f"Could not parse FINAL SINGLE POINT ENERGY from {description} output.", RuntimeWarning)
        log_status(log_paths, "WARN", f"failed to parse energy from {description} output")
        return None, None
    return energy_h * HARTREE_TO_KCAL_MOL, energy_h


def run_orca_single_point_energy(
    *,
    scratch_dir: Path,
    xyz_path: Path,
    keywords: str,
    charge: int,
    multiplicity: int,
    settings: OrcaSettings,
    dry_run: bool,
    log_paths: list[Path],
    run_command: RunCommand,
    log_status: LogStatus,
    job_name: str = "orcasp",
) -> tuple[Optional[float], Optional[float]]:
    """Run an ORCA single point with arbitrary keywords; return (kcal/mol, Hartree)."""
    output = run_orca_job(
        job_name=job_name,
        keywords=keywords,
        scratch_dir=scratch_dir,
        xyz_path=xyz_path,
        charge=charge,
        multiplicity=multiplicity,
        settings=settings,
        dry_run=dry_run,
        log_paths=log_paths,
        run_command=run_command,
        log_status=log_status,
        description="ORCA SP",
    )
    if output is None:
        return None, None
    return _energy_or_warn(
        parse_orca_final_energy_hartree(output),
        description="ORCA SP",
        log_paths=log_paths,
        log_status=log_status,
    )


def _orca_constraint_block(constraints: list[tuple[int, int, float]]) -> str:
    entries = "\n".join(f"  {{ B {heavy} {h} {dist:.2f} C }}" for heavy, h, dist in constraints)
    return f"%geom\n Constraints\n{entries}\n end\nend"


def run_orca_optimization(
    *,
    scratch_dir: Path,
    xyz_path: Path,
    keywords: str,
    opt_level: str,
    charge: int,
    multiplicity: int,
    settings: OrcaSettings,
    dry_run: bool,
    log_paths: list[Path],
    run_command: RunCommand,
    log_status: LogStatus,
    input_mol: Optional[Chem.Mol] = None,
    solvent_keyword: Optional[str] = None,
    job_name: str = "orcaopt",
) -> tuple[Path, Optional[float], Optional[float]]:
    """
    Run an ORCA geometry optimization and return (optimized xyz, kcal/mol, Hartree).

    ``solvent_keyword`` (e.g. ``CPCM(water)``) switches on implicit solvent.
    Positively charged X-H bonds on ``input_mol`` are frozen at their target
    distances, mirroring the xTB constraints used for zwitterions.
    """
    job_keywords = keywords
    if not (_keyword_tokens(keywords) & _ORCA_OPT_TOKENS):
        job_keywords = f"{job_keywords} {orca_opt_keyword(opt_level)}"
    if solvent_keyword:
        job_keywords = f"{job_keywords} {solvent_keyword}"

    blocks: list[str] = []
    if input_mol is not None:
        constraints = zwitterion_xh_constraints(input_mol, fixed_distance=1.01)
        if constraints:
            blocks.append(_orca_constraint_block(constraints))
            log_status(
                log_paths,
                "OK",
                f"generated ORCA bond constraints for positively charged X-H sites: n_constraints={len(constraints)}",
            )

    description = "ORCA optimization" + (f" ({solvent_keyword})" if solvent_keyword else " (gas phase)")
    opt_xyz_path = scratch_dir / f"{job_name}.xyz"
    output = run_orca_job(
        job_name=job_name,
        keywords=job_keywords,
        scratch_dir=scratch_dir,
        xyz_path=xyz_path,
        charge=charge,
        multiplicity=multiplicity,
        settings=settings,
        dry_run=dry_run,
        log_paths=log_paths,
        run_command=run_command,
        log_status=log_status,
        blocks=blocks,
        description=description,
    )
    if output is None:
        return opt_xyz_path, None, None
    if ORCA_OPT_CONVERGED_MARKER not in output:
        log_status(log_paths, "FAIL", f"{description} did not converge")
        raise RuntimeError(f"{description} terminated without converging (level={opt_level}).")
    if not opt_xyz_path.is_file():
        log_status(log_paths, "FAIL", f"{description} finished but {opt_xyz_path.name} was not produced")
        raise FileNotFoundError(f"Expected ORCA output {opt_xyz_path.name} was not produced.")
    log_status(log_paths, "OK", f"{description} produced {opt_xyz_path.name}")
    energy_kcal_mol, energy_h = _energy_or_warn(
        parse_orca_final_energy_hartree(output),
        description=description,
        log_paths=log_paths,
        log_status=log_status,
    )
    return opt_xyz_path, energy_kcal_mol, energy_h


def run_orca_frequencies(
    *,
    scratch_dir: Path,
    xyz_path: Path,
    keywords: str,
    charge: int,
    multiplicity: int,
    settings: OrcaSettings,
    dry_run: bool,
    log_paths: list[Path],
    run_command: RunCommand,
    log_status: LogStatus,
    temperature_k: float = DEFAULT_TEMPERATURE_K,
    job_name: str = "orcafreq",
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Run an ORCA frequency calculation.

    Returns (electronic energy kcal/mol, RRHO free-energy correction kcal/mol,
    electronic energy Hartree). The RRHO term is ORCA's ``G-E(el)`` at ``temperature_k``.
    """
    if temperature_k <= 0:
        raise ValueError(f"temperature_k must be > 0 K, got {temperature_k}")
    job_keywords = keywords
    if not (_keyword_tokens(keywords) & _ORCA_FREQ_TOKENS):
        job_keywords = f"{job_keywords} FREQ"
    output = run_orca_job(
        job_name=job_name,
        keywords=job_keywords,
        scratch_dir=scratch_dir,
        xyz_path=xyz_path,
        charge=charge,
        multiplicity=multiplicity,
        settings=settings,
        dry_run=dry_run,
        log_paths=log_paths,
        run_command=run_command,
        log_status=log_status,
        blocks=[f"%freq\n Temp {float(temperature_k):.2f}\nend"],
        description="ORCA frequencies",
    )
    if output is None:
        return None, None, None
    energy_kcal_mol, energy_h = _energy_or_warn(
        parse_orca_final_energy_hartree(output),
        description="ORCA frequencies",
        log_paths=log_paths,
        log_status=log_status,
    )
    rrho_h = parse_orca_rrho_contrib_hartree(output)
    rrho_kcal_mol = None
    if rrho_h is None:
        warnings.warn("Could not parse G-E(el) from ORCA frequency output.", RuntimeWarning)
        log_status(log_paths, "WARN", "failed to parse G-E(el) from ORCA frequency output")
    else:
        rrho_kcal_mol = rrho_h * HARTREE_TO_KCAL_MOL
    return energy_kcal_mol, rrho_kcal_mol, energy_h


def orca_solvent_keyword(model: str, solvent: str) -> str:
    return f"{model.upper()}({solvent})"


def run_orca_solvation_single_point(
    *,
    scratch_dir: Path,
    xyz_path: Path,
    keywords: Optional[str],
    model: str,
    solvent: str,
    charge: int,
    multiplicity: int,
    settings: OrcaSettings,
    dry_run: bool,
    log_paths: list[Path],
    run_command: RunCommand,
    log_status: LogStatus,
) -> Optional[float]:
    """
    Compute dG_solv (kcal/mol) with ORCA.

    - ``cosmors``: openCOSMO-RS (``! COSMORS(solvent)``); dG_solv is parsed directly.
    - ``cpcm`` / ``smd``: dG_solv = E(solvated SP) - E(gas SP), both at ``keywords``.
    """
    if model == "cosmors":
        job_keywords = orca_solvent_keyword("COSMORS", solvent)
        if keywords:
            job_keywords = f"{keywords} {job_keywords}"
        output = run_orca_job(
            job_name="orcacosmors",
            keywords=job_keywords,
            scratch_dir=scratch_dir,
            xyz_path=xyz_path,
            charge=charge,
            multiplicity=multiplicity,
            settings=settings,
            dry_run=dry_run,
            log_paths=log_paths,
            run_command=run_command,
            log_status=log_status,
            description="ORCA openCOSMO-RS",
        )
        if output is None:
            return None
        dgsolv = parse_orca_cosmo_rs_dgsolv_kcal_mol(output)
        if dgsolv is None:
            log_status(log_paths, "WARN", "could not parse dGsolv (kcal/mol) from ORCA openCOSMO-RS output")
        log_status(log_paths, "OK", f"parsed solvation_free_energy_kcal_mol={dgsolv}")
        return dgsolv

    if not keywords:
        raise ValueError(f"ORCA {model.upper()} solvation requires method keywords.")
    _, gas_h = run_orca_single_point_energy(
        scratch_dir=scratch_dir,
        xyz_path=xyz_path,
        keywords=keywords,
        charge=charge,
        multiplicity=multiplicity,
        settings=settings,
        dry_run=dry_run,
        log_paths=log_paths,
        run_command=run_command,
        log_status=log_status,
        job_name="orcasolv_gas",
    )
    _, solv_h = run_orca_single_point_energy(
        scratch_dir=scratch_dir,
        xyz_path=xyz_path,
        keywords=f"{keywords} {orca_solvent_keyword(model, solvent)}",
        charge=charge,
        multiplicity=multiplicity,
        settings=settings,
        dry_run=dry_run,
        log_paths=log_paths,
        run_command=run_command,
        log_status=log_status,
        job_name="orcasolv_solv",
    )
    if gas_h is None or solv_h is None:
        return None
    dgsolv = (solv_h - gas_h) * HARTREE_TO_KCAL_MOL
    log_status(log_paths, "OK", f"parsed solvation_free_energy_kcal_mol={dgsolv}")
    return dgsolv
