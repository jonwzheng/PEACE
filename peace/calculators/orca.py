from __future__ import annotations

import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Optional

from rdkit import Chem

from .common import float_regex, parse_last_float


def _mol_h_to_orca_coord_lines(mol: Chem.Mol, *, conf_id: int = 0) -> str:
    mol_h = Chem.AddHs(Chem.Mol(mol), addCoords=True)
    if mol_h.GetNumConformers() == 0:
        raise ValueError("Molecule has no conformers; cannot write ORCA coordinates.")
    conf = mol_h.GetConformer(int(conf_id))
    lines: list[str] = []
    for i, atom in enumerate(mol_h.GetAtoms()):
        p = conf.GetAtomPosition(i)
        sym = atom.GetSymbol()
        lines.append(f"{sym:>2}    {p.x:20.14f}       {p.y:20.14f}       {p.z:20.14f}")
    return "\n".join(lines)


def parse_orca_cosmo_rs_dgsolv_kcal_mol(text: str) -> Optional[float]:
    float_re = float_regex()
    kcal_token = re.compile(rf"({float_re})\s+kcal/mol", re.IGNORECASE)
    for line in text.splitlines():
        if "Free energy of solvation" in line and "dGsolv" in line and "kcal/mol" in line:
            matches = list(kcal_token.finditer(line))
            if matches:
                try:
                    return float(matches[-1].group(1))
                except ValueError:
                    continue
    for line in text.splitlines():
        if "dgsolv" in line.lower() and "kcal/mol" in line.lower():
            matches = list(kcal_token.finditer(line))
            if matches:
                try:
                    return float(matches[-1].group(1))
                except ValueError:
                    continue
    return None


def parse_orca_solute_gas_phase_energy_hartree(text: str) -> Optional[float]:
    float_re = float_regex()
    patterns = [
        rf"FINAL SINGLE POINT ENERGY \(Solute-gas-phase\)\s*({float_re})",
    ]
    return parse_last_float(patterns, text)


def run_orca_cosmo_rs(
    *,
    mol: Chem.Mol,
    scratch_dir: Path,
    charge: int,
    multiplicity: int,
    solvent: str,
    orca_executable: str,
    timeout_s: Optional[int],
    dry_run: bool,
    log_paths: list[Path],
    run_command: Callable[..., subprocess.CompletedProcess[str]],
    log_status: Callable[[list[Path], str, str], None],
) -> tuple[Optional[float], Optional[float], str]:
    scratch_dir.mkdir(parents=True, exist_ok=True)
    inp_name = "cosmo_job.inp"
    inp_path = scratch_dir / inp_name
    coord_block = _mol_h_to_orca_coord_lines(mol)
    inp_body = (
        f"!COSMORS({solvent})\n"
        f"* xyz {charge} {multiplicity}\n"
        f"{coord_block}\n"
        f"*\n"
    )
    inp_path.write_text(inp_body, encoding="utf-8")
    log_status(log_paths, "OK", f"wrote ORCA input {inp_path.name}")

    if dry_run:
        log_status(log_paths, "SKIP", "dry_run; skipping ORCA COSMO-RS")
        return None, None, ""

    cmd = [orca_executable, inp_name]
    log_status(log_paths, "STEP", f"running ORCA COSMO-RS: {' '.join(shlex.quote(x) for x in cmd)}")
    cp = run_command(cmd, cwd=scratch_dir, timeout_s=timeout_s, dry_run=False)
    out_path = scratch_dir / inp_name.replace(".inp", ".out")
    merged = cp.stdout
    if cp.stderr:
        merged += "\n" + cp.stderr
    if out_path.exists():
        merged += "\n" + out_path.read_text(encoding="utf-8", errors="replace")

    merged_log = scratch_dir / "orca_cosmo_run.log"
    merged_log.write_text(merged, encoding="utf-8")
    log_status(log_paths, "OK", f"saved ORCA merged output to {merged_log.name}")

    if cp.returncode != 0:
        log_status(
            log_paths,
            "FAIL",
            f"ORCA failed returncode={cp.returncode} tail={merged[-800:]}",
        )
        return None, None, merged

    dgsolv = parse_orca_cosmo_rs_dgsolv_kcal_mol(merged)
    if dgsolv is None:
        log_status(log_paths, "WARN", "could not parse dGsolv (kcal/mol) from ORCA output")
    else:
        log_status(log_paths, "OK", f"parsed ORCA dGsolv_kcal_mol={dgsolv}")
    gas_sp_h = parse_orca_solute_gas_phase_energy_hartree(merged)
    if gas_sp_h is None:
        log_status(log_paths, "WARN", "could not parse ORCA solute gas-phase SP energy (Hartree)")
    else:
        log_status(log_paths, "OK", f"parsed ORCA solute gas-phase SP (Hartree)={gas_sp_h}")
    return dgsolv, gas_sp_h, merged


def cleanup_orca_refine_scratch_keep_log(
    scratch_dir: Path,
    keep_scratch: bool = False,
    *,
    log_paths: list[Path],
    log_status: Callable[[list[Path], str, str], None],
) -> None:
    keep_name = "orca_cosmo_run.log"
    if not scratch_dir.exists() or keep_scratch:
        return
    for entry in scratch_dir.iterdir():
        if entry.name == keep_name:
            continue
        try:
            if entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)
            else:
                entry.unlink(missing_ok=True)
        except Exception as exc:
            log_status(log_paths, "WARN", f"failed to remove ORCA refine artifact {entry.name}: {exc}")
    log_status(log_paths, "CLEANUP", f"kept only {keep_name} in {scratch_dir}")
