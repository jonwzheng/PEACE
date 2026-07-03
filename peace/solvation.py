import copy
import json
import math
import shutil
import subprocess
import warnings

from datetime import datetime

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Optional, Union

from rdkit import Chem
from rdkit.Chem import AllChem, rdDetermineBonds, rdMolTransforms

from .calculators import (
    run_aimnet2_optimization,
    run_aimnet2_single_point_energy,
    run_cpcmx_single_point,
    run_gxtb_single_point_energy,
    run_gxtb2_optimization,
    run_gxtb2_single_point_energy,
    run_gxtb_optimization,
    run_hessian_and_parse_energies,
    run_xtb_optimization,
)
from .calculators.common import opt_convergence_retry_levels
from .protomer import Protomer, Species, Tautomer

HARTREE_TO_KCAL_MOL = 627.5094740631
KCAL_MOL_PER_K = 0.00198720425864083

XtbVersion = Literal["legacy", "default"]
ConformerMode = Literal["kdg", "external_xyz", "skip_search"]

REFINEMENT_MIN_EMBED_CONFORMERS = 20
REFINEMENT_MAX_QM_CONFORMERS = 20
REFINEMENT_MAX_EMBED_CONFORMERS = 500
CONFORMER_DIHEDRAL_MAX_DEV_DEG = 10.0
CONFORMER_DIHEDRAL_RMS_DEV_DEG = 20.0
CONFORMER_HEAVY_ATOM_RMSD_ANG = 0.20

EnergyListValue = Union[float, str]


def _resolve_gxtb_single_point_runner(xtb_version: XtbVersion):
    if xtb_version == "default":
        return run_gxtb2_single_point_energy
    return run_gxtb_single_point_energy


def _resolve_gxtb_optimization_runner(xtb_version: XtbVersion):
    if xtb_version == "default":
        return run_gxtb2_optimization
    return run_gxtb_optimization


@dataclass
class ConformerEnergyTerms:
    """Per-conformer solvation workflow terms (numeric or failure token)."""

    gas_sp_energy_kcal_mol: EnergyListValue
    gas_sp_energy_xtb_kcal_mol: Optional[EnergyListValue] = None
    solvation_free_energy_kcal_mol: EnergyListValue = "not-run"
    rrho_contribution_kcal_mol: EnergyListValue = "not-run"
    solution_phase_free_energy_kcal_mol: EnergyListValue = "not-run"
    workflow_status: str = "not-run"


@dataclass
class ConformerWorkflowResult:
    terms: ConformerEnergyTerms
    opt_xyz_path: Optional[Path] = None


def _format_energy_entry(value: Optional[float], *, failed_step: str) -> EnergyListValue:
    if value is None:
        return f"{failed_step}-failed"
    return float(value)


def _serialize_energy_list(values: list[EnergyListValue]) -> str:
    return json.dumps(values)


def _set_energy_list_prop(mol: Chem.Mol, key: str, values: list[EnergyListValue]) -> None:
    mol.SetProp(key, _serialize_energy_list(values))


def _numeric_energy_values(values: list[EnergyListValue]) -> list[float]:
    numeric: list[float] = []
    for value in values:
        if isinstance(value, (int, float)):
            numeric.append(float(value))
    return numeric


def _boltzmann_aggregate_energy(
    energies: list[EnergyListValue],
    *,
    temperature_k: float = 298.15,
) -> Optional[float]:
    numeric = _numeric_energy_values(energies)
    if not numeric:
        return None
    if len(numeric) == 1:
        return numeric[0]
    rt = KCAL_MOL_PER_K * float(temperature_k)
    min_energy = min(numeric)
    log_sum = math.log(sum(math.exp(-(energy - min_energy) / rt) for energy in numeric))
    return min_energy - rt * log_sum


def _wrap_dihedral_delta_deg(delta: float) -> float:
    wrapped = (float(delta) + 180.0) % 360.0 - 180.0
    return abs(wrapped)


def _ensure_ring_info(mol: Chem.Mol) -> None:
    """Initialize RingInfo before IsInRing(), MMFF, or rotatable-bond queries."""
    if mol.GetNumAtoms() == 0:
        return
    mol.UpdatePropertyCache(strict=False)
    Chem.GetSSSR(mol)


def _iter_rotatable_dihedral_quads(mol_h: Chem.Mol) -> list[tuple[int, int, int, int]]:
    quads: list[tuple[int, int, int, int]] = []
    for bond in mol_h.GetBonds():
        begin = bond.GetBeginAtom()
        end = bond.GetEndAtom()
        if begin.GetDegree() < 2 or end.GetDegree() < 2:
            continue
        begin_neighbors = [n.GetIdx() for n in begin.GetNeighbors() if n.GetIdx() != end.GetIdx()]
        end_neighbors = [n.GetIdx() for n in end.GetNeighbors() if n.GetIdx() != begin.GetIdx()]
        if not begin_neighbors or not end_neighbors:
            continue
        quads.append((begin_neighbors[0], begin.GetIdx(), end.GetIdx(), end_neighbors[0]))
    return quads


def _rotatable_dihedral_signature(mol_h: Chem.Mol, conf_id: int) -> list[float]:
    conf = mol_h.GetConformer(int(conf_id))
    return [
        float(rdMolTransforms.GetDihedralDeg(conf, a, b, c, d))
        for a, b, c, d in _iter_rotatable_dihedral_quads(mol_h)
    ]


def _heavy_atom_rmsd(mol_h: Chem.Mol, conf_id_a: int, conf_id_b: int) -> float:
    mol_a = Chem.Mol(mol_h)
    mol_b = Chem.Mol(mol_h)
    mol_a.RemoveAllConformers()
    mol_b.RemoveAllConformers()
    mol_a.AddConformer(mol_h.GetConformer(int(conf_id_a)), assignId=True)
    mol_b.AddConformer(mol_h.GetConformer(int(conf_id_b)), assignId=True)
    mol_a_no_h = Chem.RemoveHs(mol_a)
    mol_b_no_h = Chem.RemoveHs(mol_b)
    return float(AllChem.GetBestRMS(mol_a_no_h, mol_b_no_h))


def _conformers_are_redundant(
    mol_h: Chem.Mol,
    conf_id_a: int,
    conf_id_b: int,
    *,
    dihedral_max_dev_deg: float = CONFORMER_DIHEDRAL_MAX_DEV_DEG,
    dihedral_rms_dev_deg: float = CONFORMER_DIHEDRAL_RMS_DEV_DEG,
    heavy_atom_rmsd_ang: float = CONFORMER_HEAVY_ATOM_RMSD_ANG,
) -> bool:
    dihedrals_a = _rotatable_dihedral_signature(mol_h, conf_id_a)
    dihedrals_b = _rotatable_dihedral_signature(mol_h, conf_id_b)
    if dihedrals_a and len(dihedrals_a) == len(dihedrals_b):
        diffs = [_wrap_dihedral_delta_deg(a - b) for a, b in zip(dihedrals_a, dihedrals_b)]
        max_dev = max(diffs)
        dihedral_rms = math.sqrt(sum(diff * diff for diff in diffs) / len(diffs))
        if max_dev < dihedral_max_dev_deg and dihedral_rms < dihedral_rms_dev_deg:
            return True
    try:
        return _heavy_atom_rmsd(mol_h, conf_id_a, conf_id_b) < heavy_atom_rmsd_ang
    except Exception:
        return False


def _read_xyz_coords(path: Path) -> list[tuple[str, float, float, float]]:
    lines = path.read_text().strip().splitlines()
    n_atoms = int(lines[0].strip())
    coords: list[tuple[str, float, float, float]] = []
    for line in lines[2 : 2 + n_atoms]:
        parts = line.split()
        coords.append((parts[0], float(parts[1]), float(parts[2]), float(parts[3])))
    return coords


def _heavy_atom_rmsd_from_xyz(path_a: Path, path_b: Path) -> float:
    coords_a = [(x, y, z) for sym, x, y, z in _read_xyz_coords(path_a) if sym.upper() != "H"]
    coords_b = [(x, y, z) for sym, x, y, z in _read_xyz_coords(path_b) if sym.upper() != "H"]
    if not coords_a or len(coords_a) != len(coords_b):
        return float("inf")
    sum_sq = 0.0
    for (x1, y1, z1), (x2, y2, z2) in zip(coords_a, coords_b):
        sum_sq += (x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2
    return math.sqrt(sum_sq / len(coords_a))


@dataclass
class ConformerPoolEntry:
    label: str
    terms: ConformerEnergyTerms
    opt_xyz_path: Optional[Path]


def _kdg_embed_parameters(*, random_seed: int) -> AllChem.EmbedParameters:
    params = AllChem.EmbedParameters()
    params.randomSeed = int(random_seed)
    params.numThreads = 0
    params.useRandomCoords = False
    return params


def _mmff94_conformer_energy_kcal_mol(mol_h: Chem.Mol, conf_id: int) -> Optional[float]:
    try:
        _ensure_ring_info(mol_h)
        mmff_props = AllChem.MMFFGetMoleculeProperties(mol_h, mmffVariant="MMFF94")
        if mmff_props is None:
            return None
        ff = AllChem.MMFFGetMoleculeForceField(mol_h, mmff_props, confId=int(conf_id))
        if ff is None:
            return None
        ff.Initialize()
        return float(ff.CalcEnergy())
    except Exception:
        return None


def _connectivity_signature_for_mol(mol: Chem.Mol) -> set[tuple[int, int]]:
    return _all_atom_connectivity_signature(Chem.AddHs(Chem.Mol(mol)))


def _connectivity_matches_reference(mol: Chem.Mol, reference_mol: Chem.Mol) -> bool:
    return _connectivity_signature_for_mol(mol) == _connectivity_signature_for_mol(reference_mol)


def _mol_from_xyz_with_connectivity(xyz_path: Path) -> Optional[Chem.Mol]:
    mol = Chem.MolFromXYZFile(str(xyz_path))
    if mol is None or mol.GetNumConformers() == 0:
        return None
    rdDetermineBonds.DetermineConnectivity(mol)
    _ensure_ring_info(mol)
    return mol


def _xyz_connectivity_matches_reference(xyz_path: Path, reference_mol: Chem.Mol) -> bool:
    mol = _mol_from_xyz_with_connectivity(xyz_path)
    if mol is None:
        return False
    return _connectivity_matches_reference(mol, reference_mol)


def _dihedrals_from_xyz(xyz_path: Path) -> Optional[list[float]]:
    mol = _mol_from_xyz_with_connectivity(xyz_path)
    if mol is None:
        return None
    try:
        mol_h = Chem.AddHs(mol)
        return _rotatable_dihedral_signature(mol_h, 0)
    except Exception:
        return None


def _dihedral_signatures_match(dih_a: list[float], dih_b: list[float]) -> bool:
    if not dih_a or len(dih_a) != len(dih_b):
        return False
    diffs = [_wrap_dihedral_delta_deg(a - b) for a, b in zip(dih_a, dih_b)]
    max_dev = max(diffs)
    dihedral_rms = math.sqrt(sum(diff * diff for diff in diffs) / len(diffs))
    return max_dev < CONFORMER_DIHEDRAL_MAX_DEV_DEG and dihedral_rms < CONFORMER_DIHEDRAL_RMS_DEV_DEG


def _optimized_xyz_are_redundant(path_a: Path, path_b: Path) -> bool:
    dih_a = _dihedrals_from_xyz(path_a)
    dih_b = _dihedrals_from_xyz(path_b)
    if dih_a is not None and dih_b is not None and _dihedral_signatures_match(dih_a, dih_b):
        return True
    try:
        return _heavy_atom_rmsd_from_xyz(path_a, path_b) < CONFORMER_HEAVY_ATOM_RMSD_ANG
    except Exception:
        return False


def _prune_redundant_pool_entries(
    entries: list[ConformerPoolEntry],
    *,
    reference_mol: Chem.Mol,
    max_conformers: Optional[int] = None,
) -> list[ConformerPoolEntry]:
    valid: list[ConformerPoolEntry] = []
    for entry in entries:
        if not isinstance(entry.terms.solution_phase_free_energy_kcal_mol, (int, float)):
            continue
        if entry.opt_xyz_path is not None and not _xyz_connectivity_matches_reference(
            entry.opt_xyz_path,
            reference_mol,
        ):
            continue
        valid.append(entry)

    valid.sort(key=lambda entry: float(entry.terms.solution_phase_free_energy_kcal_mol))

    kept: list[ConformerPoolEntry] = []
    for entry in valid:
        if entry.opt_xyz_path is None:
            kept.append(entry)
            continue
        if any(
            other.opt_xyz_path is not None
            and _optimized_xyz_are_redundant(entry.opt_xyz_path, other.opt_xyz_path)
            for other in kept
        ):
            continue
        kept.append(entry)
        if max_conformers is not None and len(kept) >= max_conformers:
            break
    return kept


def _screening_terms_from_protomer(protomer: Protomer) -> Optional[ConformerEnergyTerms]:
    mol = protomer.mol
    if mol is None or not mol.HasProp("screening_solution_phase_free_energy_kcal_mol"):
        return None

    def _prop(name: str) -> EnergyListValue:
        if not mol.HasProp(name):
            return "not-run"
        try:
            return float(mol.GetDoubleProp(name))
        except ValueError:
            return mol.GetProp(name)

    return ConformerEnergyTerms(
        gas_sp_energy_kcal_mol=_prop("screening_gas_sp_energy_kcal_mol"),
        gas_sp_energy_xtb_kcal_mol=_prop("screening_gas_sp_energy_xtb_kcal_mol"),
        solvation_free_energy_kcal_mol=_prop("screening_solvation_free_energy_kcal_mol"),
        rrho_contribution_kcal_mol=_prop("screening_rrho_contribution_kcal_mol"),
        solution_phase_free_energy_kcal_mol=_prop("screening_solution_phase_free_energy_kcal_mol"),
        workflow_status="screening",
    )


def _screening_opt_xyz_path(protomer: Protomer) -> Optional[Path]:
    mol = protomer.mol
    if mol is None or not mol.HasProp("screening_opt_xyz_path"):
        return None
    path = Path(mol.GetProp("screening_opt_xyz_path"))
    return path if path.is_file() else None


def _embed_kdg_conformer(mol: Chem.Mol, *, random_seed: int = 42) -> Chem.Mol:
    mol_h = Chem.AddHs(Chem.Mol(mol))
    _ensure_ring_info(mol_h)
    params = _kdg_embed_parameters(random_seed=random_seed)
    conf_id = AllChem.EmbedMolecule(mol_h, params)
    if conf_id < 0:
        raise RuntimeError("RDKit KDG conformer embedding failed.")
    mol_out = Chem.RemoveHs(mol_h)
    return mol_out


def _generate_kdg_conformer_ensemble(
    mol: Chem.Mol,
    *,
    random_seed: int = 42,
    min_conformers: int = REFINEMENT_MIN_EMBED_CONFORMERS,
    max_conformers: int = REFINEMENT_MAX_EMBED_CONFORMERS,
) -> tuple[Chem.Mol, list[int]]:
    mol_h = Chem.AddHs(Chem.Mol(mol))
    _ensure_ring_info(mol_h)
    params = _kdg_embed_parameters(random_seed=random_seed)
    n_rotatable_bonds = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol_h)
    n_confs = max(int(min_conformers), min(2 ** n_rotatable_bonds, int(max_conformers)))
    conf_ids = list(AllChem.EmbedMultipleConfs(mol_h, int(n_confs), params))
    if not conf_ids:
        raise RuntimeError("RDKit KDG conformer embedding produced no conformers.")

    ranked: list[tuple[float, int]] = []
    for conf_id in conf_ids:
        energy = _mmff94_conformer_energy_kcal_mol(mol_h, conf_id)
        ranked.append((energy if energy is not None else float("inf"), int(conf_id)))
    ranked.sort(key=lambda row: row[0])
    return mol_h, [conf_id for _energy, conf_id in ranked]


def _select_lowest_embedded_conformers(
    mol_h: Chem.Mol,
    ranked_conf_ids: list[int],
    reference_mol: Chem.Mol,
    *,
    max_conformers: int,
) -> list[int]:
    selected: list[int] = []
    for conf_id in ranked_conf_ids:
        conf_mol = _mol_from_conf_id(mol_h, conf_id)
        if not _connectivity_matches_reference(conf_mol, reference_mol):
            continue
        if any(
            _conformers_are_redundant(mol_h, conf_id, kept_id)
            for kept_id in selected
        ):
            continue
        selected.append(int(conf_id))
        if len(selected) >= max_conformers:
            break
    return selected


def _mol_from_conf_id(mol_h: Chem.Mol, conf_id: int) -> Chem.Mol:
    mol_one = Chem.Mol(mol_h)
    mol_one.RemoveAllConformers()
    mol_one.AddConformer(mol_h.GetConformer(int(conf_id)), assignId=True)
    return Chem.RemoveHs(mol_one)


_RELAXED_OPT_WORKFLOW_STATUS_PREFIX = "optimization_retried_with_convergence:"


def _reset_optimization_artifacts(scratch_dir: Path) -> None:
    for name in ("xtbopt.xyz", "xtbopt.log", "xtbopt_run.log", "gxtbopt_run.log"):
        path = scratch_dir / name
        if path.exists():
            path.unlink()


def _mark_relaxed_optimization(
    protomer: Protomer,
    *,
    retried_opt_level: str,
    initial_opt_level: str,
    engine: str,
) -> None:
    _set_optimization_convergence_props(
        protomer,
        opt_level=retried_opt_level,
        initial_opt_level=initial_opt_level,
        engine=engine,
        relaxed_retry=True,
    )


def _set_optimization_convergence_props(
    protomer: Protomer,
    *,
    opt_level: str,
    initial_opt_level: str,
    engine: str,
    relaxed_retry: bool = False,
) -> None:
    if protomer.mol is None:
        return
    _set_mol_prop_str(protomer.mol, "optimization_opt_level", opt_level)
    _set_mol_prop_str(protomer.mol, "optimization_initial_opt_level", initial_opt_level)
    _set_mol_prop_str(protomer.mol, "optimization_engine", engine)
    if relaxed_retry:
        _set_mol_prop_str(
            protomer.mol,
            "workflow_status",
            f"{_RELAXED_OPT_WORKFLOW_STATUS_PREFIX}{opt_level}",
        )


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


def _has_connectivity_mismatch(protomer: Protomer, xyz_path: Path) -> bool:
    mol_opt = Chem.MolFromXYZFile(str(xyz_path))
    if mol_opt is None or mol_opt.GetNumConformers() == 0:
        return False
    rdDetermineBonds.DetermineConnectivity(mol_opt)
    input_mol = protomer.input_mol if getattr(protomer, "input_mol", None) is not None else protomer.mol
    if input_mol is None:
        return False
    input_mol_with_hydrogens = Chem.AddHs(input_mol)
    input_edges = _all_atom_connectivity_signature(input_mol_with_hydrogens)
    opt_edges = _all_atom_connectivity_signature(mol_opt)
    return input_edges != opt_edges


def _clear_connectivity_mismatch_flags(protomer: Protomer) -> None:
    if protomer.mol is None:
        return
    for key in ("connectivity_mismatch", "connectivity_mismatch_error"):
        if protomer.mol.HasProp(key):
            protomer.mol.ClearProp(key)


def _report_optimization_event(
    log_paths: list[Path],
    *,
    status: str,
    message: str,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> None:
    _log_status(log_paths, status, message)
    if progress_callback is not None:
        progress_callback(message)


def _run_optimization_with_convergence_retry(
    *,
    protomer: Protomer,
    scratch_dir: Path,
    opt_level: str,
    engine: str,
    log_paths: list[Path],
    run_at_level: Callable[[str], tuple[Path, Optional[float], Optional[float]]],
    progress_callback: Optional[Callable[[str], None]] = None,
) -> tuple[Path, Optional[float], Optional[float]]:
    levels = opt_convergence_retry_levels(opt_level)

    for attempt_idx, level in enumerate(levels):
        if attempt_idx > 0:
            _reset_optimization_artifacts(scratch_dir)
            _clear_connectivity_mismatch_flags(protomer)

        try:
            result = run_at_level(level)
        except (RuntimeError, FileNotFoundError) as exc:
            if attempt_idx < len(levels) - 1:
                next_level = levels[attempt_idx + 1]
                summary = (
                    f"{engine} optimization failed at convergence={level}; "
                    f"retrying from initial geometry with convergence={next_level}"
                )
                _log_status(log_paths, "WARN", f"{summary}: {exc}")
                if progress_callback is not None:
                    progress_callback(summary)
                continue
            raise

        opt_xyz_path, *_ = result
        if _has_connectivity_mismatch(protomer, opt_xyz_path):
            if attempt_idx < len(levels) - 1:
                next_level = levels[attempt_idx + 1]
                _report_optimization_event(
                    log_paths,
                    status="WARN",
                    message=(
                        f"{engine} optimization connectivity mismatch at convergence={level}; "
                        f"retrying from initial geometry with convergence={next_level}"
                    ),
                    progress_callback=progress_callback,
                )
                continue
            if attempt_idx > 0:
                _mark_relaxed_optimization(
                    protomer,
                    retried_opt_level=level,
                    initial_opt_level=opt_level,
                    engine=engine,
                )
                _report_optimization_event(
                    log_paths,
                    status="WARN",
                    message=(
                        f"{engine} optimization still has connectivity mismatch after "
                        f"relaxed retry at convergence={level} (initial={opt_level})"
                    ),
                    progress_callback=progress_callback,
                )
            return result

        if attempt_idx > 0:
            _mark_relaxed_optimization(
                protomer,
                retried_opt_level=level,
                initial_opt_level=opt_level,
                engine=engine,
            )
            _report_optimization_event(
                log_paths,
                status="OK",
                message=(
                    f"{engine} optimization succeeded with relaxed convergence={level} "
                    f"(initial={opt_level})"
                ),
                progress_callback=progress_callback,
            )
        else:
            _set_optimization_convergence_props(
                protomer,
                opt_level=level,
                initial_opt_level=opt_level,
                engine=engine,
            )
        return result

    raise RuntimeError(f"{engine} optimization failed")


def _run_xtb_optimization_with_retry(
    *,
    protomer: Protomer,
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
    progress_callback: Optional[Callable[[str], None]] = None,
) -> tuple[Path, Optional[float], Optional[float]]:
    return _run_optimization_with_convergence_retry(
        protomer=protomer,
        scratch_dir=scratch_dir,
        opt_level=opt_level,
        engine="xtb",
        log_paths=log_paths,
        progress_callback=progress_callback,
        run_at_level=lambda level: run_xtb_optimization(
            mol=mol,
            scratch_dir=scratch_dir,
            input_xyz_path=input_xyz_path,
            xtb_executable=xtb_executable,
            opt_level=level,
            charge=charge,
            solvent=solvent,
            timeout_s=timeout_s,
            dry_run=dry_run,
            log_paths=log_paths,
            run_command=_run,
            log_status=_log_status,
        ),
    )


def _run_gxtb_optimization_with_retry(
    *,
    protomer: Protomer,
    scratch_dir: Path,
    input_xyz_path: Path,
    input_mol: Optional[Chem.Mol],
    xtb_executable: str,
    xtb_version: XtbVersion,
    opt_level: str,
    charge: int,
    timeout_s: Optional[int],
    dry_run: bool,
    log_paths: list[Path],
    progress_callback: Optional[Callable[[str], None]] = None,
) -> tuple[Path, Optional[float], Optional[float]]:
    run_gxtb_opt = _resolve_gxtb_optimization_runner(xtb_version)

    def _run_at_level(level: str) -> tuple[Path, Optional[float], Optional[float]]:
        opt_kwargs = dict(
            scratch_dir=scratch_dir,
            xyz_path=input_xyz_path,
            xtb_executable=xtb_executable,
            opt_level=level,
            charge=charge,
            timeout_s=timeout_s,
            dry_run=dry_run,
            log_paths=log_paths,
            run_command=_run,
            log_status=_log_status,
        )
        if xtb_version == "default":
            opt_kwargs["input_mol"] = input_mol
        return run_gxtb_opt(**opt_kwargs)

    return _run_optimization_with_convergence_retry(
        protomer=protomer,
        scratch_dir=scratch_dir,
        opt_level=opt_level,
        engine="gxtb",
        log_paths=log_paths,
        progress_callback=progress_callback,
        run_at_level=_run_at_level,
    )

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

@dataclass(frozen=True)
class ConformerSearchResult:
    """Single-protomer conformer search output (extensible to multi-conformer ensembles)."""

    mol: Chem.Mol
    best_energy_kcal_mol: float
    best_conf_id: int
    conformer_energies_kcal_mol: dict[int, float]


def _search_conformers_rdkit_mmff94(
    mol: Chem.Mol,
    *,
    random_seed: int = 42,
    mmff_max_iters: int = 1000,
    max_conformers: Optional[int] = None,
) -> ConformerSearchResult:
    """
    Generate conformers with RDKit ETKDG and rank by MMFF94 energy.

    Returns the lowest-MMFF-energy structure plus per-conformer energies for
    optional downstream ensemble averaging.
    """
    if mol is None:
        raise ValueError("mol is None")

    mol_in = Chem.Mol(mol)
    mol_h = Chem.AddHs(mol_in, addCoords=False)
    _ensure_ring_info(mol_h)

    params = AllChem.ETKDGv3()
    params.randomSeed = int(random_seed)
    params.numThreads = 0
    params.useExpTorsionAnglePrefs = False # better for liq. phase

    n_rotatable_bonds = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol_h)
    n_confs = min(50, 2 ** n_rotatable_bonds, 500) 
    if max_conformers is not None:
        n_confs = min(int(n_confs), int(max_conformers))
    conf_ids = list(AllChem.EmbedMultipleConfs(mol_h, int(n_confs), params))
    if not conf_ids:
        raise RuntimeError("RDKit conformer embedding produced no conformers.")

    best_energy = float("inf")
    best_conf_id = conf_ids[0]
    conformer_energies: dict[int, float] = {}

    for conf_id in conf_ids:
        status = AllChem.MMFFOptimizeMolecule(
            mol_h,
            mmffVariant="MMFF94",
            maxIters=int(mmff_max_iters),
            confId=int(conf_id),
        )
        if status == -1:
            continue

        mmff_props = AllChem.MMFFGetMoleculeProperties(mol_h, mmffVariant="MMFF94")
        if mmff_props is None:
            raise RuntimeError("MMFF94 molecule properties could not be created.")
        ff = AllChem.MMFFGetMoleculeForceField(mol_h, mmff_props, confId=int(conf_id))
        ff.Initialize()
        energy = float(ff.CalcEnergy())
        conformer_energies[int(conf_id)] = energy
        if energy < best_energy:
            best_energy = energy
            best_conf_id = int(conf_id)

    if not conformer_energies:
        raise RuntimeError("MMFF94 optimization failed for all embedded conformers.")

    mol_best_h = Chem.Mol(mol_h)
    mol_best_h.RemoveAllConformers()
    mol_best_h.AddConformer(mol_h.GetConformer(best_conf_id), assignId=True)
    mol_no_h = Chem.RemoveHs(mol_best_h)

    return ConformerSearchResult(
        mol=mol_no_h,
        best_energy_kcal_mol=best_energy,
        best_conf_id=best_conf_id,
        conformer_energies_kcal_mol=conformer_energies,
    )


ProtomerRef = tuple[int, int, Protomer]


def run_batch_conformer_generation(
    protomer_refs: list[ProtomerRef],
    *,
    conformer_mode: ConformerMode,
    external_xyz_path: Optional[str | Path] = None,
    scratch_root: str | Path = "./scratch_conformers",
    random_seed: int = 42,
    dry_run: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> None:
    """
    Generate or attach 3D conformers for all protomers in one upfront step.

    Writes geometries onto each protomer.mol before any xTB/QM workflows run.
    """
    def _progress(message: str) -> None:
        if progress_callback is not None:
            progress_callback(message)

    scratch_root_path = Path(scratch_root)
    scratch_root_path.mkdir(parents=True, exist_ok=True)
    log_paths = [_species_workflow_log_path(scratch_root_path)]
    _log_status(log_paths, "START", f"batch conformer generation mode={conformer_mode} n_protomers={len(protomer_refs)}")

    if dry_run:
        _log_status(log_paths, "SKIP", "dry_run enabled; skipping batch conformer generation")
        return

    for taut_idx, prot_idx, protomer in protomer_refs:
        prefix = f"tautomer {taut_idx} protomer {prot_idx}"
        _progress(f"preparing conformer for {prefix}")
        try:
            mol, conformer_energy_kcal_mol = _prepare_protomer_conformer(
                protomer,
                conformer_mode=conformer_mode,
                external_xyz_path=external_xyz_path,
                log_paths=log_paths,
                random_seed=random_seed,
            )
            _set_mol_prop_double(protomer.mol, "conformer_energy_kcal_mol", conformer_energy_kcal_mol)
            _log_status(
                log_paths,
                "OK",
                f"{prefix} conformer ready energy_kcal_mol={conformer_energy_kcal_mol}",
            )
        except Exception as exc:
            _log_status(log_paths, "FAIL", f"{prefix} conformer generation failed: {exc}")
            _set_mol_prop_str(protomer.mol, "conformer_generation_error", str(exc)[:4000])
            if protomer.mol is not None:
                protomer.mol.SetProp("workflow_status", "conformer_generation_failed")

    _log_status(log_paths, "DONE", "batch conformer generation finished")


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
    conformer_mode: ConformerMode,
    external_xyz_path: Optional[str | Path],
    log_paths: list[Path],
    random_seed: int = 42,
) -> tuple[Chem.Mol, Optional[float]]:
    mol = protomer.mol
    if mol is None:
        raise ValueError("Protomer.mol is None; cannot run workflow.")
    if getattr(protomer, "input_mol", None) is None:
        protomer.input_mol = Chem.Mol(mol)

    _log_status(log_paths, "STEP", "starting conformer preparation")
    conformer_energy_kcal_mol: Optional[float] = None

    if conformer_mode == "kdg":
        protomer.mol = _embed_kdg_conformer(mol, random_seed=random_seed)
        protomer.mol.SetProp("conformer_mode", "kdg")
        protomer.mol.SetIntProp("conformer_count", 1)
        _log_status(log_paths, "OK", "rdkit KDG conformer embedding complete")
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

        if _has_connectivity_mismatch(protomer, xyz_path):
            input_mol = protomer.input_mol if getattr(protomer, "input_mol", None) is not None else protomer.mol
            input_mol_with_hydrogens = Chem.AddHs(input_mol)
            input_edges = _all_atom_connectivity_signature(input_mol_with_hydrogens)
            opt_edges = _all_atom_connectivity_signature(mol_opt)
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
        mol_opt.SetProp("connectivity_mismatch", "false")
        if protomer.mol is not None:
            protomer.mol.SetProp("connectivity_mismatch", "false")
        previous_mol = protomer.mol
        if previous_mol is not None:
            for key in ("conformer_energy_kcal_mol", "conformer_delta_kcal_mol", "gas_sp_energy_xtb_kcal_mol"):
                if previous_mol.HasProp(key):
                    try:
                        mol_opt.SetDoubleProp(key, float(previous_mol.GetDoubleProp(key)))
                    except (ValueError, KeyError, TypeError):
                        mol_opt.SetProp(key, previous_mol.GetProp(key))
            for key in (
                "optimization_opt_level",
                "optimization_initial_opt_level",
                "optimization_engine",
                "workflow_status",
            ):
                if previous_mol.HasProp(key):
                    mol_opt.SetProp(key, previous_mol.GetProp(key))
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
    gas_sp_energy_xtb_kcal_mol: Optional[float],
    rrho_contribution_kcal_mol: Optional[float],
    solution_phase_free_energy_kcal_mol: Optional[float],
) -> None:
    protomer.mol.SetProp("charge", str(charge))
    _set_mol_prop_double(protomer.mol, "conformer_energy_kcal_mol", conformer_energy_kcal_mol)
    _set_mol_prop_double(protomer.mol, "solvation_free_energy_kcal_mol", solvation_free_energy_kcal_mol)
    _set_mol_prop_double(protomer.mol, "gas_sp_energy_kcal_mol", gas_sp_energy_kcal_mol)
    _set_mol_prop_double(protomer.mol, "gas_sp_energy_xtb_kcal_mol", gas_sp_energy_xtb_kcal_mol)
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


def promote_screening_xtb_terms_to_final(
    protomer: Protomer,
    *,
    clear_gas_sp: bool = False,
) -> None:
    """
    Copy xTB screening solvation, RRHO, and GFN2-xTB Hessian gas terms onto final mol props.

    g-xTB gas-phase SP (``screening_gas_sp_energy_kcal_mol``) is intentionally not
    promoted: final ``gas_sp_energy_kcal_mol`` is reserved for post-screen g-xTB.
    """
    mol = protomer.mol
    if mol is None:
        return
    if mol.HasProp("screening_solvation_free_energy_kcal_mol"):
        _set_mol_prop_double(
            mol,
            "solvation_free_energy_kcal_mol",
            float(mol.GetDoubleProp("screening_solvation_free_energy_kcal_mol")),
        )
    if mol.HasProp("screening_rrho_contribution_kcal_mol"):
        _set_mol_prop_double(
            mol,
            "rrho_contribution_kcal_mol",
            float(mol.GetDoubleProp("screening_rrho_contribution_kcal_mol")),
        )
    if mol.HasProp("screening_gas_sp_energy_xtb_kcal_mol"):
        _set_mol_prop_double(
            mol,
            "gas_sp_energy_xtb_kcal_mol",
            float(mol.GetDoubleProp("screening_gas_sp_energy_xtb_kcal_mol")),
        )
    if clear_gas_sp and mol.HasProp("gas_sp_energy_kcal_mol"):
        mol.ClearProp("gas_sp_energy_kcal_mol")


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


def _persist_conformer_ensemble_results(
    protomer: Protomer,
    *,
    charge: int,
    conformer_terms: list[ConformerEnergyTerms],
    conformer_labels: Optional[list[str]] = None,
    temperature_k: float = 298.15,
) -> None:
    if protomer.mol is None:
        return

    gas_sp_values = [term.gas_sp_energy_kcal_mol for term in conformer_terms]
    gas_sp_xtb_values = [term.gas_sp_energy_xtb_kcal_mol for term in conformer_terms]
    solvation_values = [term.solvation_free_energy_kcal_mol for term in conformer_terms]
    rrho_values = [term.rrho_contribution_kcal_mol for term in conformer_terms]
    solution_values = [term.solution_phase_free_energy_kcal_mol for term in conformer_terms]

    _set_energy_list_prop(protomer.mol, "conformer_gas_sp_energy_kcal_mol_list", gas_sp_values)
    _set_energy_list_prop(protomer.mol, "conformer_gas_sp_energy_xtb_kcal_mol_list", gas_sp_xtb_values)
    _set_energy_list_prop(protomer.mol, "conformer_solvation_free_energy_kcal_mol_list", solvation_values)
    _set_energy_list_prop(protomer.mol, "conformer_rrho_contribution_kcal_mol_list", rrho_values)
    _set_energy_list_prop(protomer.mol, "conformer_solution_phase_free_energy_kcal_mol_list", solution_values)
    protomer.mol.SetIntProp("conformer_qm_count", len(conformer_terms))
    if conformer_labels is not None:
        _set_mol_prop_str(protomer.mol, "conformer_labels", json.dumps(conformer_labels))

    aggregate_solution = _boltzmann_aggregate_energy(solution_values, temperature_k=temperature_k)
    aggregate_gas = _boltzmann_aggregate_energy(gas_sp_values, temperature_k=temperature_k)
    aggregate_solv = _boltzmann_aggregate_energy(solvation_values, temperature_k=temperature_k)
    aggregate_rrho = _boltzmann_aggregate_energy(rrho_values, temperature_k=temperature_k)
    aggregate_gas_xtb = _boltzmann_aggregate_energy(
        [value for value in gas_sp_xtb_values if value is not None],
        temperature_k=temperature_k,
    )

    _persist_protomer_results(
        protomer,
        charge=charge,
        conformer_energy_kcal_mol=None,
        solvation_free_energy_kcal_mol=aggregate_solv,
        gas_sp_energy_kcal_mol=aggregate_gas,
        gas_sp_energy_xtb_kcal_mol=aggregate_gas_xtb,
        rrho_contribution_kcal_mol=aggregate_rrho,
        solution_phase_free_energy_kcal_mol=aggregate_solution,
    )


def _run_single_conformer_workflow(
    protomer: Protomer,
    mol: Chem.Mol,
    *,
    scratch_dir: Path,
    xtb_executable: str,
    xtb_version: XtbVersion,
    charge: int,
    solvent: str,
    gfn: int,
    opt_level: str,
    optimization_engine: Literal["xtb", "aimnet2"],
    dry_run: bool,
    timeout_s: Optional[int],
    log_paths: list[Path],
    progress_callback: Optional[Callable[[str], None]] = None,
) -> ConformerWorkflowResult:
    def _progress(message: str) -> None:
        if progress_callback is not None:
            progress_callback(message)

    terms = ConformerEnergyTerms(gas_sp_energy_kcal_mol="not-run")
    scratch_dir.mkdir(parents=True, exist_ok=True)

    try:
        input_xyz_path = _write_workflow_inputs(mol, scratch_dir, charge, log_paths)
        _progress("optimizing geometry with GFN2-xTB")
        if optimization_engine == "xtb":
            opt_xyz_path, _opt_gas_sp_kcal_mol, _opt_gas_sp_h = _run_xtb_optimization_with_retry(
                protomer=protomer,
                mol=mol,
                scratch_dir=scratch_dir,
                input_xyz_path=input_xyz_path,
                xtb_executable=xtb_executable,
                opt_level=opt_level,
                charge=charge,
                solvent=solvent,
                timeout_s=timeout_s,
                dry_run=dry_run,
                log_paths=log_paths,
                progress_callback=progress_callback,
            )
        else:
            opt_xyz_path, _opt_gas_sp_kcal_mol, _opt_gas_sp_h = run_aimnet2_optimization(
                scratch_dir=scratch_dir,
                input_xyz_path=input_xyz_path,
                charge=charge,
                dry_run=dry_run,
                log_paths=log_paths,
                log_status=_log_status,
            )

        _, has_connectivity_mismatch = _update_protomer_geometry_from_xyz(
            protomer,
            opt_xyz_path,
            log_paths,
        )
        active_xyz_path = input_xyz_path if has_connectivity_mismatch else opt_xyz_path
        if has_connectivity_mismatch:
            terms.workflow_status = "connectivity-failed"
            terms.gas_sp_energy_kcal_mol = _format_energy_entry(None, failed_step="connectivity")
            terms.solvation_free_energy_kcal_mol = _format_energy_entry(None, failed_step="connectivity")
            terms.rrho_contribution_kcal_mol = _format_energy_entry(None, failed_step="connectivity")
            terms.solution_phase_free_energy_kcal_mol = _format_energy_entry(None, failed_step="connectivity")
            return ConformerWorkflowResult(terms=terms, opt_xyz_path=active_xyz_path)

        _progress("computing g-xTB gas-phase single point")
        run_gxtb_sp = _resolve_gxtb_single_point_runner(xtb_version)
        gas_sp_energy_kcal_mol, _ = run_gxtb_sp(
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
        terms.gas_sp_energy_kcal_mol = _format_energy_entry(
            gas_sp_energy_kcal_mol,
            failed_step="gxtb-sp",
        )

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
        terms.solvation_free_energy_kcal_mol = _format_energy_entry(
            solvation_free_energy_kcal_mol,
            failed_step="solvation",
        )

        _progress("computing RRHO contribution with xTB frequencies")
        gas_sp_energy_xtb_kcal_mol, rrho_contribution_kcal_mol, _ = run_hessian_and_parse_energies(
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
        terms.gas_sp_energy_xtb_kcal_mol = _format_energy_entry(
            gas_sp_energy_xtb_kcal_mol,
            failed_step="frequency",
        )
        terms.rrho_contribution_kcal_mol = _format_energy_entry(
            rrho_contribution_kcal_mol,
            failed_step="frequency",
        )

        if (
            isinstance(gas_sp_energy_kcal_mol, (int, float))
            and isinstance(solvation_free_energy_kcal_mol, (int, float))
            and isinstance(rrho_contribution_kcal_mol, (int, float))
        ):
            solution_phase_free_energy_kcal_mol = _compute_solution_phase_energy(
                gas_sp_energy_kcal_mol,
                solvation_free_energy_kcal_mol,
                rrho_contribution_kcal_mol,
                log_paths,
            )
            terms.solution_phase_free_energy_kcal_mol = _format_energy_entry(
                solution_phase_free_energy_kcal_mol,
                failed_step="solution-phase",
            )
            terms.workflow_status = "ok"
        else:
            terms.solution_phase_free_energy_kcal_mol = _format_energy_entry(
                None,
                failed_step="solution-phase",
            )
            terms.workflow_status = "partial-failed"

        return ConformerWorkflowResult(terms=terms, opt_xyz_path=active_xyz_path)
    except Exception as exc:
        _log_status(log_paths, "FAIL", f"single-conformer workflow failed: {exc}")
        terms.workflow_status = "optimization-failed"
        terms.gas_sp_energy_kcal_mol = _format_energy_entry(None, failed_step="optimization")
        terms.solvation_free_energy_kcal_mol = _format_energy_entry(None, failed_step="optimization")
        terms.rrho_contribution_kcal_mol = _format_energy_entry(None, failed_step="optimization")
        terms.solution_phase_free_energy_kcal_mol = _format_energy_entry(None, failed_step="optimization")
        return ConformerWorkflowResult(terms=terms, opt_xyz_path=None)


def run_protomer_screening(
    protomer: Protomer,
    *,
    protomer_id: int | str = 0,
    scratch_root: str | Path = "./scratch_solvation",
    xtb_executable: str,
    xtb_version: XtbVersion = "default",
    conformer_mode: ConformerMode = "kdg",
    external_xyz_path: Optional[str | Path] = None,
    charge_override: Optional[int] = None,
    solvent: Literal["water"] = "water",
    gfn: int = 2,
    optimization_engine: Literal["xtb", "aimnet2"] = "xtb",
    opt_level: str = "loose",
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
    1) Build/choose an initial KDG conformer geometry.
    2) Run GFN2-xTB geometry optimization in implicit solvent.
    3) Run g-xTB gas-phase single point on the optimized geometry.
    4) Run CPCM-X solvation and xTB Hessian (RRHO) on the optimized geometry.
    5) Compute screening solution-phase free energy.
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
        f"screening protomer_id={protomer_id} scratch_dir={scratch_dir.name} charge={charge} conformer_mode={conformer_mode} xtb_version={xtb_version}",
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
        _set_mol_prop_str(protomer.mol, "screening_optimization_engine", optimization_engine)

        workflow_result = _run_single_conformer_workflow(
            protomer,
            mol,
            scratch_dir=scratch_dir,
            xtb_executable=xtb_executable,
            xtb_version=xtb_version,
            charge=charge,
            solvent=solvent,
            gfn=gfn,
            opt_level=opt_level,
            optimization_engine=optimization_engine,
            dry_run=dry_run,
            timeout_s=timeout_s,
            log_paths=log_paths,
            progress_callback=_progress,
        )
        terms = workflow_result.terms
        _persist_conformer_ensemble_results(
            protomer,
            charge=charge,
            conformer_terms=[terms],
        )

        gas_sp_energy_kcal_mol = (
            float(terms.gas_sp_energy_kcal_mol)
            if isinstance(terms.gas_sp_energy_kcal_mol, (int, float))
            else None
        )
        gas_sp_energy_xtb_kcal_mol = (
            float(terms.gas_sp_energy_xtb_kcal_mol)
            if isinstance(terms.gas_sp_energy_xtb_kcal_mol, (int, float))
            else None
        )
        solvation_free_energy_kcal_mol = (
            float(terms.solvation_free_energy_kcal_mol)
            if isinstance(terms.solvation_free_energy_kcal_mol, (int, float))
            else None
        )
        rrho_contribution_kcal_mol = (
            float(terms.rrho_contribution_kcal_mol)
            if isinstance(terms.rrho_contribution_kcal_mol, (int, float))
            else None
        )
        solution_phase_free_energy_kcal_mol = (
            float(terms.solution_phase_free_energy_kcal_mol)
            if isinstance(terms.solution_phase_free_energy_kcal_mol, (int, float))
            else None
        )

        _set_mol_prop_double(protomer.mol, "screening_conformer_energy_kcal_mol", conformer_energy_kcal_mol)
        _set_mol_prop_double(protomer.mol, "screening_solvation_free_energy_kcal_mol", solvation_free_energy_kcal_mol)
        _set_mol_prop_double(protomer.mol, "screening_gas_sp_energy_kcal_mol", gas_sp_energy_kcal_mol)
        _set_mol_prop_double(protomer.mol, "screening_gas_sp_energy_xtb_kcal_mol", gas_sp_energy_xtb_kcal_mol)
        _set_mol_prop_double(protomer.mol, "screening_rrho_contribution_kcal_mol", rrho_contribution_kcal_mol)
        _set_mol_prop_double(
            protomer.mol,
            "screening_solution_phase_free_energy_kcal_mol",
            solution_phase_free_energy_kcal_mol,
        )
        if workflow_result.opt_xyz_path is not None and workflow_result.opt_xyz_path.is_file():
            _set_mol_prop_str(protomer.mol, "screening_opt_xyz_path", str(workflow_result.opt_xyz_path))
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
    if _preserved_xtbopt is not None and protomer.mol is not None:
        _set_mol_prop_str(protomer.mol, "screening_opt_xyz_path", str(_preserved_xtbopt))
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
    xtb_executable: str,
    xtb_version: XtbVersion = "default",
    conformer_mode: ConformerMode = "skip_search",
    external_xyz_path: Optional[str | Path] = None,
    optimization_engine: Literal["xtb", "aimnet2"] = "xtb",
    charge_override: Optional[int] = None,
    solvent: Literal["water"] = "water",
    gfn: int = 2,
    opt_level: str = "loose",
    sp_energy: Literal["gxtb", "xtb", "aimnet2"] = "gxtb",
    gxtb_post_optimize: bool = False,
    recompute_solvation: bool = False,
    recompute_frequencies: bool = False,
    reuse_screening_terms: bool = True,
    keep_scratch: bool = False,
    keep_logs: bool = False,
    keep_scratch_on_failure: bool = False,
    dry_run: bool = False,
    timeout_s: Optional[int] = None,
    random_seed: int = 42,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> SolvationWorkflowResult:
    """
    Conformer-refinement workflow for protomers that pass screening.

    Generates a KDG conformer ensemble ranked by MMFF94, prunes redundant
    geometries (dihedral/RMSD plus connectivity), runs GFN2-xTB/CPCM-X
    optimization plus g-xTB SP/solvation/frequencies on up to 10 conformers,
    merges the screening conformer into the pool, prunes optimized structures
    again (dihedrals from xyz2mol connectivity, XYZ RMSD fallback), and
    Boltzmann-weights the surviving conformer solution-phase energies.
    """
    def _progress(message: str) -> None:
        if progress_callback is not None:
            progress_callback(message)

    scratch_context = _create_scratch_context(scratch_root, protomer_id)
    scratch_dir = scratch_context.scratch_dir
    log_paths = scratch_context.log_paths

    if protomer.mol is None:
        raise ValueError("Protomer does not have mol; cannot run conformer refinement.")
    charge = int(charge_override) if charge_override is not None else _formal_charge(protomer.mol)
    _log_status(
        log_paths,
        "START",
        f"conformer refinement protomer_id={protomer_id} scratch_dir={scratch_dir.name} charge={charge}",
    )

    conformer_energy_kcal_mol: Optional[float] = None
    solvation_free_energy_kcal_mol: Optional[float] = None
    gas_sp_energy_kcal_mol: Optional[float] = None
    rrho_contribution_kcal_mol: Optional[float] = None
    solution_phase_free_energy_kcal_mol: Optional[float] = None
    final_opt_xyz: Optional[Path] = None

    try:
        if dry_run:
            _progress("dry run enabled; skipping conformer refinement")
            _log_status(log_paths, "SKIP", "dry_run enabled; skipping conformer refinement")
            return SolvationWorkflowResult(
                conformer_energy_kcal_mol=None,
                xtb_optimized_xyz=None,
                solvation_free_energy_kcal_mol=None,
                gas_sp_energy_kcal_mol=None,
                rrho_contribution_kcal_mol=None,
                solution_phase_free_energy_kcal_mol=None,
                stdout_tail="dry_run; skipped conformer refinement.",
            )

        if sp_energy != "gxtb":
            _log_status(
                log_paths,
                "WARN",
                f"conformer refinement always uses g-xTB gas-phase SP; ignoring sp_energy={sp_energy}",
            )
        if gxtb_post_optimize or recompute_solvation or recompute_frequencies or reuse_screening_terms:
            _log_status(
                log_paths,
                "WARN",
                "legacy post-screen flags (gxtb_post_optimize/recompute_*/reuse_screening_terms) are ignored",
            )

        graph_mol = Chem.MolFromSmiles(protomer.smiles)
        if graph_mol is None:
            raise ValueError(f"Could not rebuild 3D graph from SMILES: {protomer.smiles}")
        _ensure_ring_info(graph_mol)

        _progress("generating KDG conformer ensemble")
        mol_h, ranked_conf_ids = _generate_kdg_conformer_ensemble(
            graph_mol,
            random_seed=random_seed,
        )
        selected_conf_ids = _select_lowest_embedded_conformers(
            mol_h,
            ranked_conf_ids,
            graph_mol,
            max_conformers=REFINEMENT_MAX_QM_CONFORMERS,
        )
        _log_status(
            log_paths,
            "OK",
            f"embedded={len(ranked_conf_ids)} selected_for_qm={len(selected_conf_ids)}",
        )

        pool_entries: list[ConformerPoolEntry] = []
        screening_terms = _screening_terms_from_protomer(protomer)
        screening_xyz = _screening_opt_xyz_path(protomer)
        if (
            screening_terms is not None
            and isinstance(screening_terms.solution_phase_free_energy_kcal_mol, (int, float))
        ):
            pool_entries.append(
                ConformerPoolEntry(
                    label="screening",
                    terms=screening_terms,
                    opt_xyz_path=screening_xyz,
                )
            )

        for conf_idx, conf_id in enumerate(selected_conf_ids):
            conf_mol = _mol_from_conf_id(mol_h, conf_id)
            conf_scratch = scratch_dir / f"conformer_{conf_idx}"
            conf_protomer = copy.deepcopy(protomer)
            conf_protomer.mol = conf_mol
            _progress(f"running QM workflow for conformer {conf_idx + 1}/{len(selected_conf_ids)}")
            workflow_result = _run_single_conformer_workflow(
                conf_protomer,
                conf_mol,
                scratch_dir=conf_scratch,
                xtb_executable=xtb_executable,
                xtb_version=xtb_version,
                charge=charge,
                solvent=solvent,
                gfn=gfn,
                opt_level=opt_level,
                optimization_engine=optimization_engine,
                dry_run=dry_run,
                timeout_s=timeout_s,
                log_paths=log_paths,
                progress_callback=progress_callback,
            )
            if isinstance(workflow_result.terms.solution_phase_free_energy_kcal_mol, (int, float)):
                pool_entries.append(
                    ConformerPoolEntry(
                        label=f"refinement_{conf_idx}",
                        terms=workflow_result.terms,
                        opt_xyz_path=workflow_result.opt_xyz_path,
                    )
                )

        pruned_pool = _prune_redundant_pool_entries(
            pool_entries,
            reference_mol=graph_mol,
        )
        conformer_terms = [entry.terms for entry in pruned_pool]
        conformer_labels = [entry.label for entry in pruned_pool]
        _log_status(
            log_paths,
            "OK",
            f"pool_before_prune={len(pool_entries)} pool_after_prune={len(pruned_pool)}",
        )

        if pruned_pool:
            lowest_entry = min(
                pruned_pool,
                key=lambda entry: float(entry.terms.solution_phase_free_energy_kcal_mol),
            )
            final_opt_xyz = lowest_entry.opt_xyz_path
            if final_opt_xyz is not None and final_opt_xyz.is_file():
                opt_mol = _mol_from_xyz_with_connectivity(final_opt_xyz)
                if opt_mol is not None:
                    protomer.mol = opt_mol
                    protomer.mol.SetProp("connectivity_mismatch", "false")

        _persist_conformer_ensemble_results(
            protomer,
            charge=charge,
            conformer_terms=conformer_terms,
            conformer_labels=conformer_labels,
        )
        _set_mol_prop_str(protomer.mol, "workflow_status", "conformer_refined")

        if protomer.mol is not None and protomer.mol.HasProp("solution_phase_free_energy_kcal_mol"):
            solution_phase_free_energy_kcal_mol = float(
                protomer.mol.GetDoubleProp("solution_phase_free_energy_kcal_mol")
            )
        if protomer.mol is not None and protomer.mol.HasProp("gas_sp_energy_kcal_mol"):
            gas_sp_energy_kcal_mol = float(protomer.mol.GetDoubleProp("gas_sp_energy_kcal_mol"))
        if protomer.mol is not None and protomer.mol.HasProp("solvation_free_energy_kcal_mol"):
            solvation_free_energy_kcal_mol = float(
                protomer.mol.GetDoubleProp("solvation_free_energy_kcal_mol")
            )
        if protomer.mol is not None and protomer.mol.HasProp("rrho_contribution_kcal_mol"):
            rrho_contribution_kcal_mol = float(
                protomer.mol.GetDoubleProp("rrho_contribution_kcal_mol")
            )
        _progress("finished conformer refinement")

    except Exception as exc:
        _progress(f"failed: {exc}")
        _log_status(log_paths, "FAIL", f"conformer refinement exception for protomer_id={protomer_id}: {exc}")
        warnings.warn(
            f"Conformer refinement failed for protomer_id={protomer_id}: {exc}",
            RuntimeWarning,
        )
        if protomer.mol is not None:
            _set_mol_prop_str(protomer.mol, "workflow_error", str(exc)[:4000])
        preserved_xtbopt_path = _preserve_output_files(scratch_dir, keep_logs=keep_logs)
        _cleanup_scratch_dir(
            scratch_dir,
            keep_scratch=keep_scratch or keep_scratch_on_failure,
            dry_run=dry_run,
            log_paths=log_paths,
            success=False,
        )
        return SolvationWorkflowResult(
            conformer_energy_kcal_mol=conformer_energy_kcal_mol,
            xtb_optimized_xyz=preserved_xtbopt_path if preserved_xtbopt_path is not None else final_opt_xyz,
            solvation_free_energy_kcal_mol=solvation_free_energy_kcal_mol,
            gas_sp_energy_kcal_mol=gas_sp_energy_kcal_mol,
            rrho_contribution_kcal_mol=rrho_contribution_kcal_mol,
            solution_phase_free_energy_kcal_mol=solution_phase_free_energy_kcal_mol,
            stdout_tail=str(exc)[-4000:],
        )

    final_xtbopt_path = _preserve_output_files(scratch_dir, keep_logs=keep_logs)
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
        f"protomer_id={protomer_id} n_conformers={len(conformer_terms)} "
        f"solution={solution_phase_free_energy_kcal_mol}",
    )
    _progress("success")

    return SolvationWorkflowResult(
        conformer_energy_kcal_mol=conformer_energy_kcal_mol,
        xtb_optimized_xyz=final_xtbopt_path if final_xtbopt_path is not None else final_opt_xyz,
        solvation_free_energy_kcal_mol=solvation_free_energy_kcal_mol,
        gas_sp_energy_kcal_mol=gas_sp_energy_kcal_mol,
        rrho_contribution_kcal_mol=rrho_contribution_kcal_mol,
        solution_phase_free_energy_kcal_mol=solution_phase_free_energy_kcal_mol,
        stdout_tail=(
            f"conformer refinement ok; n_conformers={len(conformer_terms)} "
            f"solution={solution_phase_free_energy_kcal_mol}"
        )[-4000:],
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
