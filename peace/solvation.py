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

import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem, rdDetermineBonds

from prism_pruner.pruner import prune

from .logging_utils import LogLevel, emit_progress, record_user_warning
from .calculators import (
    CalculatorSettings,
    XtbFatalError,
    orca_opt_keyword,
    orca_solvent_keyword,
    report_xtb_fatal_and_exit,
    run_aimnet2_optimization,
    run_aimnet2_single_point_energy,
    run_cpcmx_single_point,
    run_gxtb_hessian_and_parse_energies,
    run_gxtb_single_point_energy,
    run_gxtb2_hessian_and_parse_energies,
    run_gxtb2_optimization,
    run_gxtb2_single_point_energy,
    run_gxtb_optimization,
    run_hessian_and_parse_energies,
    run_orca_frequencies,
    run_orca_optimization,
    run_orca_single_point_energy,
    run_orca_solvation_single_point,
    run_xtb_command,
    run_xtb_optimization,
    run_xtb_single_point_energy,
    spin_multiplicity,
)
from .calculators.xtb import XTB_SCF_RETRY_MARKER_FILE
from .calculators.common import DEFAULT_TEMPERATURE_K, opt_convergence_retry_levels
from .calculators.settings import XtbVersion
from .protomer import Protomer, Species, Tautomer
from .solvents import SolventNames, resolve_solvent

HARTREE_TO_KCAL_MOL = 627.5094740631
KCAL_MOL_PER_K = 0.00198720425864083

ConformerMode = Literal["kdg", "external_xyz", "skip_search"]

REFINEMENT_MIN_EMBED_CONFORMERS = 200
REFINEMENT_MAX_QM_CONFORMERS = 20
REFINEMENT_MAX_EMBED_CONFORMERS = 500
DEFAULT_CONFORMER_ENERGY_THRESHOLD_KCAL_MOL = 25.0

EnergyListValue = Union[float, str]


def _resolve_gxtb_single_point_runner(xtb_version: XtbVersion):
    if xtb_version == "default":
        return run_gxtb2_single_point_energy
    return run_gxtb_single_point_energy


def _resolve_gxtb_optimization_runner(xtb_version: XtbVersion):
    if xtb_version == "default":
        return run_gxtb2_optimization
    return run_gxtb_optimization


def _resolve_gxtb_hessian_runner(xtb_version: XtbVersion):
    if xtb_version == "default":
        return run_gxtb2_hessian_and_parse_energies
    return run_gxtb_hessian_and_parse_energies


@dataclass
class ConformerEnergyTerms:
    """Per-conformer solvation workflow terms (numeric or failure token)."""

    gas_sp_energy_kcal_mol: EnergyListValue
    freq_electronic_energy_kcal_mol: Optional[EnergyListValue] = None
    solvation_free_energy_kcal_mol: EnergyListValue = "not-run"
    rrho_contribution_kcal_mol: EnergyListValue = "not-run"
    solution_phase_free_energy_kcal_mol: EnergyListValue = "not-run"
    workflow_status: str = "not-run"


@dataclass
class ConformerWorkflowResult:
    terms: ConformerEnergyTerms
    opt_xyz_path: Optional[Path] = None


@dataclass(frozen=True)
class GasRefinementResult:
    gas_sp_energy_kcal_mol: Optional[float]
    gas_opt_xyz_path: Optional[Path] = None


def _input_mol_for_connectivity(protomer: Protomer, mol: Chem.Mol) -> Chem.Mol:
    input_mol = getattr(protomer, "input_mol", None)
    return input_mol if input_mol is not None else mol


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
    temperature_k: float = DEFAULT_TEMPERATURE_K,
) -> Optional[float]:
    if temperature_k <= 0:
        raise ValueError(f"temperature_k must be > 0 K, got {temperature_k}")
    numeric = _numeric_energy_values(energies)
    if not numeric:
        return None
    if len(numeric) == 1:
        return numeric[0]
    rt = KCAL_MOL_PER_K * float(temperature_k)
    min_energy = min(numeric)
    log_sum = math.log(sum(math.exp(-(energy - min_energy) / rt) for energy in numeric))
    return min_energy - rt * log_sum


def _prism_atoms_from_mol_h(mol_h: Chem.Mol) -> np.ndarray:
    return np.asarray([atom.GetSymbol() for atom in mol_h.GetAtoms()], dtype=str)


def _prism_coords_from_conf(mol_h: Chem.Mol, conf_id: int) -> np.ndarray:
    conf = mol_h.GetConformer(int(conf_id))
    return np.asarray(
        [list(conf.GetAtomPosition(i)) for i in range(mol_h.GetNumAtoms())],
        dtype=float,
    )


def _prism_atoms_and_coords_from_xyz(xyz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    lines = xyz_path.read_text().strip().splitlines()
    n_atoms = int(lines[0].strip())
    atoms: list[str] = []
    coords: list[list[float]] = []
    for line in lines[2 : 2 + n_atoms]:
        parts = line.split()
        atoms.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.asarray(atoms, dtype=str), np.asarray(coords, dtype=float)


def _prism_prune_structure_mask(
    structures: np.ndarray,
    atoms: np.ndarray,
    *,
    energies: Optional[np.ndarray] = None,
    max_dE: float = float("inf"),
    log_paths: Optional[list[Path]] = None,
) -> np.ndarray:
    if len(structures) <= 1:
        return np.ones(len(structures), dtype=bool)

    _, mask = prune(
        structures,
        atoms,
        moi_pruning=True,
        rmsd_pruning=True,
        rot_corr_rmsd_pruning=True,
        energies=energies,
        max_dE=max_dE,
        debugfunction=(
            (lambda msg: _log_status(log_paths, "DEBUG", msg)) if log_paths is not None else None
        ),
        logfunction=None,
    )
    return mask


def _prune_redundant_conf_ids(
    mol_h: Chem.Mol,
    ranked_conf_ids: list[int],
    *,
    log_paths: Optional[list[Path]] = None,
) -> list[int]:
    conf_ids = [int(conf_id) for conf_id in ranked_conf_ids]
    if len(conf_ids) <= 1:
        return conf_ids

    atoms = _prism_atoms_from_mol_h(mol_h)
    structures = np.stack([_prism_coords_from_conf(mol_h, conf_id) for conf_id in conf_ids])
    energies = np.asarray(
        [_mmff94_conformer_energy_kcal_mol(mol_h, conf_id) or float("inf") for conf_id in conf_ids],
        dtype=float,
    )
    mask = _prism_prune_structure_mask(
        structures,
        atoms,
        energies=energies,
        log_paths=log_paths,
    )
    pruned = [conf_id for conf_id, keep in zip(conf_ids, mask) if keep]
    if log_paths is not None and len(pruned) < len(conf_ids):
        _log_status(
            log_paths,
            "OK",
            f"prism-pruner removed {len(conf_ids) - len(pruned)} redundant embedded conformer(s)",
        )
    return pruned


@dataclass
class ConformerPoolEntry:
    label: str
    terms: ConformerEnergyTerms
    opt_xyz_path: Optional[Path]
    conformer_index: Optional[int] = None


def _workflow_log_prefix(
    charge_state: int,
    taut_idx: int,
    n_taut: int,
    prot_idx: int,
    n_prot: int,
    *,
    conf_idx: Optional[int] = None,
    n_conf: Optional[int] = None,
) -> str:
    parts = [
        f"chrg {charge_state:+d}",
        f"taut {taut_idx + 1}/{n_taut}",
        f"prot {prot_idx + 1}/{n_prot}",
    ]
    if conf_idx is not None and n_conf is not None:
        parts.append(f"conf {conf_idx + 1}/{n_conf}")
    return " ".join(parts)


def _format_solution_energy(value: EnergyListValue) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.2f} kcal/mol"
    return str(value)


def _log_conformer_summary(
    log_paths: list[Path],
    *,
    log_prefix: Optional[str],
    run_records: list[tuple[str, ConformerEnergyTerms]],
    pruned_pool: list[ConformerPoolEntry],
    aggregate_solution_energy: Optional[float],
    progress_callback: Optional[Callable[[str], None]] = None,
) -> None:
    header = f"{log_prefix} " if log_prefix else ""
    lines = [f"{header}conformer summary ({len(run_records)} run(s)):"]
    for label, terms in run_records:
        energy = _format_solution_energy(terms.solution_phase_free_energy_kcal_mol)
        lines.append(f"  {label}: {energy} ({terms.workflow_status})")
    if pruned_pool:
        pool_parts = [
            f"{entry.label}={_format_solution_energy(entry.terms.solution_phase_free_energy_kcal_mol)}"
            for entry in pruned_pool
        ]
        lines.append(f"  pool ({len(pruned_pool)}): {', '.join(pool_parts)}")
    if aggregate_solution_energy is not None:
        lines.append(f"  Boltzmann aggregate: {aggregate_solution_energy:.2f} kcal/mol")
    summary = "\n".join(lines)
    _log_status(log_paths, "OK", summary.replace("\n", " | "))
    if progress_callback is not None:
        for line in lines:
            emit_progress(progress_callback, line, level=LogLevel.VERBOSE)


def _protomer_has_zwitterionic_xh_sites(mol: Chem.Mol) -> bool:
    """Return True when the protomer has +1 heavy atoms bearing hydrogens (e.g. ammonium)."""
    mol_h = Chem.AddHs(Chem.Mol(mol), addCoords=True)
    for atom in mol_h.GetAtoms():
        if atom.GetAtomicNum() == 1 or atom.GetFormalCharge() != 1:
            continue
        if any(nbr.GetAtomicNum() == 1 for nbr in atom.GetNeighbors()):
            return True
    return False


def _effective_gas_optimize(mol: Chem.Mol, *, gas_optimize: bool) -> bool:
    """
    Zwitterionic X-H sites need a gas-phase re-optimization after the implicit-solvent
    optimization; evaluating the gas SP directly on that geometry gives unphysical gas energies.
    """
    if gas_optimize:
        return True
    return _protomer_has_zwitterionic_xh_sites(mol)


def _kdg_embed_parameters(*, random_seed: int) -> AllChem.EmbedParameters:
    params = AllChem.KDG()
    params.randomSeed = int(random_seed)
    params.numThreads = 0
    params.useRandomCoords = False
    params.useExpTorsionAnglePrefs = False
    return params


def _mmff94_conformer_energy_kcal_mol(mol_h: Chem.Mol, conf_id: int) -> Optional[float]:
    try:
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


def _optimize_mmff94_conformers(
    mol_h: Chem.Mol,
    conf_ids: list[int],
    *,
    log_paths: Optional[list[Path]] = None,
) -> None:
    """Relax embedded conformers with the RDKit MMFF94 force field before energy ranking."""
    if not conf_ids:
        return
    try:
        n_failed = 0
        for conf_id in conf_ids:
            code = AllChem.MMFFOptimizeMolecule(
                mol_h,
                confId=int(conf_id),
                mmffVariant="MMFF94",
            )
            if int(code) != 0:
                n_failed += 1
        if log_paths is not None:
            _log_status(
                log_paths,
                "OK",
                f"MMFF94 optimization complete optimized={len(conf_ids) - n_failed} failed={n_failed}",
            )
    except Exception as exc:
        if log_paths is not None:
            _log_status(log_paths, "WARN", f"MMFF94 conformer optimization failed: {exc}")


def _connectivity_signature_for_mol(mol: Chem.Mol) -> set[tuple[int, int]]:
    return _all_atom_connectivity_signature(Chem.AddHs(Chem.Mol(mol)))


def _connectivity_matches_reference(mol: Chem.Mol, reference_mol: Chem.Mol) -> bool:
    return _connectivity_signature_for_mol(mol) == _connectivity_signature_for_mol(reference_mol)


def _mol_from_xyz_with_connectivity(xyz_path: Path) -> Optional[Chem.Mol]:
    mol = Chem.MolFromXYZFile(str(xyz_path))
    if mol is None or mol.GetNumConformers() == 0:
        return None
    rdDetermineBonds.DetermineConnectivity(mol)
    return mol


def _xyz_connectivity_matches_reference(xyz_path: Path, reference_mol: Chem.Mol) -> bool:
    mol = _mol_from_xyz_with_connectivity(xyz_path)
    if mol is None:
        return False
    return _connectivity_matches_reference(mol, reference_mol)


def _prune_redundant_pool_entries(
    entries: list[ConformerPoolEntry],
    *,
    reference_mol: Chem.Mol,
    max_conformers: Optional[int] = None,
    log_paths: Optional[list[Path]] = None,
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

    entries_without_xyz = [entry for entry in valid if entry.opt_xyz_path is None]
    entries_with_xyz = [entry for entry in valid if entry.opt_xyz_path is not None]

    pruned_with_xyz: list[ConformerPoolEntry] = []
    if entries_with_xyz:
        atoms: Optional[np.ndarray] = None
        structures: list[np.ndarray] = []
        energies: list[float] = []
        kept_indices: list[int] = []
        for idx, entry in enumerate(entries_with_xyz):
            entry_atoms, coords = _prism_atoms_and_coords_from_xyz(entry.opt_xyz_path)
            if atoms is None:
                atoms = entry_atoms
            elif not np.array_equal(atoms, entry_atoms):
                continue
            structures.append(coords)
            energies.append(float(entry.terms.solution_phase_free_energy_kcal_mol))
            kept_indices.append(idx)

        if atoms is not None and structures:
            structures_arr = np.stack(structures)
            energies_arr = np.asarray(energies, dtype=float)
            mask = _prism_prune_structure_mask(
                structures_arr,
                atoms,
                energies=energies_arr,
                log_paths=log_paths,
            )
            pruned_with_xyz = [
                entries_with_xyz[kept_indices[i]]
                for i, keep in enumerate(mask)
                if keep
            ]
            if log_paths is not None and len(pruned_with_xyz) < len(entries_with_xyz):
                _log_status(
                    log_paths,
                    "OK",
                    (
                        "prism-pruner removed "
                        f"{len(entries_with_xyz) - len(pruned_with_xyz)} redundant "
                        "solvent-optimized conformer(s)"
                    ),
                )

    kept = sorted(
        entries_without_xyz + pruned_with_xyz,
        key=lambda entry: float(entry.terms.solution_phase_free_energy_kcal_mol),
    )
    if max_conformers is not None:
        kept = kept[: int(max_conformers)]
    return kept


def _embed_kdg_conformer(mol: Chem.Mol, *, random_seed: int = 42) -> Chem.Mol:
    mol_h = Chem.AddHs(Chem.Mol(mol))
    params = _kdg_embed_parameters(random_seed=random_seed)
    conf_id = AllChem.EmbedMolecule(mol_h, params)
    if conf_id < 0:
        raise RuntimeError("RDKit KDG conformer embedding failed.")
    mol_out = Chem.RemoveHs(mol_h)
    return mol_out


def _resolve_kdg_embedded_conformer_count(
    mol_h: Chem.Mol,
    *,
    embedded_conformers: Optional[int] = None,
    min_conformers: int = REFINEMENT_MIN_EMBED_CONFORMERS,
    max_embed_conformers: int = REFINEMENT_MAX_EMBED_CONFORMERS,
) -> tuple[int, str]:
    if embedded_conformers is not None:
        n_confs = int(embedded_conformers)
        return n_confs, "custom"

    n_rotatable_bonds = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol_h)
    n_confs = max(int(min_conformers), min(3 ** n_rotatable_bonds, int(max_embed_conformers)))
    return n_confs, f"rotatable_bonds(n_rot={n_rotatable_bonds})"


def _generate_kdg_conformer_ensemble(
    mol: Chem.Mol,
    *,
    random_seed: int = 42,
    min_conformers: int = REFINEMENT_MIN_EMBED_CONFORMERS,
    max_embed_conformers: int = REFINEMENT_MAX_EMBED_CONFORMERS,
    embedded_conformers: Optional[int] = None,
    log_paths: Optional[list[Path]] = None,
) -> tuple[Chem.Mol, list[int]]:
    """
    Embed a KDG conformer ensemble with _kdg_embed_parameters and rank conformers
    by MMFF94 single-point energy on the embedded geometries.

    When ``embedded_conformers`` is set, that many conformers are embedded.
    Otherwise the count follows the rotatable-bond heuristic
    (``max(min_conformers, min(3**n_rotatable_bonds, max_embed_conformers))``).

    MMFF94 relaxation is applied later to the pruned candidate pool, before
    applying the final max_qm_conformers limit.
    """
    mol_h = Chem.AddHs(Chem.Mol(mol))
    params = _kdg_embed_parameters(random_seed=random_seed)
    n_confs, size_source = _resolve_kdg_embedded_conformer_count(
        mol_h,
        embedded_conformers=embedded_conformers,
        min_conformers=min_conformers,
        max_embed_conformers=max_embed_conformers,
    )
    conf_ids = list(AllChem.EmbedMultipleConfs(mol_h, int(n_confs), params))
    if not conf_ids:
        raise RuntimeError("RDKit KDG conformer embedding produced no conformers.")

    ranked: list[tuple[float, int]] = []
    for conf_id in conf_ids:
        energy = _mmff94_conformer_energy_kcal_mol(mol_h, conf_id)
        ranked.append((energy if energy is not None else float("inf"), int(conf_id)))
    ranked.sort(key=lambda row: row[0])
    if log_paths is not None:
        _log_status(
            log_paths,
            "OK",
            f"embedded KDG ensemble n_conformers={len(conf_ids)} size_source={size_source}",
        )
    return mol_h, [conf_id for _energy, conf_id in ranked]


def _filter_ranked_embedded_conformers(
    mol_h: Chem.Mol,
    ranked_conf_ids: list[int],
    reference_mol: Chem.Mol,
    *,
    energy_threshold_kcal_mol: Optional[float] = DEFAULT_CONFORMER_ENERGY_THRESHOLD_KCAL_MOL,
) -> list[int]:
    conf_energies: dict[int, Optional[float]] = {
        int(conf_id): _mmff94_conformer_energy_kcal_mol(mol_h, int(conf_id))
        for conf_id in ranked_conf_ids
    }
    finite_energies = [e for e in conf_energies.values() if e is not None]
    min_energy = min(finite_energies) if finite_energies else None

    filtered: list[int] = []
    for conf_id in ranked_conf_ids:
        conf_id = int(conf_id)
        energy = conf_energies.get(conf_id)
        if (
            energy_threshold_kcal_mol is not None
            and min_energy is not None
            and (energy is None or energy - min_energy > float(energy_threshold_kcal_mol))
        ):
            continue
        conf_mol = _mol_from_conf_id(mol_h, conf_id)
        if not _connectivity_matches_reference(conf_mol, reference_mol):
            continue
        filtered.append(conf_id)
    return filtered


def _mol_from_conf_id(mol_h: Chem.Mol, conf_id: int, *, remove_hydrogens: bool = True) -> Chem.Mol:
    mol_one = Chem.Mol(mol_h)
    mol_one.RemoveAllConformers()
    mol_one.AddConformer(mol_h.GetConformer(int(conf_id)), assignId=True)
    if remove_hydrogens:
        return Chem.RemoveHs(mol_one)
    return mol_one


_RELAXED_OPT_WORKFLOW_STATUS_PREFIX = "optimization_retried_with_convergence:"
_GEOMETRY_REOPT_WORKFLOW_STATUS_PREFIX = "geometry_reoptimized_with_convergence:"


def _reset_optimization_artifacts(scratch_dir: Path) -> None:
    for name in (
        "xtbopt.xyz",
        "xtbopt.log",
        "xtbopt_run.log",
        "gxtbopt_run.log",
        "orcaopt.xyz",
        "orcaopt_run.log",
        "aimnet2opt.xyz",
    ):
        path = scratch_dir / name
        if path.exists():
            path.unlink()


def _mark_relaxed_optimization(
    protomer: Protomer,
    *,
    retried_opt_level: str,
    initial_opt_level: str,
    engine: str,
    geometry_retry: bool = False,
) -> None:
    _set_optimization_convergence_props(
        protomer,
        opt_level=retried_opt_level,
        initial_opt_level=initial_opt_level,
        engine=engine,
        relaxed_retry=True,
        geometry_retry=geometry_retry,
    )


def _set_optimization_convergence_props(
    protomer: Protomer,
    *,
    opt_level: str,
    initial_opt_level: str,
    engine: str,
    relaxed_retry: bool = False,
    geometry_retry: bool = False,
) -> None:
    if protomer.mol is None:
        return
    _set_mol_prop_str(protomer.mol, "optimization_opt_level", opt_level)
    _set_mol_prop_str(protomer.mol, "optimization_initial_opt_level", initial_opt_level)
    _set_mol_prop_str(protomer.mol, "optimization_engine", engine)
    if geometry_retry:
        _set_mol_prop_bool(protomer.mol, "geometry_reoptimization_retry", True)
        _set_mol_prop_str(protomer.mol, "optimization_retry_reason", "connectivity_mismatch")
        _set_mol_prop_str(
            protomer.mol,
            "workflow_status",
            f"{_GEOMETRY_REOPT_WORKFLOW_STATUS_PREFIX}{opt_level}",
        )
    if relaxed_retry and not geometry_retry:
        _set_mol_prop_str(
            protomer.mol,
            "workflow_status",
            f"{_RELAXED_OPT_WORKFLOW_STATUS_PREFIX}{opt_level}",
        )


def _scratch_has_scf_retry_marker(scratch_dir: Path) -> bool:
    return any(scratch_dir.rglob(XTB_SCF_RETRY_MARKER_FILE))


def _mark_scf_retry_from_scratch(protomer: Protomer, scratch_dir: Path) -> None:
    if protomer.mol is None or not _scratch_has_scf_retry_marker(scratch_dir):
        return
    _set_mol_prop_bool(protomer.mol, "scf_convergence_retry", True)


def _optional_bool_prop(mol: Chem.Mol, key: str) -> bool:
    return mol.HasProp(key) and mol.GetProp(key).strip().lower() in {"true", "1", "yes"}


def _copy_warning_flags_from_conformer(protomer: Protomer, conformer_protomer: Protomer) -> None:
    if protomer.mol is None or conformer_protomer.mol is None:
        return
    for key in (
        "scf_convergence_retry",
        "geometry_reoptimization_retry",
        "geometry_fallback",
        "connectivity_mismatch",
    ):
        if _optional_bool_prop(conformer_protomer.mol, key):
            _set_mol_prop_bool(protomer.mol, key, True)


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
    user_warning: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> None:
    _log_status(log_paths, status, message)
    if user_warning:
        record_user_warning(message)
    emit_progress(
        progress_callback,
        message,
        level=LogLevel.DEFAULT if status in {"WARN", "FAIL"} else LogLevel.VERBOSE,
    )


def _run_optimization_with_convergence_retry(
    *,
    protomer: Protomer,
    scratch_dir: Path,
    opt_level: str,
    engine: str,
    log_paths: list[Path],
    run_at_level: Callable[[str], tuple[Path, Optional[float], Optional[float]]],
    progress_callback: Optional[Callable[[str], None]] = None,
    level_key: Optional[Callable[[str], str]] = None,
) -> tuple[Path, Optional[float], Optional[float]]:
    """
    Run ``run_at_level`` from ``opt_level`` down to crude until it succeeds.

    ``level_key`` maps convergence presets onto backend settings; presets that map
    to an already-attempted setting are skipped.
    """
    levels = opt_convergence_retry_levels(opt_level)
    if level_key is not None:
        seen_keys: set[str] = set()
        distinct_levels: list[str] = []
        for level in levels:
            key = level_key(level)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            distinct_levels.append(level)
        levels = distinct_levels
    geometry_retry_attempted = False

    for attempt_idx, level in enumerate(levels):
        if attempt_idx > 0:
            _reset_optimization_artifacts(scratch_dir)
            _clear_connectivity_mismatch_flags(protomer)

        try:
            result = run_at_level(level)
        except XtbFatalError as exc:
            report_xtb_fatal_and_exit(exc)
        except (RuntimeError, FileNotFoundError) as exc:
            if attempt_idx < len(levels) - 1:
                next_level = levels[attempt_idx + 1]
                summary = (
                    f"{engine} optimization failed at convergence={level}; "
                    f"retrying from initial geometry with convergence={next_level}"
                )
                _log_status(log_paths, "WARN", f"{summary}: {exc}")
                emit_progress(progress_callback, summary, level=LogLevel.DEFAULT)
                continue
            raise

        opt_xyz_path, *_ = result
        _mark_scf_retry_from_scratch(protomer, scratch_dir)
        if _has_connectivity_mismatch(protomer, opt_xyz_path):
            if attempt_idx < len(levels) - 1:
                geometry_retry_attempted = True
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
                    geometry_retry=True,
                )
                _report_optimization_event(
                    log_paths,
                    status="WARN",
                    message=(
                        f"{engine} optimization still has connectivity mismatch after "
                        f"relaxed retry at convergence={level} (initial={opt_level}) "
                    ),
                    user_warning=True,
                    progress_callback=progress_callback,
                )
            return result

        if attempt_idx > 0:
            _mark_relaxed_optimization(
                protomer,
                retried_opt_level=level,
                initial_opt_level=opt_level,
                engine=engine,
                geometry_retry=geometry_retry_attempted,
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


def _orca_solvent_name(calculators: CalculatorSettings, solvent: SolventNames) -> str:
    return calculators.orca.solvent or solvent.alpb


def _run_solvent_optimization_with_retry(
    *,
    calculators: CalculatorSettings,
    protomer: Protomer,
    mol: Chem.Mol,
    scratch_dir: Path,
    input_xyz_path: Path,
    opt_level: str,
    charge: int,
    multiplicity: int,
    solvent: SolventNames,
    dry_run: bool,
    log_paths: list[Path],
    progress_callback: Optional[Callable[[str], None]] = None,
) -> tuple[Path, Optional[float], Optional[float]]:
    """Optimize in implicit solvent with the configured ``solvent_opt`` calculator."""
    backend = calculators.solvent_opt
    level_key: Optional[Callable[[str], str]] = None

    if backend == "gfn2":
        def _run_at_level(level: str) -> tuple[Path, Optional[float], Optional[float]]:
            return run_xtb_optimization(
                mol=mol,
                scratch_dir=scratch_dir,
                input_xyz_path=input_xyz_path,
                xtb_executable=calculators.xtb_executable,
                opt_level=level,
                charge=charge,
                solvent=solvent.alpb,
                dry_run=dry_run,
                log_paths=log_paths,
                run_command=_run_xtb,
                log_status=_log_status,
            )
    elif backend == "aimnet2":
        level_key = lambda _level: "aimnet2"

        def _run_at_level(level: str) -> tuple[Path, Optional[float], Optional[float]]:
            return run_aimnet2_optimization(
                scratch_dir=scratch_dir,
                input_xyz_path=input_xyz_path,
                charge=charge,
                dry_run=dry_run,
                log_paths=log_paths,
                log_status=_log_status,
            )
    elif backend == "orca":
        level_key = orca_opt_keyword
        keywords = calculators.require_orca_keywords("solvent_opt")
        solvent_keyword = orca_solvent_keyword(
            calculators.solvent_opt_orca_model,
            _orca_solvent_name(calculators, solvent),
        )

        def _run_at_level(level: str) -> tuple[Path, Optional[float], Optional[float]]:
            return run_orca_optimization(
                scratch_dir=scratch_dir,
                xyz_path=input_xyz_path,
                keywords=keywords,
                opt_level=level,
                charge=charge,
                multiplicity=multiplicity,
                settings=calculators.orca,
                dry_run=dry_run,
                log_paths=log_paths,
                run_command=_run,
                log_status=_log_status,
                input_mol=mol,
                solvent_keyword=solvent_keyword,
            )
    else:
        raise ValueError(f"Unknown solvent-optimization calculator: {backend}")

    return _run_optimization_with_convergence_retry(
        protomer=protomer,
        scratch_dir=scratch_dir,
        opt_level=opt_level,
        engine=calculators.label("solvent_opt"),
        log_paths=log_paths,
        progress_callback=progress_callback,
        run_at_level=_run_at_level,
        level_key=level_key,
    )


def _run_gas_optimization_with_retry(
    *,
    calculators: CalculatorSettings,
    protomer: Protomer,
    scratch_dir: Path,
    input_xyz_path: Path,
    input_mol: Chem.Mol,
    opt_level: str,
    charge: int,
    multiplicity: int,
    dry_run: bool,
    log_paths: list[Path],
    progress_callback: Optional[Callable[[str], None]] = None,
) -> tuple[Path, Optional[float], Optional[float]]:
    """Optimize in the gas phase with the configured ``gas`` calculator."""
    backend = calculators.gas
    level_key: Optional[Callable[[str], str]] = None

    if backend == "gxtb":
        run_gxtb_opt = _resolve_gxtb_optimization_runner(calculators.xtb_version)

        def _run_at_level(level: str) -> tuple[Path, Optional[float], Optional[float]]:
            return run_gxtb_opt(
                scratch_dir=scratch_dir,
                xyz_path=input_xyz_path,
                input_mol=input_mol,
                xtb_executable=calculators.xtb_executable,
                opt_level=level,
                charge=charge,
                dry_run=dry_run,
                log_paths=log_paths,
                run_command=_run_xtb,
                log_status=_log_status,
            )
    elif backend == "gfn2":
        def _run_at_level(level: str) -> tuple[Path, Optional[float], Optional[float]]:
            return run_xtb_optimization(
                mol=input_mol,
                scratch_dir=scratch_dir,
                input_xyz_path=input_xyz_path,
                xtb_executable=calculators.xtb_executable,
                opt_level=level,
                charge=charge,
                solvent=None,
                dry_run=dry_run,
                log_paths=log_paths,
                run_command=_run_xtb,
                log_status=_log_status,
            )
    elif backend == "aimnet2":
        level_key = lambda _level: "aimnet2"

        def _run_at_level(level: str) -> tuple[Path, Optional[float], Optional[float]]:
            return run_aimnet2_optimization(
                scratch_dir=scratch_dir,
                input_xyz_path=input_xyz_path,
                charge=charge,
                dry_run=dry_run,
                log_paths=log_paths,
                log_status=_log_status,
            )
    elif backend == "orca":
        level_key = orca_opt_keyword
        keywords = calculators.require_orca_keywords("gas")

        def _run_at_level(level: str) -> tuple[Path, Optional[float], Optional[float]]:
            return run_orca_optimization(
                scratch_dir=scratch_dir,
                xyz_path=input_xyz_path,
                keywords=keywords,
                opt_level=level,
                charge=charge,
                multiplicity=multiplicity,
                settings=calculators.orca,
                dry_run=dry_run,
                log_paths=log_paths,
                run_command=_run,
                log_status=_log_status,
                input_mol=input_mol,
            )
    else:
        raise ValueError(f"Unknown gas-phase calculator: {backend}")

    return _run_optimization_with_convergence_retry(
        protomer=protomer,
        scratch_dir=scratch_dir,
        opt_level=opt_level,
        engine=calculators.label("gas"),
        log_paths=log_paths,
        progress_callback=progress_callback,
        run_at_level=_run_at_level,
        level_key=level_key,
    )


def _prepare_scratch_xyz(scratch_dir: Path, source_xyz: Path, dest_name: str = "input.xyz") -> Path:
    scratch_dir.mkdir(parents=True, exist_ok=True)
    dest = scratch_dir / dest_name
    shutil.copy2(source_xyz, dest)
    return dest


def _run_gas_single_point_on_xyz(
    *,
    calculators: CalculatorSettings,
    scratch_dir: Path,
    xyz_path: Path,
    charge: int,
    multiplicity: int,
    dry_run: bool,
    log_paths: list[Path],
) -> Optional[float]:
    """Gas-phase single point (kcal/mol) with the configured ``gas`` calculator."""
    backend = calculators.gas
    if backend == "gxtb":
        run_gxtb_sp = _resolve_gxtb_single_point_runner(calculators.xtb_version)
        energy_kcal_mol, _ = run_gxtb_sp(
            scratch_dir=scratch_dir,
            xyz_path=xyz_path,
            xtb_executable=calculators.xtb_executable,
            charge=charge,
            dry_run=dry_run,
            log_paths=log_paths,
            run_command=_run_xtb,
            log_status=_log_status,
        )
    elif backend == "gfn2":
        energy_kcal_mol, _ = run_xtb_single_point_energy(
            scratch_dir=scratch_dir,
            xyz_path=xyz_path,
            xtb_executable=calculators.xtb_executable,
            charge=charge,
            gfn=2,
            dry_run=dry_run,
            log_paths=log_paths,
            run_command=_run_xtb,
            log_status=_log_status,
        )
    elif backend == "aimnet2":
        energy_kcal_mol, _ = run_aimnet2_single_point_energy(
            scratch_dir=scratch_dir,
            xyz_path=xyz_path,
            charge=charge,
            dry_run=dry_run,
            log_paths=log_paths,
            log_status=_log_status,
        )
    elif backend == "orca":
        energy_kcal_mol, _ = run_orca_single_point_energy(
            scratch_dir=scratch_dir,
            xyz_path=xyz_path,
            keywords=calculators.require_orca_keywords("gas"),
            charge=charge,
            multiplicity=multiplicity,
            settings=calculators.orca,
            dry_run=dry_run,
            log_paths=log_paths,
            run_command=_run,
            log_status=_log_status,
        )
    else:
        raise ValueError(f"Unknown gas-phase calculator: {backend}")
    return energy_kcal_mol


def _run_solvation_single_point_on_xyz(
    *,
    calculators: CalculatorSettings,
    scratch_dir: Path,
    xyz_path: Path,
    solvent: SolventNames,
    charge: int,
    multiplicity: int,
    dry_run: bool,
    log_paths: list[Path],
) -> Optional[float]:
    """Implicit-solvation free energy dG_solv (kcal/mol) with the configured ``solvation`` calculator."""
    backend = calculators.solvation
    if backend == "cpcmx":
        return run_cpcmx_single_point(
            scratch_dir=scratch_dir,
            xyz_path=xyz_path,
            xtb_executable=calculators.xtb_executable,
            solvent=solvent.cpcm,
            charge=charge,
            gfn=2,
            dry_run=dry_run,
            log_paths=log_paths,
            run_command=_run_xtb,
            log_status=_log_status,
        )
    if backend == "orca":
        return run_orca_solvation_single_point(
            scratch_dir=scratch_dir,
            xyz_path=xyz_path,
            keywords=calculators.solvation_orca_job_keywords(),
            model=calculators.solvation_orca_model,
            solvent=_orca_solvent_name(calculators, solvent),
            charge=charge,
            multiplicity=multiplicity,
            settings=calculators.orca,
            dry_run=dry_run,
            log_paths=log_paths,
            run_command=_run,
            log_status=_log_status,
        )
    raise ValueError(f"Unknown solvation calculator: {backend}")


def _run_frequencies_on_xyz(
    *,
    calculators: CalculatorSettings,
    scratch_dir: Path,
    xyz_path: Path,
    charge: int,
    multiplicity: int,
    dry_run: bool,
    log_paths: list[Path],
    temperature_k: float,
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Frequencies with the configured ``freq`` calculator.

    Returns (electronic energy kcal/mol, RRHO contribution kcal/mol, electronic energy Hartree).
    """
    backend = calculators.freq
    if backend == "gfn2":
        return run_hessian_and_parse_energies(
            scratch_dir=scratch_dir,
            xyz_path=xyz_path,
            xtb_executable=calculators.xtb_executable,
            charge=charge,
            gfn=2,
            dry_run=dry_run,
            log_paths=log_paths,
            run_command=_run_xtb,
            log_status=_log_status,
            temperature_k=temperature_k,
        )
    if backend == "gxtb":
        run_gxtb_hess = _resolve_gxtb_hessian_runner(calculators.xtb_version)
        return run_gxtb_hess(
            scratch_dir=scratch_dir,
            xyz_path=xyz_path,
            xtb_executable=calculators.xtb_executable,
            charge=charge,
            dry_run=dry_run,
            log_paths=log_paths,
            run_command=_run_xtb,
            log_status=_log_status,
            temperature_k=temperature_k,
        )
    if backend == "orca":
        return run_orca_frequencies(
            scratch_dir=scratch_dir,
            xyz_path=xyz_path,
            keywords=calculators.require_orca_keywords("freq"),
            charge=charge,
            multiplicity=multiplicity,
            settings=calculators.orca,
            dry_run=dry_run,
            log_paths=log_paths,
            run_command=_run,
            log_status=_log_status,
            temperature_k=temperature_k,
        )
    raise ValueError(f"Unknown frequency calculator: {backend}")


def _try_gas_phase_refinement(
    *,
    calculators: CalculatorSettings,
    protomer: Protomer,
    mol: Chem.Mol,
    solvent_opt_xyz_path: Path,
    scratch_dir: Path,
    opt_level: str,
    charge: int,
    multiplicity: int,
    dry_run: bool,
    log_paths: list[Path],
    progress_callback: Optional[Callable[[str], None]] = None,
) -> GasRefinementResult:
    """
    Re-optimize in the gas phase, then compute the gas-phase SP on the optimized geometry.

    If the gas-phase optimization fails or the optimized geometry has a connectivity
    mismatch, falls back to a gas-phase single point on the implicit-solvent geometry.
    """
    def _progress(message: str, *, level: LogLevel = LogLevel.VERBOSE) -> None:
        emit_progress(progress_callback, message, level=level)

    gas_label = calculators.label("gas")
    solvent_opt_label = calculators.label("solvent_opt")
    reference_mol = _input_mol_for_connectivity(protomer, mol)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    refine_input_xyz = _prepare_scratch_xyz(scratch_dir, solvent_opt_xyz_path, "gas_refine_input.xyz")

    def _fallback_sp_on_solvent_geometry(reason: str) -> GasRefinementResult:
        _log_status(log_paths, "WARN", reason)
        record_user_warning(reason, context="gas-phase refinement fallback")
        if protomer.mol is not None:
            _set_mol_prop_bool(protomer.mol, "geometry_fallback", True)
        _progress(f"computing {gas_label} gas-phase single point on {solvent_opt_label} geometry (fallback)")
        fallback_scratch = scratch_dir / "gas_sp_fallback"
        fallback_xyz = _prepare_scratch_xyz(fallback_scratch, solvent_opt_xyz_path, "input.xyz")
        try:
            fallback_gas_sp = _run_gas_single_point_on_xyz(
                calculators=calculators,
                scratch_dir=fallback_scratch,
                xyz_path=fallback_xyz,
                charge=charge,
                multiplicity=multiplicity,
                dry_run=dry_run,
                log_paths=log_paths,
            )
        except XtbFatalError as exc:
            report_xtb_fatal_and_exit(exc)
        except (RuntimeError, FileNotFoundError) as exc:
            _log_status(
                log_paths,
                "WARN",
                f"{gas_label} fallback SP on {solvent_opt_label} geometry failed: {exc}",
            )
            return GasRefinementResult(gas_sp_energy_kcal_mol=None)
        return GasRefinementResult(gas_sp_energy_kcal_mol=fallback_gas_sp)

    _progress(f"re-optimizing geometry with {gas_label} (gas phase)")
    try:
        gas_opt_xyz, _, _ = _run_gas_optimization_with_retry(
            calculators=calculators,
            protomer=protomer,
            scratch_dir=scratch_dir,
            input_xyz_path=refine_input_xyz,
            input_mol=mol,
            opt_level=opt_level,
            charge=charge,
            multiplicity=multiplicity,
            dry_run=dry_run,
            log_paths=log_paths,
            progress_callback=progress_callback,
        )
    except XtbFatalError:
        raise
    except (RuntimeError, FileNotFoundError) as exc:
        return _fallback_sp_on_solvent_geometry(
            f"{gas_label} gas-phase re-optimization failed; falling back to {gas_label} SP "
            f"on {solvent_opt_label} geometry: {exc}",
        )

    if not _xyz_connectivity_matches_reference(gas_opt_xyz, reference_mol):
        return _fallback_sp_on_solvent_geometry(
            f"{gas_label} gas-phase re-optimization connectivity mismatch against input mol; "
            f"discarding gas-phase geometry and falling back to {gas_label} SP on {solvent_opt_label} geometry",
        )

    gas_opt_xyz_path = scratch_dir / "gas_opt.xyz"
    shutil.copy2(gas_opt_xyz, gas_opt_xyz_path)

    _progress(f"computing {gas_label} gas-phase single point on gas-phase-optimized geometry")
    try:
        sp_scratch = scratch_dir / "gas_sp"
        sp_xyz = _prepare_scratch_xyz(sp_scratch, gas_opt_xyz_path, "input.xyz")
        _log_status(
            log_paths,
            "GEOM",
            f"refinement gas-phase SP geometry={gas_opt_xyz_path.name} scratch={sp_xyz}",
        )
        refined_gas_sp = _run_gas_single_point_on_xyz(
            calculators=calculators,
            scratch_dir=sp_scratch,
            xyz_path=sp_xyz,
            charge=charge,
            multiplicity=multiplicity,
            dry_run=dry_run,
            log_paths=log_paths,
        )
    except XtbFatalError:
        raise
    except (RuntimeError, FileNotFoundError) as exc:
        return _fallback_sp_on_solvent_geometry(
            f"{gas_label} gas-phase SP on optimized geometry failed; falling back to {gas_label} SP "
            f"on {solvent_opt_label} geometry: {exc}",
        )

    if refined_gas_sp is None:
        return _fallback_sp_on_solvent_geometry(
            f"{gas_label} gas-phase SP on optimized geometry returned no energy; "
            f"falling back to {gas_label} SP on {solvent_opt_label} geometry",
        )

    _log_status(log_paths, "OK", f"{gas_label} gas-phase re-optimization and SP succeeded")
    return GasRefinementResult(
        gas_sp_energy_kcal_mol=refined_gas_sp,
        gas_opt_xyz_path=gas_opt_xyz_path,
    )


def _try_frequencies_with_fallback(
    *,
    calculators: CalculatorSettings,
    primary_xyz_path: Optional[Path],
    fallback_xyz_path: Path,
    scratch_dir: Path,
    charge: int,
    multiplicity: int,
    dry_run: bool,
    log_paths: list[Path],
    temperature_k: float = DEFAULT_TEMPERATURE_K,
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Run frequencies on the gas-phase geometry, falling back to ``fallback_xyz_path``."""
    freq_label = calculators.label("freq")
    if primary_xyz_path is not None:
        primary_scratch = scratch_dir / "freq_gas_geom"
        primary_xyz = _prepare_scratch_xyz(primary_scratch, primary_xyz_path, "input.xyz")
        try:
            freq_energy_kcal_mol, rrho_contribution_kcal_mol, freq_energy_h = _run_frequencies_on_xyz(
                calculators=calculators,
                scratch_dir=primary_scratch,
                xyz_path=primary_xyz,
                charge=charge,
                multiplicity=multiplicity,
                dry_run=dry_run,
                log_paths=log_paths,
                temperature_k=temperature_k,
            )
            if freq_energy_kcal_mol is not None and rrho_contribution_kcal_mol is not None:
                _log_status(log_paths, "OK", f"{freq_label} RRHO computed at gas-phase-optimized geometry")
                return freq_energy_kcal_mol, rrho_contribution_kcal_mol, freq_energy_h
            _log_status(
                log_paths,
                "WARN",
                f"{freq_label} RRHO at gas-phase-optimized geometry returned incomplete terms; "
                "falling back to implicit-solvent geometry",
            )
        except XtbFatalError as exc:
            report_xtb_fatal_and_exit(exc)
        except (RuntimeError, FileNotFoundError) as exc:
            _log_status(
                log_paths,
                "WARN",
                f"{freq_label} RRHO at gas-phase-optimized geometry failed; "
                f"falling back to implicit-solvent geometry: {exc}",
            )

    fallback_scratch = scratch_dir / "freq_solvent_geom"
    fallback_xyz = _prepare_scratch_xyz(fallback_scratch, fallback_xyz_path, "input.xyz")
    return _run_frequencies_on_xyz(
        calculators=calculators,
        scratch_dir=fallback_scratch,
        xyz_path=fallback_xyz,
        charge=charge,
        multiplicity=multiplicity,
        dry_run=dry_run,
        log_paths=log_paths,
        temperature_k=temperature_k,
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
        check=False,
    )


def _run_xtb(
    cmd: str | list[str],
    *,
    cwd: Path,
    dry_run: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run an xTB command with log inspection and SCF etemp retry."""
    return run_xtb_command(
        cmd,
        cwd=cwd,
        dry_run=dry_run,
        run_fn=_run,
    )


def _set_mol_prop_str(mol: Chem.Mol, key: str, value: Optional[str]) -> None:
    if value is None:
        return
    mol.SetProp(key, value)


def _set_mol_prop_bool(mol: Chem.Mol, key: str, value: bool) -> None:
    mol.SetProp(key, "true" if value else "false")


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

    Writes geometries onto each protomer.mol before any QM workflows run.
    """
    def _progress(message: str, *, level: LogLevel = LogLevel.VERBOSE) -> None:
        emit_progress(progress_callback, message, level=level)

    scratch_root_path = Path(scratch_root)
    scratch_root_path.mkdir(parents=True, exist_ok=True)
    log_paths = [_species_workflow_log_path(scratch_root_path)]
    _log_status(log_paths, "START", f"batch conformer generation mode={conformer_mode} n_protomers={len(protomer_refs)}")

    if dry_run:
        _log_status(log_paths, "SKIP", "dry_run enabled; skipping batch conformer generation")
        return

    for taut_idx, prot_idx, protomer in protomer_refs:
        prefix = f"taut {taut_idx + 1} prot {prot_idx + 1}"
        _progress(f"preparing conformer for {prefix}")
        try:
            mol, conformer_energy_kcal_mol = _prepare_protomer_conformer(
                protomer,
                conformer_mode=conformer_mode,
                external_xyz_path=external_xyz_path,
                log_paths=log_paths,
                random_seed=random_seed,
            )
            seed_xyz_path = _save_runtime_xyz(
                scratch_root=scratch_root_path,
                filename=f"tautomer_{taut_idx}_protomer_{prot_idx}_seed_geom.xyz",
                mol=protomer.mol,
                log_paths=log_paths,
            )
            _set_mol_prop_str(protomer.mol, "seed_geom_xyz_path", str(seed_xyz_path))
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
    if any(atom.GetAtomicNum() == 1 for atom in mol.GetAtoms()):
        mol_out = mol
    else:
        mol_out = Chem.AddHs(Chem.Mol(mol), addCoords=True)
    return Chem.MolToXYZBlock(mol_out, confId=conf_id)


def _write_xyz(mol: Chem.Mol, path: Path, *, conf_id: int = 0) -> None:
    xyz = _mol_to_xyz_block(mol, conf_id=conf_id)
    path.write_text(xyz)


def _runtime_xyz_dir(scratch_root: Path) -> Path:
    dest = scratch_root / "xyz"
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def _save_runtime_xyz(
    *,
    scratch_root: Path,
    filename: str,
    source_xyz: Optional[Path] = None,
    mol: Optional[Chem.Mol] = None,
    conf_id: int = 0,
    log_paths: Optional[list[Path]] = None,
) -> Path:
    dest = _runtime_xyz_dir(scratch_root) / filename
    if source_xyz is not None:
        shutil.copy2(source_xyz, dest)
    elif mol is not None:
        _write_xyz(mol, dest, conf_id=conf_id)
    else:
        raise ValueError("Either source_xyz or mol must be provided.")
    if log_paths is not None:
        _log_status(log_paths, "KEEP", f"saved runtime xyz {dest.name}")
    return dest


def _remove_unkept_conformer_artifacts(
    *,
    scratch_root: Path,
    protomer_id: int | str,
    kept_conformer_indices: frozenset[int],
    log_paths: Optional[list[Path]] = None,
) -> None:
    xyz_dir = _runtime_xyz_dir(scratch_root)
    prefix = f"protomer_{protomer_id}_conformer_"
    for path in sorted(xyz_dir.glob(f"{prefix}*")):
        index_part = path.name[len(prefix) :].split("_", 1)[0]
        try:
            conf_index = int(index_part)
        except ValueError:
            continue
        if conf_index in kept_conformer_indices:
            continue
        path.unlink(missing_ok=True)
        if log_paths is not None:
            _log_status(log_paths, "CLEANUP", f"removed pruned conformer artifact {path.name}")

    log_dir = scratch_root / "log"
    if log_dir.is_dir():
        for path in sorted(log_dir.glob(f"protomer_{protomer_id}_conformer_*")):
            index_part = path.name.split("_conformer_", 1)[-1]
            try:
                conf_index = int(index_part)
            except ValueError:
                continue
            if conf_index in kept_conformer_indices:
                continue
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
            if log_paths is not None:
                _log_status(log_paths, "CLEANUP", f"removed pruned conformer artifact {path.name}")


@dataclass(frozen=True)
class SolvationWorkflowResult:
    conformer_energy_kcal_mol: Optional[float]
    optimized_xyz: Optional[Path]
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
                _set_mol_prop_bool(protomer.mol, "geometry_fallback", True)
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
            for key in ("conformer_energy_kcal_mol", "conformer_delta_kcal_mol", "freq_electronic_energy_kcal_mol"):
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
    freq_electronic_energy_kcal_mol: Optional[float],
    rrho_contribution_kcal_mol: Optional[float],
    solution_phase_free_energy_kcal_mol: Optional[float],
) -> None:
    protomer.mol.SetProp("charge", str(charge))
    _set_mol_prop_double(protomer.mol, "conformer_energy_kcal_mol", conformer_energy_kcal_mol)
    _set_mol_prop_double(protomer.mol, "solvation_free_energy_kcal_mol", solvation_free_energy_kcal_mol)
    _set_mol_prop_double(protomer.mol, "gas_sp_energy_kcal_mol", gas_sp_energy_kcal_mol)
    _set_mol_prop_double(protomer.mol, "freq_electronic_energy_kcal_mol", freq_electronic_energy_kcal_mol)
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


def promote_screening_terms_to_final(
    protomer: Protomer,
    *,
    clear_gas_sp: bool = False,
) -> None:
    """
    Copy screening solvation, RRHO, and frequency electronic-energy terms onto final mol props.

    The screening gas-phase SP (``screening_gas_sp_energy_kcal_mol``) is intentionally not
    promoted: final ``gas_sp_energy_kcal_mol`` is reserved for the post-screen refinement.
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
    if mol.HasProp("screening_freq_electronic_energy_kcal_mol"):
        _set_mol_prop_double(
            mol,
            "freq_electronic_energy_kcal_mol",
            float(mol.GetDoubleProp("screening_freq_electronic_energy_kcal_mol")),
        )
    if clear_gas_sp and mol.HasProp("gas_sp_energy_kcal_mol"):
        mol.ClearProp("gas_sp_energy_kcal_mol")


def _preserve_output_files(
    scratch_dir: Path,
    *,
    keep_logs: bool = False,
    kept_conformer_indices: Optional[frozenset[int]] = None,
) -> Optional[Path]:
    if not scratch_dir.exists():
        return None

    log_paths = [_species_workflow_log_path(scratch_dir.parent)]
    preserved_dir = scratch_dir.parent / "xyz"
    preserved_dir.mkdir(parents=True, exist_ok=True)

    preserved_opt_path: Optional[Path] = None
    files_to_preserve = [
        "input.xyz",
        "screening_geom.xyz",
        "kdg_geom.xyz",
        "mmff94_opt.xyz",
        "solvent_opt.xyz",
        "gas_opt.xyz",
        "xtbopt.xyz",
        "aimnet2opt.xyz",
        "orcaopt.xyz",
        "xtbopt.log",
    ]
    for file_name in files_to_preserve:
        src = scratch_dir / file_name
        if not src.exists():
            continue
        dst = preserved_dir / f"{scratch_dir.name}_{file_name}"
        shutil.copy2(src, dst)
        _log_status(log_paths, "KEEP", f"preserved {file_name} at {dst}")
        if file_name in (
            "xtbopt.xyz",
            "aimnet2opt.xyz",
            "orcaopt.xyz",
            "screening_geom.xyz",
            "kdg_geom.xyz",
            "solvent_opt.xyz",
            "gas_opt.xyz",
            "mmff94_opt.xyz",
        ):
            preserved_opt_path = dst

    conformer_xyz_names = ("mmff94_opt.xyz", "solvent_opt.xyz", "gas_opt.xyz")
    for conf_dir in sorted(scratch_dir.glob("conformer_*")):
        if not conf_dir.is_dir():
            continue
        conf_suffix = conf_dir.name.removeprefix("conformer_")
        if kept_conformer_indices is not None:
            try:
                conf_index = int(conf_suffix)
            except ValueError:
                conf_index = None
            if conf_index is not None and conf_index not in kept_conformer_indices:
                continue
        for file_name in conformer_xyz_names:
            src = conf_dir / file_name
            if not src.exists():
                continue
            dst = preserved_dir / f"{scratch_dir.name}_{conf_dir.name}_{file_name}"
            shutil.copy2(src, dst)
            _log_status(log_paths, "KEEP", f"preserved {conf_dir.name}/{file_name} at {dst}")
            if preserved_opt_path is None and file_name == "solvent_opt.xyz":
                preserved_opt_path = dst
    if keep_logs:
        preserved_log_dir = scratch_dir.parent / "log"
        preserved_log_dir.mkdir(parents=True, exist_ok=True)
        preserved_log_names: set[Path] = set()
        for log_path in scratch_dir.rglob("*"):
            if not log_path.is_file():
                continue
            if not (log_path.name.endswith("_run.log") or log_path.name == "xtbopt.log"):
                continue
            rel = log_path.relative_to(scratch_dir)
            if kept_conformer_indices is not None and rel.parts and rel.parts[0].startswith("conformer_"):
                try:
                    conf_index = int(rel.parts[0].removeprefix("conformer_"))
                except ValueError:
                    conf_index = None
                if conf_index is not None and conf_index not in kept_conformer_indices:
                    continue
            if rel in preserved_log_names:
                continue
            preserved_log_names.add(rel)
            dst = preserved_log_dir / f"{scratch_dir.name}_{rel}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(log_path, dst)
            _log_status(log_paths, "KEEP", f"preserved {rel} at {dst}")

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
    temperature_k: float = DEFAULT_TEMPERATURE_K,
) -> None:
    if protomer.mol is None:
        return

    gas_sp_values = [term.gas_sp_energy_kcal_mol for term in conformer_terms]
    freq_energy_values = [term.freq_electronic_energy_kcal_mol for term in conformer_terms]
    solvation_values = [term.solvation_free_energy_kcal_mol for term in conformer_terms]
    rrho_values = [term.rrho_contribution_kcal_mol for term in conformer_terms]
    solution_values = [term.solution_phase_free_energy_kcal_mol for term in conformer_terms]

    _set_energy_list_prop(protomer.mol, "conformer_gas_sp_energy_kcal_mol_list", gas_sp_values)
    _set_energy_list_prop(protomer.mol, "conformer_freq_electronic_energy_kcal_mol_list", freq_energy_values)
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
    aggregate_freq_energy = _boltzmann_aggregate_energy(
        [value for value in freq_energy_values if value is not None],
        temperature_k=temperature_k,
    )

    _persist_protomer_results(
        protomer,
        charge=charge,
        conformer_energy_kcal_mol=None,
        solvation_free_energy_kcal_mol=aggregate_solv,
        gas_sp_energy_kcal_mol=aggregate_gas,
        freq_electronic_energy_kcal_mol=aggregate_freq_energy,
        rrho_contribution_kcal_mol=aggregate_rrho,
        solution_phase_free_energy_kcal_mol=aggregate_solution,
    )


def _apply_energy_terms(
    terms: ConformerEnergyTerms,
    *,
    gas_sp_energy_kcal_mol: Optional[float],
    solvation_free_energy_kcal_mol: Optional[float],
    freq_electronic_energy_kcal_mol: Optional[float],
    rrho_contribution_kcal_mol: Optional[float],
    log_paths: list[Path],
) -> None:
    terms.gas_sp_energy_kcal_mol = _format_energy_entry(
        gas_sp_energy_kcal_mol,
        failed_step="gas-sp",
    )
    terms.freq_electronic_energy_kcal_mol = _format_energy_entry(
        freq_electronic_energy_kcal_mol,
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


def _run_screening_conformer_workflow(
    protomer: Protomer,
    mol: Chem.Mol,
    *,
    scratch_dir: Path,
    scratch_root: Path,
    runtime_xyz_label: str,
    calculators: CalculatorSettings,
    charge: int,
    multiplicity: int,
    solvent: SolventNames,
    dry_run: bool,
    log_paths: list[Path],
    temperature_k: float = DEFAULT_TEMPERATURE_K,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> ConformerWorkflowResult:
    """
    Lightweight screening workflow on a KDG geometry.

    Skips all geometry optimizations; runs the implicit-solvation SP, gas-phase SP,
    and frequency (RRHO) calculations on the input conformer.
    """
    def _progress(message: str, *, level: LogLevel = LogLevel.VERBOSE) -> None:
        emit_progress(progress_callback, message, level=level)

    gas_label = calculators.label("gas")
    freq_label = calculators.label("freq")
    solvation_label = calculators.label("solvation")
    terms = ConformerEnergyTerms(gas_sp_energy_kcal_mol="not-run")
    scratch_dir.mkdir(parents=True, exist_ok=True)

    try:
        input_xyz_path = _write_workflow_inputs(mol, scratch_dir, charge, log_paths)
        screening_geom_path = scratch_dir / "screening_geom.xyz"
        shutil.copy2(input_xyz_path, screening_geom_path)
        _save_runtime_xyz(
            scratch_root=scratch_root,
            filename=f"{runtime_xyz_label}_screening_geom.xyz",
            source_xyz=screening_geom_path,
            log_paths=log_paths,
        )

        _progress(f"computing {solvation_label} solvation on seeded screening geometry")
        solvation_scratch = scratch_dir / "solvation_sp"
        solvation_xyz_path = _prepare_scratch_xyz(solvation_scratch, screening_geom_path, "input.xyz")
        _log_status(
            log_paths,
            "GEOM",
            f"screening solvation SP geometry={screening_geom_path.name} scratch={solvation_xyz_path}",
        )
        solvation_free_energy_kcal_mol = _run_solvation_single_point_on_xyz(
            calculators=calculators,
            scratch_dir=solvation_scratch,
            xyz_path=solvation_xyz_path,
            solvent=solvent,
            charge=charge,
            multiplicity=multiplicity,
            dry_run=dry_run,
            log_paths=log_paths,
        )
        terms.solvation_free_energy_kcal_mol = _format_energy_entry(
            solvation_free_energy_kcal_mol,
            failed_step="solvation",
        )

        gas_sp_energy_kcal_mol: Optional[float] = None
        freq_electronic_energy_kcal_mol: Optional[float] = None
        rrho_contribution_kcal_mol: Optional[float] = None
        try:
            gas_scratch = scratch_dir / "gas_sp"
            gas_xyz_path = _prepare_scratch_xyz(gas_scratch, screening_geom_path, "input.xyz")
            _log_status(
                log_paths,
                "GEOM",
                f"screening gas-phase SP geometry={screening_geom_path.name} scratch={gas_xyz_path}",
            )
            _progress(f"computing {gas_label} gas-phase single point on seeded screening geometry")
            gas_sp_energy_kcal_mol = _run_gas_single_point_on_xyz(
                calculators=calculators,
                scratch_dir=gas_scratch,
                xyz_path=gas_xyz_path,
                charge=charge,
                multiplicity=multiplicity,
                dry_run=dry_run,
                log_paths=log_paths,
            )

            _progress(
                f"computing RRHO contribution with {freq_label} frequencies at seeded screening geometry "
                f"(T={temperature_k:g} K)"
            )
            freq_electronic_energy_kcal_mol, rrho_contribution_kcal_mol, _ = _try_frequencies_with_fallback(
                calculators=calculators,
                primary_xyz_path=None,
                fallback_xyz_path=screening_geom_path,
                scratch_dir=scratch_dir,
                charge=charge,
                multiplicity=multiplicity,
                dry_run=dry_run,
                log_paths=log_paths,
                temperature_k=temperature_k,
            )
        except XtbFatalError as exc:
            report_xtb_fatal_and_exit(exc)
        except Exception as exc:
            _log_status(
                log_paths,
                "WARN",
                f"gas-phase or frequency steps failed after {solvation_label} solvation: {exc}",
            )

        _apply_energy_terms(
            terms,
            gas_sp_energy_kcal_mol=gas_sp_energy_kcal_mol,
            solvation_free_energy_kcal_mol=solvation_free_energy_kcal_mol,
            freq_electronic_energy_kcal_mol=freq_electronic_energy_kcal_mol,
            rrho_contribution_kcal_mol=rrho_contribution_kcal_mol,
            log_paths=log_paths,
        )
        return ConformerWorkflowResult(terms=terms, opt_xyz_path=screening_geom_path)
    except XtbFatalError:
        raise
    except Exception as exc:
        _log_status(log_paths, "FAIL", f"screening conformer workflow failed: {exc}")
        terms.workflow_status = "screening-failed"
        if terms.solvation_free_energy_kcal_mol == "not-run":
            terms.solvation_free_energy_kcal_mol = _format_energy_entry(None, failed_step="screening")
        if terms.gas_sp_energy_kcal_mol == "not-run":
            terms.gas_sp_energy_kcal_mol = _format_energy_entry(None, failed_step="screening")
        if terms.rrho_contribution_kcal_mol == "not-run":
            terms.rrho_contribution_kcal_mol = _format_energy_entry(None, failed_step="screening")
        if terms.solution_phase_free_energy_kcal_mol == "not-run":
            terms.solution_phase_free_energy_kcal_mol = _format_energy_entry(None, failed_step="screening")
        return ConformerWorkflowResult(terms=terms, opt_xyz_path=None)


def _run_single_conformer_workflow(
    protomer: Protomer,
    mol: Chem.Mol,
    *,
    scratch_dir: Path,
    scratch_root: Path,
    runtime_xyz_label: str,
    calculators: CalculatorSettings,
    charge: int,
    multiplicity: int,
    solvent: SolventNames,
    opt_level: str,
    gas_optimize: bool = True,
    dry_run: bool,
    log_paths: list[Path],
    temperature_k: float = DEFAULT_TEMPERATURE_K,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> ConformerWorkflowResult:
    """
    Full QM refinement workflow for one conformer.

    Geometry usage:
    - Implicit-solvation SP: implicit-solvent (``solvent_opt``) optimized geometry.
    - Gas-phase SP and frequencies: when ``gas_optimize`` is enabled (default), a
      gas-phase re-optimization with the ``gas`` calculator is run first and SP /
      frequencies use that geometry when available; otherwise both use the
      implicit-solvent geometry.
    """
    def _progress(message: str, *, level: LogLevel = LogLevel.VERBOSE) -> None:
        emit_progress(progress_callback, message, level=level)

    gas_label = calculators.label("gas")
    freq_label = calculators.label("freq")
    solvent_opt_label = calculators.label("solvent_opt")
    solvation_label = calculators.label("solvation")
    terms = ConformerEnergyTerms(gas_sp_energy_kcal_mol="not-run")
    scratch_dir.mkdir(parents=True, exist_ok=True)

    try:
        input_xyz_path = _write_workflow_inputs(mol, scratch_dir, charge, log_paths)
        mmff94_geom_path = scratch_dir / "mmff94_opt.xyz"
        shutil.copy2(input_xyz_path, mmff94_geom_path)
        _save_runtime_xyz(
            scratch_root=scratch_root,
            filename=f"{runtime_xyz_label}_mmff94_opt.xyz",
            source_xyz=mmff94_geom_path,
            log_paths=log_paths,
        )
        _progress(f"optimizing geometry with {solvent_opt_label}")
        opt_xyz_path, _opt_energy_kcal_mol, _opt_energy_h = _run_solvent_optimization_with_retry(
            calculators=calculators,
            protomer=protomer,
            mol=mol,
            scratch_dir=scratch_dir,
            input_xyz_path=input_xyz_path,
            opt_level=opt_level,
            charge=charge,
            multiplicity=multiplicity,
            solvent=solvent,
            dry_run=dry_run,
            log_paths=log_paths,
            progress_callback=progress_callback,
        )

        reference_mol = _input_mol_for_connectivity(protomer, mol)
        if not _xyz_connectivity_matches_reference(opt_xyz_path, reference_mol):
            warning_message = (
                f"{solvent_opt_label} optimization connectivity mismatch against input mol; "
                "discarding conformer and excluding its energies from the conformer pool"
            )
            _log_status(log_paths, "WARN", warning_message)
            record_user_warning(warning_message, context="conformer refinement")
            if protomer.mol is not None:
                _set_mol_prop_bool(protomer.mol, "connectivity_mismatch", True)
                _set_mol_prop_bool(protomer.mol, "geometry_fallback", True)
            terms.workflow_status = "connectivity-failed"
            terms.gas_sp_energy_kcal_mol = _format_energy_entry(None, failed_step="connectivity")
            terms.solvation_free_energy_kcal_mol = _format_energy_entry(None, failed_step="connectivity")
            terms.rrho_contribution_kcal_mol = _format_energy_entry(None, failed_step="connectivity")
            terms.solution_phase_free_energy_kcal_mol = _format_energy_entry(None, failed_step="connectivity")
            return ConformerWorkflowResult(terms=terms, opt_xyz_path=None)

        _update_protomer_geometry_from_xyz(
            protomer,
            opt_xyz_path,
            log_paths,
        )

        solvent_opt_xyz_path = scratch_dir / "solvent_opt.xyz"
        shutil.copy2(opt_xyz_path, solvent_opt_xyz_path)
        _save_runtime_xyz(
            scratch_root=scratch_root,
            filename=f"{runtime_xyz_label}_solvent_opt.xyz",
            source_xyz=solvent_opt_xyz_path,
            log_paths=log_paths,
        )

        _progress(f"computing {solvation_label} solvation on {solvent_opt_label} geometry")
        solvation_scratch = scratch_dir / "solvation_sp"
        solvation_xyz_path = _prepare_scratch_xyz(solvation_scratch, solvent_opt_xyz_path, "input.xyz")
        _log_status(
            log_paths,
            "GEOM",
            f"refinement solvation SP geometry={solvent_opt_xyz_path.name} scratch={solvation_xyz_path}",
        )
        solvation_free_energy_kcal_mol = _run_solvation_single_point_on_xyz(
            calculators=calculators,
            scratch_dir=solvation_scratch,
            xyz_path=solvation_xyz_path,
            solvent=solvent,
            charge=charge,
            multiplicity=multiplicity,
            dry_run=dry_run,
            log_paths=log_paths,
        )
        terms.solvation_free_energy_kcal_mol = _format_energy_entry(
            solvation_free_energy_kcal_mol,
            failed_step="solvation",
        )

        gas_sp_energy_kcal_mol: Optional[float] = None
        gas_opt_xyz_path: Optional[Path] = None
        freq_electronic_energy_kcal_mol: Optional[float] = None
        rrho_contribution_kcal_mol: Optional[float] = None
        try:
            if gas_optimize:
                gas_refinement = _try_gas_phase_refinement(
                    calculators=calculators,
                    protomer=protomer,
                    mol=mol,
                    solvent_opt_xyz_path=solvent_opt_xyz_path,
                    scratch_dir=scratch_dir / "gas_refine",
                    opt_level=opt_level,
                    charge=charge,
                    multiplicity=multiplicity,
                    dry_run=dry_run,
                    log_paths=log_paths,
                    progress_callback=progress_callback,
                )
                gas_sp_energy_kcal_mol = gas_refinement.gas_sp_energy_kcal_mol
                gas_opt_xyz_path = gas_refinement.gas_opt_xyz_path
                if gas_opt_xyz_path is not None:
                    _save_runtime_xyz(
                        scratch_root=scratch_root,
                        filename=f"{runtime_xyz_label}_gas_opt.xyz",
                        source_xyz=gas_opt_xyz_path,
                        log_paths=log_paths,
                    )
            else:
                _progress(f"computing {gas_label} gas-phase single point on {solvent_opt_label} geometry")
                gas_scratch = scratch_dir / "gas_sp"
                gas_sp_xyz = _prepare_scratch_xyz(gas_scratch, solvent_opt_xyz_path, "input.xyz")
                _log_status(
                    log_paths,
                    "GEOM",
                    f"refinement gas-phase SP geometry={solvent_opt_xyz_path.name} scratch={gas_sp_xyz}",
                )
                gas_sp_energy_kcal_mol = _run_gas_single_point_on_xyz(
                    calculators=calculators,
                    scratch_dir=gas_scratch,
                    xyz_path=gas_sp_xyz,
                    charge=charge,
                    multiplicity=multiplicity,
                    dry_run=dry_run,
                    log_paths=log_paths,
                )

            if gas_opt_xyz_path is not None:
                _progress(
                    f"computing RRHO contribution with {freq_label} frequencies at gas-phase-optimized geometry "
                    f"(T={temperature_k:g} K)"
                )
                _log_status(
                    log_paths,
                    "GEOM",
                    f"refinement RRHO primary geometry={gas_opt_xyz_path.name} fallback={solvent_opt_xyz_path.name}",
                )
            else:
                _progress(
                    f"computing RRHO contribution with {freq_label} frequencies at {solvent_opt_label} geometry "
                    f"(T={temperature_k:g} K)"
                )
                _log_status(
                    log_paths,
                    "GEOM",
                    f"refinement RRHO geometry={solvent_opt_xyz_path.name}",
                )
            freq_electronic_energy_kcal_mol, rrho_contribution_kcal_mol, _ = _try_frequencies_with_fallback(
                calculators=calculators,
                primary_xyz_path=gas_opt_xyz_path,
                fallback_xyz_path=solvent_opt_xyz_path,
                scratch_dir=scratch_dir,
                charge=charge,
                multiplicity=multiplicity,
                dry_run=dry_run,
                log_paths=log_paths,
                temperature_k=temperature_k,
            )
        except XtbFatalError as exc:
            report_xtb_fatal_and_exit(exc)
        except Exception as exc:
            _log_status(
                log_paths,
                "WARN",
                f"gas-phase or frequency steps failed after {solvation_label} solvation: {exc}",
            )

        _apply_energy_terms(
            terms,
            gas_sp_energy_kcal_mol=gas_sp_energy_kcal_mol,
            solvation_free_energy_kcal_mol=solvation_free_energy_kcal_mol,
            freq_electronic_energy_kcal_mol=freq_electronic_energy_kcal_mol,
            rrho_contribution_kcal_mol=rrho_contribution_kcal_mol,
            log_paths=log_paths,
        )
        return ConformerWorkflowResult(terms=terms, opt_xyz_path=solvent_opt_xyz_path)
    except XtbFatalError:
        raise
    except Exception as exc:
        _log_status(log_paths, "FAIL", f"single-conformer workflow failed: {exc}")
        terms.workflow_status = "optimization-failed"
        if terms.solvation_free_energy_kcal_mol == "not-run":
            terms.solvation_free_energy_kcal_mol = _format_energy_entry(None, failed_step="optimization")
        if terms.gas_sp_energy_kcal_mol == "not-run":
            terms.gas_sp_energy_kcal_mol = _format_energy_entry(None, failed_step="optimization")
        if terms.rrho_contribution_kcal_mol == "not-run":
            terms.rrho_contribution_kcal_mol = _format_energy_entry(None, failed_step="optimization")
        if terms.solution_phase_free_energy_kcal_mol == "not-run":
            terms.solution_phase_free_energy_kcal_mol = _format_energy_entry(None, failed_step="optimization")
        return ConformerWorkflowResult(terms=terms, opt_xyz_path=None)


def run_protomer_screening(
    protomer: Protomer,
    *,
    protomer_id: int | str = 0,
    scratch_root: str | Path = "./scratch_solvation",
    calculators: CalculatorSettings | None = None,
    conformer_mode: ConformerMode = "kdg",
    external_xyz_path: Optional[str | Path] = None,
    charge_override: Optional[int] = None,
    solvent: SolventNames | None = None,
    keep_scratch: bool = False,
    keep_logs: bool = False,
    keep_scratch_on_failure: bool = False,
    dry_run: bool = False,
    temperature_k: float = DEFAULT_TEMPERATURE_K,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> ScreeningWorkflowResult:
    """
    Lightweight pre-screening workflow for protomer pruning.

    Steps:
    1) Build/choose an initial KDG conformer geometry.
    2) Run the implicit-solvation SP (``solvation`` calculator) on the KDG geometry.
    3) Run the gas-phase SP (``gas`` calculator) on the KDG geometry.
    4) Run frequencies (``freq`` calculator) for RRHO on the KDG geometry.
    5) Compute screening solution-phase free energy.

    All geometry optimizations are skipped during screening.
    """
    def _progress(message: str, *, level: LogLevel = LogLevel.VERBOSE) -> None:
        emit_progress(progress_callback, message, level=level)

    if calculators is None:
        calculators = CalculatorSettings()
    calculators.validate()
    scratch_context = _create_scratch_context(scratch_root, protomer_id)
    scratch_dir = scratch_context.scratch_dir
    log_paths = scratch_context.log_paths

    if protomer.mol is None:
        raise ValueError("Protomer does not have mol; cannot run screening workflow.")
    charge = int(charge_override) if charge_override is not None else _formal_charge(protomer.mol)
    multiplicity = spin_multiplicity(protomer.mol)
    if solvent is None:
        solvent = resolve_solvent("water")
    _set_mol_prop_str(protomer.mol, "solvent", solvent.alpb)
    _set_mol_prop_str(protomer.mol, "calculator_settings", calculators.summary())
    _log_status(
        log_paths,
        "START",
        f"screening protomer_id={protomer_id} scratch_dir={scratch_dir.name} charge={charge} "
        f"multiplicity={multiplicity} solvent_alpb={solvent.alpb} solvent_cpcm={solvent.cpcm} "
        f"conformer_mode={conformer_mode} calculators=[{calculators.summary()}] "
        f"temperature_k={temperature_k:g}",
    )
    _progress("preparing conformer")

    conformer_energy_kcal_mol: Optional[float] = None
    solvation_free_energy_kcal_mol: Optional[float] = None
    gas_sp_energy_kcal_mol: Optional[float] = None
    rrho_contribution_kcal_mol: Optional[float] = None
    solution_phase_free_energy_kcal_mol: Optional[float] = None

    try:
        if dry_run:
            _progress("dry run enabled; skipping screening workflow", level=LogLevel.DEFAULT)
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

        workflow_result = _run_screening_conformer_workflow(
            protomer,
            mol,
            scratch_dir=scratch_dir,
            scratch_root=scratch_context.scratch_root,
            runtime_xyz_label=f"protomer_{protomer_id}",
            calculators=calculators,
            charge=charge,
            multiplicity=multiplicity,
            solvent=solvent,
            dry_run=dry_run,
            log_paths=log_paths,
            temperature_k=temperature_k,
            progress_callback=_progress,
        )
        terms = workflow_result.terms
        _persist_conformer_ensemble_results(
            protomer,
            charge=charge,
            conformer_terms=[terms],
            temperature_k=temperature_k,
        )

        gas_sp_energy_kcal_mol = (
            float(terms.gas_sp_energy_kcal_mol)
            if isinstance(terms.gas_sp_energy_kcal_mol, (int, float))
            else None
        )
        freq_electronic_energy_kcal_mol = (
            float(terms.freq_electronic_energy_kcal_mol)
            if isinstance(terms.freq_electronic_energy_kcal_mol, (int, float))
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
        _set_mol_prop_double(
            protomer.mol,
            "screening_freq_electronic_energy_kcal_mol",
            freq_electronic_energy_kcal_mol,
        )
        _set_mol_prop_double(protomer.mol, "screening_rrho_contribution_kcal_mol", rrho_contribution_kcal_mol)
        _set_mol_prop_double(
            protomer.mol,
            "screening_solution_phase_free_energy_kcal_mol",
            solution_phase_free_energy_kcal_mol,
        )
        if workflow_result.opt_xyz_path is not None and workflow_result.opt_xyz_path.is_file():
            _set_mol_prop_str(protomer.mol, "screening_opt_xyz_path", str(workflow_result.opt_xyz_path))
        _progress("finished screening workflow", level=LogLevel.VERBOSE)

    except XtbFatalError:
        raise
    except Exception as e:
        _progress(f"failed screening: {e}", level=LogLevel.DEFAULT)
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

    preserved_opt_path = _preserve_output_files(scratch_dir, keep_logs=keep_logs)
    if preserved_opt_path is not None and protomer.mol is not None:
        _set_mol_prop_str(protomer.mol, "screening_opt_xyz_path", str(preserved_opt_path))
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
    calculators: CalculatorSettings | None = None,
    conformer_mode: ConformerMode = "skip_search",
    external_xyz_path: Optional[str | Path] = None,
    charge_override: Optional[int] = None,
    solvent: SolventNames | None = None,
    opt_level: str = "loose",
    gas_optimize: bool = True,
    keep_scratch: bool = False,
    keep_logs: bool = False,
    keep_scratch_on_failure: bool = False,
    dry_run: bool = False,
    random_seed: int = 42,
    max_qm_conformers: int = REFINEMENT_MAX_QM_CONFORMERS,
    embedded_conformers: Optional[int] = None,
    conformer_energy_threshold_kcal_mol: float = DEFAULT_CONFORMER_ENERGY_THRESHOLD_KCAL_MOL,
    temperature_k: float = DEFAULT_TEMPERATURE_K,
    log_prefix: Optional[str] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> SolvationWorkflowResult:
    """
    Conformer-refinement workflow for protomers that pass screening.

    Generates a KDG conformer ensemble, quickly filters/prunes obvious duplicate
    embedded geometries, MMFF94-relaxes the resulting candidate pool, then
    selects up to max_qm_conformers from the relaxed/reranked pool and, per
    conformer, runs the implicit-solvent optimization (``solvent_opt``), the
    implicit-solvation SP (``solvation``) on that geometry, a gas-phase
    re-optimization and SP (``gas``; skipped when ``gas_optimize`` is False, in
    which case the gas SP uses the implicit-solvent geometry), and frequencies
    (``freq``) on the gas-phase geometry. Optimized structures are pruned again
    (keeping the lowest-energy instance of each duplicate), and the surviving
    conformer solution-phase energies are Boltzmann-weighted.
    """
    def _progress(message: str, *, level: LogLevel = LogLevel.VERBOSE) -> None:
        text = f"[{log_prefix}] {message}" if log_prefix else message
        emit_progress(progress_callback, text, level=level)

    if calculators is None:
        calculators = CalculatorSettings()
    calculators.validate()
    scratch_context = _create_scratch_context(scratch_root, protomer_id)
    scratch_dir = scratch_context.scratch_dir
    log_paths = scratch_context.log_paths

    if protomer.mol is None:
        raise ValueError("Protomer does not have mol; cannot run conformer refinement.")
    charge = int(charge_override) if charge_override is not None else _formal_charge(protomer.mol)
    multiplicity = spin_multiplicity(protomer.mol)
    if solvent is None:
        solvent = resolve_solvent("water")
    _set_mol_prop_str(protomer.mol, "solvent", solvent.alpb)
    _set_mol_prop_str(protomer.mol, "gas_optimize", "true" if gas_optimize else "false")
    _set_mol_prop_str(protomer.mol, "calculator_settings", calculators.summary())
    _log_status(
        log_paths,
        "START",
        f"conformer refinement protomer_id={protomer_id} scratch_dir={scratch_dir.name} "
        f"charge={charge} multiplicity={multiplicity} solvent_alpb={solvent.alpb} "
        f"solvent_cpcm={solvent.cpcm} calculators=[{calculators.summary()}] "
        f"gas_optimize={gas_optimize} temperature_k={temperature_k:g}",
    )

    conformer_energy_kcal_mol: Optional[float] = None
    solvation_free_energy_kcal_mol: Optional[float] = None
    gas_sp_energy_kcal_mol: Optional[float] = None
    rrho_contribution_kcal_mol: Optional[float] = None
    solution_phase_free_energy_kcal_mol: Optional[float] = None
    final_opt_xyz: Optional[Path] = None
    kept_conformer_indices: frozenset[int] = frozenset()

    try:
        if dry_run:
            _progress("dry run enabled; skipping conformer refinement", level=LogLevel.DEFAULT)
            _log_status(log_paths, "SKIP", "dry_run enabled; skipping conformer refinement")
            return SolvationWorkflowResult(
                conformer_energy_kcal_mol=None,
                optimized_xyz=None,
                solvation_free_energy_kcal_mol=None,
                gas_sp_energy_kcal_mol=None,
                rrho_contribution_kcal_mol=None,
                solution_phase_free_energy_kcal_mol=None,
                stdout_tail="dry_run; skipped conformer refinement.",
            )

        if gas_optimize:
            _progress(f"{calculators.label('gas')} gas-phase re-opt. enabled for conformer refinement")
        else:
            _progress(
                f"{calculators.label('solvent_opt')} geometry used for {calculators.label('gas')} SP "
                f"and {calculators.label('freq')} frequencies"
            )

        graph_mol = Chem.MolFromSmiles(protomer.smiles)
        if graph_mol is None:
            raise ValueError(f"Could not rebuild 3D graph from SMILES: {protomer.smiles}")

        if embedded_conformers is not None and int(embedded_conformers) < int(max_qm_conformers):
            raise ValueError(
                f"embedded_conformers ({embedded_conformers}) must be greater than "
                f"max_qm_conformers ({max_qm_conformers})."
            )

        _progress("generating KDG conformer ensemble")
        mol_h, ranked_conf_ids = _generate_kdg_conformer_ensemble(
            graph_mol,
            random_seed=random_seed,
            embedded_conformers=embedded_conformers,
            log_paths=log_paths,
        )
        filtered_conf_ids = _filter_ranked_embedded_conformers(
            mol_h,
            ranked_conf_ids,
            graph_mol,
            energy_threshold_kcal_mol=conformer_energy_threshold_kcal_mol,
        )
        candidate_pool_conf_ids = _prune_redundant_conf_ids(
            mol_h,
            filtered_conf_ids,
            log_paths=log_paths,
        )
        _log_status(
            log_paths,
            "OK",
            f"embedded={len(ranked_conf_ids)} "
            f"energy_connectivity_filter={len(filtered_conf_ids)} "
            f"candidate_pool={len(candidate_pool_conf_ids)} "
            f"max_qm={max_qm_conformers} "
            f"energy_threshold={conformer_energy_threshold_kcal_mol:.2f} kcal/mol",
        )

        _progress("relaxing candidate conformer pool with MMFF94")
        _optimize_mmff94_conformers(mol_h, candidate_pool_conf_ids, log_paths=log_paths)
        ranked_relaxed_conf_ids = sorted(
            candidate_pool_conf_ids,
            key=lambda cid: _mmff94_conformer_energy_kcal_mol(mol_h, cid) or float("inf"),
        )
        n_before_mmff94_prune = len(ranked_relaxed_conf_ids)
        deduplicated_relaxed_conf_ids = _prune_redundant_conf_ids(
            mol_h,
            ranked_relaxed_conf_ids,
            log_paths=log_paths,
        )
        selected_conf_ids = deduplicated_relaxed_conf_ids[: int(max_qm_conformers)]
        _log_status(
            log_paths,
            "OK",
            f"MMFF94-relaxed {n_before_mmff94_prune} conformers; "
            f"pre_qm_redundant_prune={n_before_mmff94_prune - len(deduplicated_relaxed_conf_ids)} "
            f"relaxed_candidate_pool={len(deduplicated_relaxed_conf_ids)} "
            f"selected_for_qm={len(selected_conf_ids)}",
        )

        pool_entries: list[ConformerPoolEntry] = []
        run_records: list[tuple[str, ConformerEnergyTerms]] = []
        n_selected = len(selected_conf_ids)
        for conf_idx, conf_id in enumerate(selected_conf_ids):
            conf_mol = _mol_from_conf_id(mol_h, conf_id, remove_hydrogens=False)
            conf_scratch = scratch_dir / f"conformer_{conf_idx}"
            conf_protomer = copy.deepcopy(protomer)
            conf_protomer.mol = conf_mol
            conf_label = f"conf {conf_idx + 1}/{n_selected}"
            conf_prefix = f"{log_prefix} {conf_label}" if log_prefix else conf_label

            def _conf_progress(message: str, prefix: str = conf_prefix, *, level: LogLevel = LogLevel.VERBOSE) -> None:
                formatted = f"[{prefix}] {message}"
                emit_progress(progress_callback, formatted, level=level)

            _conf_progress("running QM workflow", level=LogLevel.DEFAULT)
            workflow_result = _run_single_conformer_workflow(
                conf_protomer,
                conf_mol,
                scratch_dir=conf_scratch,
                scratch_root=scratch_context.scratch_root,
                runtime_xyz_label=f"protomer_{protomer_id}_conformer_{conf_idx}",
                calculators=calculators,
                charge=charge,
                multiplicity=multiplicity,
                solvent=solvent,
                opt_level=opt_level,
                gas_optimize=gas_optimize,
                dry_run=dry_run,
                log_paths=log_paths,
                temperature_k=temperature_k,
                progress_callback=_conf_progress,
            )
            _copy_warning_flags_from_conformer(protomer, conf_protomer)
            run_records.append((conf_label, workflow_result.terms))
            if isinstance(workflow_result.terms.solution_phase_free_energy_kcal_mol, (int, float)):
                pool_entries.append(
                    ConformerPoolEntry(
                        label=conf_label,
                        terms=workflow_result.terms,
                        opt_xyz_path=workflow_result.opt_xyz_path,
                        conformer_index=conf_idx,
                    )
                )

        pruned_pool = _prune_redundant_pool_entries(
            pool_entries,
            reference_mol=graph_mol,
            log_paths=log_paths,
        )
        if not pruned_pool and pool_entries:
            lowest_entry = min(
                pool_entries,
                key=lambda entry: float(entry.terms.solution_phase_free_energy_kcal_mol),
            )
            _log_status(
                log_paths,
                "WARN",
                "redundancy pruning removed all conformers; retaining lowest-energy conformer",
            )
            pruned_pool = [lowest_entry]
        conformer_terms = [entry.terms for entry in pruned_pool]
        conformer_labels = [entry.label for entry in pruned_pool]
        kept_conformer_indices = frozenset(
            entry.conformer_index
            for entry in pruned_pool
            if entry.conformer_index is not None
        )
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
                    previous_mol = protomer.mol
                    protomer.mol = opt_mol
                    protomer.mol.SetProp("connectivity_mismatch", "false")
                    if previous_mol is not None:
                        for key in (
                            "scf_convergence_retry",
                            "geometry_reoptimization_retry",
                            "geometry_fallback",
                        ):
                            if _optional_bool_prop(previous_mol, key):
                                _set_mol_prop_bool(protomer.mol, key, True)

        _persist_conformer_ensemble_results(
            protomer,
            charge=charge,
            conformer_terms=conformer_terms,
            conformer_labels=conformer_labels,
            temperature_k=temperature_k,
        )
        _set_mol_prop_str(protomer.mol, "workflow_status", "conformer_refined")
        _set_mol_prop_str(protomer.mol, "calculator_settings", calculators.summary())

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

        _log_conformer_summary(
            log_paths,
            log_prefix=log_prefix,
            run_records=run_records,
            pruned_pool=pruned_pool,
            aggregate_solution_energy=solution_phase_free_energy_kcal_mol,
            progress_callback=progress_callback,
        )
        _progress("finished conformer refinement", level=LogLevel.DEFAULT)

    except XtbFatalError:
        raise
    except Exception as exc:
        _progress(f"failed: {exc}", level=LogLevel.DEFAULT)
        _log_status(log_paths, "FAIL", f"conformer refinement exception for protomer_id={protomer_id}: {exc}")
        warnings.warn(
            f"Conformer refinement failed for protomer_id={protomer_id}: {exc}",
            RuntimeWarning,
        )
        if protomer.mol is not None:
            _set_mol_prop_str(protomer.mol, "workflow_error", str(exc)[:4000])
        preserved_opt_path = _preserve_output_files(scratch_dir, keep_logs=keep_logs)
        _cleanup_scratch_dir(
            scratch_dir,
            keep_scratch=keep_scratch or keep_scratch_on_failure,
            dry_run=dry_run,
            log_paths=log_paths,
            success=False,
        )
        return SolvationWorkflowResult(
            conformer_energy_kcal_mol=conformer_energy_kcal_mol,
            optimized_xyz=preserved_opt_path if preserved_opt_path is not None else final_opt_xyz,
            solvation_free_energy_kcal_mol=solvation_free_energy_kcal_mol,
            gas_sp_energy_kcal_mol=gas_sp_energy_kcal_mol,
            rrho_contribution_kcal_mol=rrho_contribution_kcal_mol,
            solution_phase_free_energy_kcal_mol=solution_phase_free_energy_kcal_mol,
            stdout_tail=str(exc)[-4000:],
        )

    final_preserved_opt_path = _preserve_output_files(
        scratch_dir,
        keep_logs=keep_logs,
        kept_conformer_indices=kept_conformer_indices if kept_conformer_indices else None,
    )
    if kept_conformer_indices:
        _remove_unkept_conformer_artifacts(
            scratch_root=scratch_context.scratch_root,
            protomer_id=protomer_id,
            kept_conformer_indices=kept_conformer_indices,
            log_paths=log_paths,
        )
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
        optimized_xyz=final_preserved_opt_path if final_preserved_opt_path is not None else final_opt_xyz,
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
