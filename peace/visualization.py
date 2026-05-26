"""
Protomer/tautomer structure visualization.

Works from a Species object when available, or directly from a results CSV dataframe.
"""

from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
from rdkit.Chem import AllChem, Draw

from .common import canon_smiles
from .protomer import Species


@dataclass
class ProtomerPlotEntry:
    species_id: str = ""
    formal_charge: Optional[int] = None
    tautomer_id: int = 0
    protomer_id: int = 0
    smiles: str = ""
    mol: Any = None
    ionization_sites: list[int] = field(default_factory=list)
    boltzmann_fraction: Optional[float] = None
    solution_phase_free_energy_kcal_mol: Optional[float] = None


def _optional_float(value) -> Optional[float]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _display_mol_for_entry(entry: ProtomerPlotEntry):
    mol = entry.mol
    if mol is None and entry.smiles:
        mol = AllChem.MolFromSmiles(entry.smiles)
    if mol is None:
        return None
    display_mol = copy.deepcopy(mol)
    display_mol.__sssAtoms = list(entry.ionization_sites)
    return display_mol


def entries_from_species(spec: Species, *, formal_charge: Optional[int] = None) -> list[ProtomerPlotEntry]:
    entries: list[ProtomerPlotEntry] = []
    for taut_idx, tautomer in spec.tautomers.items():
        for prot_idx, protomer in tautomer.protomers.items():
            boltzmann_fraction = None
            solution_energy = None
            if protomer.mol is not None:
                if protomer.mol.HasProp("boltzmann_fraction"):
                    boltzmann_fraction = _optional_float(protomer.mol.GetProp("boltzmann_fraction"))
                if protomer.mol.HasProp("solution_phase_free_energy_kcal_mol"):
                    solution_energy = _optional_float(
                        protomer.mol.GetProp("solution_phase_free_energy_kcal_mol")
                    )
            display_mol = protomer.input_mol if protomer.input_mol is not None else protomer.mol
            entries.append(
                ProtomerPlotEntry(
                    species_id=spec.key,
                    formal_charge=formal_charge,
                    tautomer_id=int(taut_idx),
                    protomer_id=int(prot_idx),
                    smiles=protomer.smiles,
                    mol=display_mol,
                    ionization_sites=list(protomer.ionization_sites),
                    boltzmann_fraction=boltzmann_fraction,
                    solution_phase_free_energy_kcal_mol=solution_energy,
                )
            )
    return entries


def entries_from_dataframe(df: pd.DataFrame) -> list[ProtomerPlotEntry]:
    entries: list[ProtomerPlotEntry] = []
    for row_dict in df.to_dict(orient="records"):
        smiles = str(row_dict.get("protomer_smiles", ""))
        entries.append(
            ProtomerPlotEntry(
                species_id=str(row_dict.get("species_id", "")),
                formal_charge=(
                    int(row_dict["formal_charge"])
                    if "formal_charge" in row_dict and pd.notna(row_dict["formal_charge"])
                    else None
                ),
                tautomer_id=int(row_dict.get("tautomer_id", 0)),
                protomer_id=int(row_dict.get("protomer_id", 0)),
                smiles=smiles,
                mol=AllChem.MolFromSmiles(smiles) if smiles else None,
                ionization_sites=[],
                boltzmann_fraction=_optional_float(row_dict.get("boltzmann_fraction")),
                solution_phase_free_energy_kcal_mol=_optional_float(
                    row_dict.get("solution_phase_free_energy_kcal_mol")
                ),
            )
        )
    return entries


def filter_plot_entries(
    entries: list[ProtomerPlotEntry],
    mode: str = "default",
    plot_filter: Optional[float] = None,
) -> list[ProtomerPlotEntry]:
    if mode == "default":
        return list(entries)

    if mode == "cutoff":
        if plot_filter is None:
            raise ValueError("--plot-filter is required when --visualization=cutoff")
        if not any(e.boltzmann_fraction is not None for e in entries):
            raise ValueError(
                "cutoff visualization requires boltzmann_fraction values in the data."
            )
        cutoff = float(plot_filter)
        return [e for e in entries if e.boltzmann_fraction is not None and e.boltzmann_fraction >= cutoff]

    if mode == "count":
        if plot_filter is None:
            raise ValueError("--plot-filter is required when --visualization=count")
        n_keep = int(plot_filter)
        if n_keep <= 0:
            return []

        ranked = [
            e
            for e in entries
            if e.solution_phase_free_energy_kcal_mol is not None or e.boltzmann_fraction is not None
        ]
        if not ranked:
            raise ValueError(
                "count visualization requires solution_phase_free_energy_kcal_mol or "
                "boltzmann_fraction values in the data."
            )

        def _rank_key(entry: ProtomerPlotEntry):
            if entry.solution_phase_free_energy_kcal_mol is not None:
                return (0, entry.solution_phase_free_energy_kcal_mol, -entry.protomer_id)
            return (1, -(entry.boltzmann_fraction or 0.0), entry.protomer_id)

        ranked.sort(key=_rank_key)
        return ranked[:n_keep]

    raise ValueError(f"Unknown visualization mode: {mode}")


def _legend_for_entry(entry: ProtomerPlotEntry) -> str:
    f_i = ""
    if entry.boltzmann_fraction is not None:
        f_i = f"f_i: {entry.boltzmann_fraction:.4f}"
    return f"ID: {entry.protomer_id} | SMILES: {entry.smiles}\n {f_i}".rstrip()


def plot_tautomer_entries(entries: list[ProtomerPlotEntry], n_columns: int) -> Any:
    """Plot a single tautomer's protomers in a grid. Returns an RDKit PIL image."""
    mols = [_display_mol_for_entry(entry) for entry in entries]
    legends = [_legend_for_entry(entry) for entry in entries]
    highlights = [list(entry.ionization_sites) for entry in entries]

    n_rows = int(np.ceil(len(mols) / n_columns)) if mols else 1
    n_padding = n_rows * n_columns - len(mols)
    mols.extend([None] * n_padding)
    legends.extend([""] * n_padding)
    highlights.extend([[] for _ in range(n_padding)])

    mols_matrix = np.reshape(mols, (n_rows, n_columns))
    legends_matrix = np.reshape(legends, (n_rows, n_columns))
    has_highlights = any(bool(h) for h in highlights)
    if has_highlights:
        highlights_matrix = [
            highlights[row * n_columns : (row + 1) * n_columns]
            for row in range(n_rows)
        ]
        return Draw.MolsMatrixToGridImage(
            molsMatrix=mols_matrix.tolist(),
            legendsMatrix=legends_matrix.tolist(),
            subImgSize=(300, 200),
            highlightAtomListsMatrix=highlights_matrix,
        )
    return Draw.MolsMatrixToGridImage(
        molsMatrix=mols_matrix.tolist(),
        legendsMatrix=legends_matrix.tolist(),
        subImgSize=(300, 200),
    )


def plot_entries(entries: list[ProtomerPlotEntry], *, n_columns: int = 5) -> list[Any]:
    if not entries:
        return []

    by_tautomer: dict[int, list[ProtomerPlotEntry]] = {}
    for entry in entries:
        by_tautomer.setdefault(int(entry.tautomer_id), []).append(entry)

    imgs = []
    for tautomer_id in sorted(by_tautomer.keys()):
        taut_entries = sorted(by_tautomer[tautomer_id], key=lambda e: e.protomer_id)
        imgs.append(plot_tautomer_entries(taut_entries, n_columns))
    return imgs


def _group_dataframe(df: pd.DataFrame) -> dict[tuple, pd.DataFrame]:
    group_cols = [c for c in ("species_id", "formal_charge") if c in df.columns]
    if not group_cols:
        return {(): df}
    grouped = {}
    for key, group_df in df.groupby(group_cols, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        grouped[key] = group_df.reset_index(drop=True)
    return grouped


def plot_from_dataframe(
    df: pd.DataFrame,
    *,
    mode: str = "default",
    plot_filter: Optional[float] = None,
    n_columns: int = 5,
) -> list[Any]:
    """Build protomer grid images from a results-style dataframe."""
    imgs: list[Any] = []
    for _group_key, group_df in _group_dataframe(df).items():
        entries = entries_from_dataframe(group_df)
        filtered = filter_plot_entries(entries, mode=mode, plot_filter=plot_filter)
        if not filtered:
            warnings.warn("No protomers matched the visualization filter; skipping group.")
            continue
        imgs.extend(plot_entries(filtered, n_columns=n_columns))
    return imgs


def plot_from_species(
    spec: Species,
    *,
    formal_charge: Optional[int] = None,
    mode: str = "default",
    plot_filter: Optional[float] = None,
    n_columns: int = 5,
) -> list[Any]:
    """Build protomer grid images from a Species object."""
    entries = entries_from_species(spec, formal_charge=formal_charge)
    filtered = filter_plot_entries(entries, mode=mode, plot_filter=plot_filter)
    return plot_entries(filtered, n_columns=n_columns)
