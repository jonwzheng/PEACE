"""
Protomer/tautomer structure visualization.

Works from a Species object when available, or directly from a results CSV dataframe.
"""

from __future__ import annotations

import copy
import textwrap
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from rdkit.Chem import AllChem, Draw

from .protomer import Species

# Layout
_MOL_SIZE = (280, 180)
_LEGEND_HEIGHT = 108
_POP_BAR_HEIGHT = 8
_CELL_PAD = 6
_GRID_GAP = 2
_TITLE_HEIGHT = 32
_HIGHLIGHT_MAX_RANK = 5  # highlight population ranks 0 .. 4
_DEFAULT_MAX_PLOT_COUNT = 5
_DEFAULT_POPULATION_CUTOFF = 0.0001  # 0.01% Boltzmann fraction
_HIGHLIGHT_COLOR = (0, 105, 75)
_POP_BAR_COLOR = (0, 105, 75)
_POP_BAR_TRACK_COLOR = (230, 234, 236)
_FONT_SIZE = 14
_FONT_SIZE_SMALL = 13
_FONT_SIZE_TITLE = 15


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
    delta_g_kcal_mol: Optional[float] = None
    workflow_status: str = ""
    workflow_error: str = ""
    connectivity_mismatch: bool = False


def _optional_float(value) -> Optional[float]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_bool(value) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "1", "yes")


def _optional_str(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def _is_relaxed_convergence_status(status: str) -> bool:
    return status.startswith("optimization_retried_with_convergence:")


def _has_workflow_issue(entry: ProtomerPlotEntry) -> bool:
    if entry.connectivity_mismatch:
        return True
    if entry.workflow_error:
        return True
    if entry.workflow_status and not _is_relaxed_convergence_status(entry.workflow_status):
        return True
    return False


def _has_reported_thermo(entry: ProtomerPlotEntry) -> bool:
    return (
        entry.boltzmann_fraction is not None
        or entry.delta_g_kcal_mol is not None
        or entry.solution_phase_free_energy_kcal_mol is not None
    )


def _species_has_thermo(entries: list[ProtomerPlotEntry]) -> bool:
    return any(_has_reported_thermo(e) for e in entries)


def _issue_marker(entry: ProtomerPlotEntry, *, species_has_thermo: bool) -> Optional[str]:
    if not species_has_thermo:
        return None
    issue = _has_workflow_issue(entry)
    has_thermo = _has_reported_thermo(entry)
    if issue and not has_thermo:
        return "*"
    if issue and has_thermo:
        return "!"
    if not has_thermo:
        return "*"
    if _is_relaxed_convergence_status(entry.workflow_status):
        return "/"
    return None


def _format_fraction_pct(fraction: float) -> str:
    pct = 100.0 * fraction
    if pct >= 10.0:
        return f"{pct:.2f}%"
    if pct >= 0.01:
        return f"{pct:.4f}%"
    if pct >= 0.0001:
        return f"{pct:.6f}%"
    return f"{pct:.2e}%"


def _max_species_fraction(entries: list[ProtomerPlotEntry]) -> Optional[float]:
    fractions = [e.boltzmann_fraction for e in entries if e.boltzmann_fraction is not None]
    return max(fractions) if fractions else None


def _tautomer_fraction_sum(entries: list[ProtomerPlotEntry]) -> Optional[float]:
    fractions = [e.boltzmann_fraction for e in entries if e.boltzmann_fraction is not None]
    if not fractions:
        return None
    return float(sum(fractions))


def _load_font(*, size: int = 11) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except (OSError, TypeError):
        return ImageFont.load_default()


def _draw_emphasis_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    *,
    font: ImageFont.ImageFont,
    fill: str = "#222222",
) -> None:
    draw.text(xy, text, font=font, fill=fill, stroke_width=0.5, stroke_fill=fill)


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
            delta_g = None
            workflow_status = ""
            workflow_error = ""
            connectivity_mismatch = False
            if protomer.mol is not None:
                if protomer.mol.HasProp("boltzmann_fraction"):
                    boltzmann_fraction = _optional_float(protomer.mol.GetProp("boltzmann_fraction"))
                if protomer.mol.HasProp("solution_phase_free_energy_kcal_mol"):
                    solution_energy = _optional_float(
                        protomer.mol.GetProp("solution_phase_free_energy_kcal_mol")
                    )
                if protomer.mol.HasProp("delta_g_kcal_mol"):
                    delta_g = _optional_float(protomer.mol.GetProp("delta_g_kcal_mol"))
                if protomer.mol.HasProp("workflow_status"):
                    workflow_status = _optional_str(protomer.mol.GetProp("workflow_status"))
                if protomer.mol.HasProp("workflow_error"):
                    workflow_error = _optional_str(protomer.mol.GetProp("workflow_error"))
                if protomer.mol.HasProp("connectivity_mismatch"):
                    connectivity_mismatch = _optional_bool(protomer.mol.GetProp("connectivity_mismatch"))
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
                    delta_g_kcal_mol=delta_g,
                    workflow_status=workflow_status,
                    workflow_error=workflow_error,
                    connectivity_mismatch=connectivity_mismatch,
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
                delta_g_kcal_mol=_optional_float(row_dict.get("delta_g_kcal_mol")),
                workflow_status=_optional_str(row_dict.get("workflow_status")),
                workflow_error=_optional_str(row_dict.get("workflow_error")),
                connectivity_mismatch=_optional_bool(row_dict.get("connectivity_mismatch")),
            )
        )
    return entries


def filter_plot_entries(
    entries: list[ProtomerPlotEntry],
    mode: str = "default",
    plot_filter: Optional[float] = None,
) -> list[ProtomerPlotEntry]:
    if mode == "default":
        if len(entries) <= _DEFAULT_MAX_PLOT_COUNT:
            return list(entries)
        if not any(e.boltzmann_fraction is not None for e in entries):
            return list(entries)
        return [
            e
            for e in entries
            if e.boltzmann_fraction is not None
            and e.boltzmann_fraction >= _DEFAULT_POPULATION_CUTOFF
        ]

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


def _species_fraction_ranks(entries: list[ProtomerPlotEntry]) -> dict[tuple[int, int], int]:
    """Map (tautomer_id, protomer_id) -> rank across the full species (0 = highest f)."""
    ranked = [e for e in entries if e.boltzmann_fraction is not None]
    if not ranked:
        return {}
    ranked.sort(
        key=lambda e: (-e.boltzmann_fraction, e.tautomer_id, e.protomer_id)
    )
    return {(e.tautomer_id, e.protomer_id): idx for idx, e in enumerate(ranked)}


def _draw_labeled_value(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    label: str,
    value: str,
    *,
    font: ImageFont.ImageFont,
    fill: str = "#222222",
) -> float:
    _draw_emphasis_text(draw, (x, y), label, font=font, fill=fill)
    x += draw.textlength(label, font=font)
    draw.text((x, y), value, font=font, fill=fill)
    return x + draw.textlength(value, font=font)


def _wrap_smiles(smiles: str, width: int = 42) -> list[str]:
    if len(smiles) <= width:
        return [smiles]
    return textwrap.wrap(smiles, width=width, break_long_words=True, break_on_hyphens=False) or [smiles]


def _draw_legend(
    entry: ProtomerPlotEntry,
    *,
    width: int,
    species_rank: Optional[int],
) -> Image.Image:
    font = _load_font(size=_FONT_SIZE)
    font_small = _load_font(size=_FONT_SIZE_SMALL)

    legend = Image.new("RGB", (width, _LEGEND_HEIGHT), "white")
    draw = ImageDraw.Draw(legend)

    y = 2
    x = 4
    x = _draw_labeled_value(
        draw, x, y, "ID: ", str(entry.protomer_id),
        font=font,
    )
    draw.text((x + 6, y), " ", font=font, fill="#888888")
    x = x + 6 + draw.textlength(" ", font=font) + 6
    _draw_emphasis_text(draw, (x, y), "SMILES:", font=font)
    smiles_label_w = draw.textlength("SMILES:", font=font) + 4

    smiles_lines = _wrap_smiles(entry.smiles, width=max(24, (width - 12) // 7))
    smiles_y = y
    if smiles_lines:
        draw.text((x + smiles_label_w, smiles_y), smiles_lines[0], font=font_small, fill="#333333")
        for line in smiles_lines[1:]:
            smiles_y += 16
            draw.text((x + smiles_label_w, smiles_y), line, font=font_small, fill="#333333")

    y = smiles_y + 18
    x = 4
    if species_rank is not None:
        x = _draw_labeled_value(
            draw, x, y, "rank: ", str(species_rank),
            font=font,
        )
        x += 10

    if entry.delta_g_kcal_mol is not None:
        x = _draw_labeled_value(
            draw, x, y, "diff G: ", f"{entry.delta_g_kcal_mol:.2f} kcal/mol",
            font=font,
        )
        x += 10

    if entry.boltzmann_fraction is not None:
        _draw_labeled_value(
            draw, x, y, "f: ", _format_fraction_pct(entry.boltzmann_fraction),
            font=font,
        )

    return legend


def _draw_population_bar(width: int, fraction: Optional[float], *, max_fraction: Optional[float]) -> Image.Image:
    bar = Image.new("RGB", (width, _POP_BAR_HEIGHT), "white")
    if fraction is None or max_fraction is None or max_fraction <= 0:
        return bar
    draw = ImageDraw.Draw(bar)
    draw.rectangle([0, 1, width - 1, _POP_BAR_HEIGHT - 2], fill=_POP_BAR_TRACK_COLOR)
    fill_w = max(1, int(round(width * fraction / max_fraction)))
    draw.rectangle([0, 1, fill_w - 1, _POP_BAR_HEIGHT - 2], fill=_POP_BAR_COLOR)
    return bar


def _draw_issue_marker(img: Image.Image, marker: Optional[str]) -> Image.Image:
    if not marker:
        return img
    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)
    font = _load_font(size=_FONT_SIZE + 4)
    color = "#b45309" if marker == "!" else "#2563eb" if marker == "/" else "#b91c1c"
    tw = draw.textlength(marker, font=font)
    x = img.width - tw - 8
    y = 4
    pad = 3
    draw.rounded_rectangle(
        [x - pad, y - pad, x + tw + pad, y + _FONT_SIZE + pad],
        radius=4,
        fill="white",
        outline=color,
        width=2,
    )
    draw.text((x, y), marker, font=font, fill=color)
    return overlay


def _draw_rank_highlight(cell: Image.Image, rank: Optional[int]) -> Image.Image:
    if rank is None or rank >= _HIGHLIGHT_MAX_RANK:
        return cell

    overlay = Image.new("RGBA", cell.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = cell.size

    base_alpha = int(230 * (1.0 - 0.17 * rank))
    base_width = max(2, 6 - rank)
    n_layers = max(1, 4 - rank)

    for layer in range(n_layers):
        inset = layer * 2
        alpha = max(35, int(base_alpha * (1.0 - 0.32 * layer)))
        line_w = max(1, base_width - layer)
        draw.rectangle(
            [inset, inset, w - 1 - inset, h - 1 - inset],
            outline=(*_HIGHLIGHT_COLOR, alpha),
            width=line_w,
        )

    base = cell.convert("RGBA")
    return Image.alpha_composite(base, overlay).convert("RGB")


def _mol_panel(mol, *, highlights: list[int]) -> Image.Image:
    if mol is None:
        panel = Image.new("RGB", _MOL_SIZE, "white")
        draw = ImageDraw.Draw(panel)
        font = _load_font(size=_FONT_SIZE + 2)
        text = "invalid structure"
        tw = draw.textlength(text, font=font)
        draw.text(((_MOL_SIZE[0] - tw) / 2, (_MOL_SIZE[1] - 14) / 2), text, font=font, fill="#999999")
        return panel

    kwargs: dict[str, Any] = {"size": _MOL_SIZE}
    if highlights:
        kwargs["highlightAtoms"] = highlights
    img = Draw.MolToImage(mol, **kwargs)
    if img.mode != "RGB":
        img = img.convert("RGB")
    if img.size != _MOL_SIZE:
        img = img.resize(_MOL_SIZE, Image.Resampling.LANCZOS)
    return img


def _compose_cell(
    entry: ProtomerPlotEntry,
    *,
    species_rank: Optional[int],
    max_fraction: Optional[float] = None,
    species_has_thermo: bool = False,
) -> Image.Image:
    inner_w = _MOL_SIZE[0]
    cell_w = inner_w + 2 * _CELL_PAD
    cell_h = _MOL_SIZE[1] + _POP_BAR_HEIGHT + _LEGEND_HEIGHT + 2 * _CELL_PAD

    mol = _display_mol_for_entry(entry)
    mol_img = _mol_panel(mol, highlights=list(entry.ionization_sites))
    mol_img = _draw_issue_marker(mol_img, _issue_marker(entry, species_has_thermo=species_has_thermo))
    pop_bar = _draw_population_bar(inner_w, entry.boltzmann_fraction, max_fraction=max_fraction)
    legend_img = _draw_legend(
        entry,
        width=inner_w,
        species_rank=species_rank,
    )

    cell = Image.new("RGB", (cell_w, cell_h), "white")
    cell.paste(mol_img, (_CELL_PAD, _CELL_PAD))
    cell.paste(pop_bar, (_CELL_PAD, _CELL_PAD + _MOL_SIZE[1]))
    cell.paste(legend_img, (_CELL_PAD, _CELL_PAD + _MOL_SIZE[1] + _POP_BAR_HEIGHT))
    return _draw_rank_highlight(cell, species_rank)


def _panel_title(
    entries: list[ProtomerPlotEntry],
    width: int,
    *,
    tautomer_fraction_sum: Optional[float] = None,
) -> Optional[Image.Image]:
    if not entries:
        return None
    first = entries[0]
    parts = [f"Tautomer {first.tautomer_id + 1}"]
    if tautomer_fraction_sum is not None:
        parts.append(f"f: {_format_fraction_pct(tautomer_fraction_sum)}")
    if first.species_id:
        parts.append(f"species: {first.species_id}")
    if first.formal_charge is not None:
        parts.append(f"charge: {first.formal_charge:+d}")
    title = "  ·  ".join(parts)

    banner = Image.new("RGB", (width, _TITLE_HEIGHT), "#f4f6f8")
    draw = ImageDraw.Draw(banner)
    font = _load_font(size=_FONT_SIZE_TITLE)
    _draw_emphasis_text(draw, (8, 6), title, font=font, fill="#1a1a1a")
    draw.line([(0, _TITLE_HEIGHT - 1), (width, _TITLE_HEIGHT - 1)], fill="#d0d4d8", width=1)
    return banner


def plot_tautomer_entries(
    entries: list[ProtomerPlotEntry],
    n_columns: int,
    *,
    species_ranks: Optional[dict[tuple[int, int], int]] = None,
    max_fraction: Optional[float] = None,
    species_has_thermo: bool = False,
) -> Any:
    """Plot a single tautomer's protomers in a grid. Returns a PIL image."""
    if not entries:
        return Image.new("RGB", (_MOL_SIZE[0], _MOL_SIZE[1]), "white")

    ranks = species_ranks or {}
    tautomer_f = _tautomer_fraction_sum(entries)
    cells = [
        _compose_cell(
            entry,
            species_rank=ranks.get((entry.tautomer_id, entry.protomer_id)),
            max_fraction=max_fraction,
            species_has_thermo=species_has_thermo,
        )
        for entry in entries
    ]

    n_rows = int(np.ceil(len(cells) / n_columns))
    n_padding = n_rows * n_columns - len(cells)
    cell_w, cell_h = cells[0].size
    for _ in range(n_padding):
        cells.append(Image.new("RGB", (cell_w, cell_h), "white"))

    grid_w = n_columns * cell_w + (n_columns - 1) * _GRID_GAP
    grid_h = n_rows * cell_h + (n_rows - 1) * _GRID_GAP
    title = _panel_title(entries, grid_w, tautomer_fraction_sum=tautomer_f)
    total_h = grid_h + (title.height if title else 0)

    canvas = Image.new("RGB", (grid_w, total_h), "white")
    y_offset = title.height if title else 0
    if title:
        canvas.paste(title, (0, 0))

    for idx, cell in enumerate(cells):
        row, col = divmod(idx, n_columns)
        x = col * (cell_w + _GRID_GAP)
        y = y_offset + row * (cell_h + _GRID_GAP)
        canvas.paste(cell, (x, y))

    return canvas


def plot_entries(entries: list[ProtomerPlotEntry], *, n_columns: int = 5) -> list[Any]:
    if not entries:
        return []

    by_tautomer: dict[int, list[ProtomerPlotEntry]] = {}
    for entry in entries:
        by_tautomer.setdefault(int(entry.tautomer_id), []).append(entry)

    species_ranks = _species_fraction_ranks(entries)
    max_fraction = _max_species_fraction(entries)
    species_has_thermo = _species_has_thermo(entries)
    imgs = []
    for tautomer_id in sorted(by_tautomer.keys()):
        taut_entries = by_tautomer[tautomer_id]
        if species_ranks:
            taut_entries = sorted(
                taut_entries,
                key=lambda e: (
                    species_ranks.get((e.tautomer_id, e.protomer_id), 10**9),
                    e.protomer_id,
                ),
            )
        else:
            taut_entries = sorted(taut_entries, key=lambda e: e.protomer_id)
        imgs.append(
            plot_tautomer_entries(
                taut_entries,
                n_columns,
                species_ranks=species_ranks,
                max_fraction=max_fraction,
                species_has_thermo=species_has_thermo,
            )
        )
    return imgs


def resolve_plot_save_path(base_path: str | Path, *labels: str) -> Path:
    """Build a plot output path, appending labels before the suffix when provided."""
    path = Path(base_path)
    if not labels:
        return path
    suffix = path.suffix or ".png"
    stem = path.stem
    label = "_".join(str(part) for part in labels if part is not None and str(part) != "")
    return path.with_name(f"{stem}_{label}{suffix}")


def save_plot_images(
    imgs: list,
    path: str | Path,
    *,
    mode: str = "vertical",
) -> Path:
    """Combine plot panels and write a single image file."""
    from .common import combine_images

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    combine_images(imgs, mode=mode).save(out)
    return out


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
