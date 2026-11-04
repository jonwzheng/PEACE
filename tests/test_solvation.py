"""Unit tests for the PEACE solvation workflow and --override-solvation behavior."""

from __future__ import annotations

import runpy
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

import peace.main
from peace.protomer import Species
from peace.solvation import run_species_solvation

GLYCINE_SMILES = "NCC(=O)O"


def _run_peace_cli(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["peace.main", *argv])
    runpy.run_path(str(Path(peace.main.__file__)), run_name="__main__")


@pytest.mark.skipif(shutil.which("xtb") is None, reason="xtb binary not found on PATH")
def test_glycine_solvation_zwitterion_and_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """
    Verify that:
      1) Solvation on glycine yields ~99% zwitterion ([NH3+]CC(=O)[O-]).
      2) Re-running on the same scratch directory without --override-solvation raises FileExistsError.
      3) Re-running with --override-solvation succeeds and reproduces ~99% zwitterion.
    """
    scratch_root = tmp_path / "solvation_results"
    output_csv = tmp_path / "glycine_results.csv"

    base_args = [
        "--smiles",
        GLYCINE_SMILES,
        "--solvation",
        "--no-plot",
        "--max-conformers",
        "2",
        "--scratch-root",
        str(scratch_root),
        "--output-csv",
        str(output_csv),
    ]

    # 1. Initial solvation run on glycine
    _run_peace_cli(monkeypatch, base_args)

    assert output_csv.is_file()
    df = pd.read_csv(output_csv)
    assert len(df) == 2
    assert set(df["protomer_smiles"]) == {"NCC(=O)O", "[NH3+]CC(=O)[O-]"}
    assert set(df["workflow_status"]) == {"conformer_refined"}

    zwit_rows = df[df["is_zwitterion"] == True]  # noqa: E712
    neutral_rows = df[df["is_zwitterion"] == False]  # noqa: E712
    assert len(zwit_rows) == 1
    assert len(neutral_rows) == 1

    f_zwit = float(zwit_rows["boltzmann_fraction"].iloc[0])
    f_neutral = float(neutral_rows["boltzmann_fraction"].iloc[0])
    assert f_zwit + f_neutral == pytest.approx(1.0, abs=1e-6)
    assert f_zwit >= 0.99, f"Expected glycine zwitterion fraction >= 0.99, got {f_zwit}"

    # 2. Re-running without --override-solvation must fail with FileExistsError
    with pytest.raises(FileExistsError, match="Existing solvation results detected"):
        _run_peace_cli(monkeypatch, base_args)

    # 3. Re-running with --override-solvation must succeed
    override_csv = tmp_path / "glycine_override_results.csv"
    override_args = [
        *base_args[:-1],
        str(override_csv),
        "--override-solvation",
    ]
    _run_peace_cli(monkeypatch, override_args)

    assert override_csv.is_file()
    df_override = pd.read_csv(override_csv)
    assert len(df_override) == 2
    f_zwit_override = float(
        df_override[df_override["is_zwitterion"] == True]["boltzmann_fraction"].iloc[0]  # noqa: E712
    )
    assert f_zwit_override >= 0.99


def test_run_species_solvation_override_flag_dry_run(tmp_path: Path):
    """Verify run_species_solvation enforces and honors override_solvation."""
    spec = Species.from_smiles(GLYCINE_SMILES)
    scratch_root = tmp_path / "species_scratch"

    run_species_solvation(
        spec,
        scratch_root=scratch_root,
        override_solvation=False,
        dry_run=True,
    )
    species_dir = scratch_root / f"species_{spec.key}"
    assert species_dir.is_dir()

    with pytest.raises(FileExistsError, match="Existing solvation results detected"):
        run_species_solvation(
            spec,
            scratch_root=scratch_root,
            override_solvation=False,
            dry_run=True,
        )

    with pytest.warns(UserWarning, match="OVERRIDE"):
        run_species_solvation(
            spec,
            scratch_root=scratch_root,
            override_solvation=True,
            dry_run=True,
        )
    assert species_dir.is_dir()
