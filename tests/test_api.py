"""Unit tests for the high-level Python API."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

import peace
from peace.main import PeaceResult

GLYCINE_SMILES = "NCC(=O)O"


def test_enumerate_protomers_glycine_charge_range():
    result = peace.enumerate_protomers(GLYCINE_SMILES, charge_min=-1, charge_max=1)
    assert isinstance(result, PeaceResult)
    assert sorted(result.species_by_charge) == [-1, 0, 1]
    assert set(result.dataframe["formal_charge"]) == {-1, 0, 1}
    neutral = result.dataframe[result.dataframe["formal_charge"] == 0]
    assert {"NCC(=O)O", "[NH3+]CC(=O)[O-]"} <= set(neutral["protomer_smiles"])
    assert result.pka_result is None


def test_run_matches_cli_options(tmp_path: Path):
    out_csv = tmp_path / "out.csv"
    result = peace.run(
        GLYCINE_SMILES,
        only_protomer_search=True,
        site_search_mode="none",
        add_tautomers="[NH3+]CC(=O)[O-]",
        output_csv=out_csv,
    )
    assert out_csv.is_file()
    assert set(result.dataframe["protomer_smiles"]) == {"NCC(=O)O", "[NH3+]CC(=O)[O-]"}


def test_api_does_not_write_csv_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    peace.enumerate_protomers(GLYCINE_SMILES)
    assert not any(tmp_path.iterdir())


def test_api_rejects_unknown_and_invalid_options():
    with pytest.raises(TypeError, match="not_an_option"):
        peace.run(GLYCINE_SMILES, not_an_option=1)
    with pytest.raises(ValueError, match="--charge-min must be <= --charge-max"):
        peace.enumerate_protomers(GLYCINE_SMILES, charge_min=1, charge_max=0)
    with pytest.raises(ValueError, match="--pka requires --solvation"):
        peace.enumerate_protomers(GLYCINE_SMILES, charge_min=0, charge_max=1, pka=True)
    with pytest.raises(ValueError, match="solvate"):
        peace.enumerate_protomers(GLYCINE_SMILES, solvation=True)


def test_cli_defaults_covers_all_cli_options():
    defaults = peace.cli_defaults()
    assert defaults["no_plot"] is True
    assert defaults["output_csv"] is None
    assert defaults["charge_min"] == 0
    assert "solv_sp_calculator" in defaults


def test_solvate_dry_run_with_pka(tmp_path: Path):
    result = peace.solvate(
        GLYCINE_SMILES,
        charge_min=0,
        charge_max=1,
        dry_run=True,
        gas_calculator="aimnet2",
        solv_opt_calculator="aimnet2",
        pka=True,
        proton_energy=-260.0,
        scratch_root=tmp_path / "scratch",
        output_csv=tmp_path / "out.csv",
    )
    assert sorted(result.species_by_charge) == [0, 1]
    assert result.pka_dataframe is not None


@pytest.mark.skipif(shutil.which("xtb") is None, reason="xtb binary not found on PATH")
def test_solvate_glycine_zwitterion(tmp_path: Path):
    result = peace.solvate(
        GLYCINE_SMILES,
        max_conformers=2,
        scratch_root=tmp_path / "scratch",
    )
    df = result.dataframe
    f_zwit = float(df[df["protomer_smiles"] == "[NH3+]CC(=O)[O-]"]["boltzmann_fraction"].iloc[0])
    assert f_zwit >= 0.99
