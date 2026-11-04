"""AIMNet2 calculators, which need a manual `pip install "aimnet[ase]"`."""

from __future__ import annotations

from pathlib import Path

import pytest

import peace

pytestmark = [pytest.mark.xtb, pytest.mark.aimnet2]


def test_aimnet2_gas_and_solvent_optimization(tmp_path: Path):
    result = peace.solvate(
        "NCC(=O)O",
        gas_calculator="aimnet2",
        solv_opt_calculator="aimnet2",
        max_conformers=2,
        keep_logs=True,
        scratch_root=tmp_path,
    )
    df = result.dataframe
    assert set(df["workflow_status"]) == {"conformer_refined"}
    assert set(df["calculator_settings"]) == {
        "gas=AIMNet2; freq=GFN2-xTB; solvent_opt=AIMNet2 (gas phase); solvation=CPCM-X"
    }
    assert df["gas_sp_energy_kcal_mol"].astype(float).notna().all()
    assert df["boltzmann_fraction"].astype(float).sum() == pytest.approx(1.0)
    assert list(tmp_path.rglob("aimnet2sp_run.log"))
