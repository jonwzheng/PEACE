"""End-to-end solvation workflow tests that run real g-xTB/GFN2-xTB/CPCM-X calculations."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

import peace

pytestmark = pytest.mark.xtb

GLYCINE_SMILES = "NCC(=O)O"
AMINONONANOIC_ACID_SMILES = "NCCCCCCCCC(=O)O"
DOPAMINE_SMILES = "NCCc1ccc(O)c(O)c1"
M_NITROPHENYLGLYCINE_SMILES = "O=C(O)CNc1cccc([N+](=O)[O-])c1"


def _f_zwit(result: peace.PeaceResult, charge: int = 0) -> float:
    return result.species_by_charge[charge].get_f_zwit()


@pytest.mark.parametrize(
    ("smiles", "lower", "upper"),
    [
        pytest.param(DOPAMINE_SMILES, 0.9, 1.0, id="dopamine"),
        pytest.param(M_NITROPHENYLGLYCINE_SMILES, 0.0, 0.05, id="m-nitrophenylglycine"),
    ],
)
def test_aqueous_zwitterion_fraction(tmp_path: Path, smiles: str, lower: float, upper: float):
    result = peace.solvate(smiles, scratch_root=tmp_path)
    assert lower <= _f_zwit(result) <= upper


def test_glycine_in_dmso_is_less_zwitterionic_than_in_water(tmp_path: Path):
    result = peace.solvate(GLYCINE_SMILES, solvent="dmso", scratch_root=tmp_path)
    assert set(result.dataframe["solvent"]) == {"dmso"}
    assert 0.9 < _f_zwit(result) < 0.99


@pytest.mark.parametrize("max_conformers", [1, 3, 5])
def test_max_conformers_controls_refined_conformer_count(tmp_path: Path, max_conformers: int):
    result = peace.solvate(
        AMINONONANOIC_ACID_SMILES,
        max_conformers=max_conformers,
        embedded_conformers=50,
        scratch_root=tmp_path,
    )
    df = result.dataframe
    assert set(df["workflow_status"]) == {"conformer_refined"}
    counts = dict(zip(df["protomer_smiles"], df["conformer_qm_count"].astype(int)))
    assert counts[AMINONONANOIC_ACID_SMILES] == max_conformers
    # Refined conformers can collapse onto duplicates, so other protomers may end up below the cap.
    assert all(1 <= n <= max_conformers for n in counts.values())


def _species_dir(scratch_root: Path) -> Path:
    (species_dir,) = scratch_root.glob("species_*")
    return species_dir


def test_keep_logs_and_keep_scratch_preserve_files(tmp_path: Path):
    runs = {
        "plain": {},
        "keep_logs": {"keep_logs": True},
        "keep_scratch": {"keep_scratch": True},
    }
    species_dirs = {}
    for label, options in runs.items():
        peace.solvate(GLYCINE_SMILES, scratch_root=tmp_path / label, **options)
        species_dirs[label] = _species_dir(tmp_path / label) / "charge_0" / "tautomer_0"

    plain = species_dirs["plain"]
    assert (plain / "xyz" / "protomer_0_conformer_0_solvent_opt.xyz").is_file()
    assert not (plain / "log").exists()
    assert not (plain / "protomer_0").exists()
    assert not list(plain.rglob("*_run.log"))

    logs = species_dirs["keep_logs"]
    assert (logs / "log" / "protomer_0_conformer_0" / "xtbopt_run.log").is_file()
    assert (logs / "log" / "protomer_0_conformer_0" / "solvation_sp" / "xtbsolv_run.log").is_file()
    assert (logs / "screening" / "log").is_dir()
    assert not (logs / "protomer_0").exists()

    scratch = species_dirs["keep_scratch"]
    conformer_scratch = scratch / "protomer_0" / "conformer_0"
    assert (conformer_scratch / "xtbopt.xyz").is_file()
    assert (conformer_scratch / "solvation_sp" / "xtb.cosmo").is_file()


def test_custom_xtb_executable_location(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    custom_dir = tmp_path / "custom bin dir"
    custom_dir.mkdir()
    custom_xtb = custom_dir / "my-xtb"
    shutil.copy2(shutil.which("xtb"), custom_xtb)

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    monkeypatch.setenv("PATH", str(empty_dir))

    with pytest.raises(ValueError, match="not found on PATH"):
        peace.solvate(GLYCINE_SMILES, scratch_root=tmp_path / "default")

    result = peace.solvate(
        GLYCINE_SMILES,
        xtb_executable=custom_xtb,
        scratch_root=tmp_path / "custom",
    )
    assert _f_zwit(result) >= 0.99


def test_glycine_pka_microstates(tmp_path: Path):
    result = peace.solvate(
        GLYCINE_SMILES,
        charge_min=-1,
        charge_max=1,
        pka=True,
        scratch_root=tmp_path,
    )
    assert sorted(result.species_by_charge) == [-1, 0, 1]

    microstates: dict[int, set[str]] = {}
    for rec in result.pka_result.micro:
        microstates.setdefault(rec.charge_acid, set()).add(rec.acid_smiles)
        microstates.setdefault(rec.charge_base, set()).add(rec.base_smiles)
    assert microstates == {
        1: {"[NH3+]CC(=O)O"},
        0: {"NCC(=O)O", "[NH3+]CC(=O)[O-]"},
        -1: {"NCC(=O)[O-]"},
    }

    macro = {(rec.charge_acid, rec.charge_base): rec for rec in result.pka_result.macro}
    assert set(macro) == {(1, 0), (0, -1)}
    assert (macro[(1, 0)].n_acid_microstates, macro[(1, 0)].n_base_microstates) == (1, 2)
    assert (macro[(0, -1)].n_acid_microstates, macro[(0, -1)].n_base_microstates) == (2, 1)
    assert macro[(1, 0)].pka < macro[(0, -1)].pka


# GFN2-xTB total energies are on a valence-only scale (about -11,200 kcal/mol for glycine),
# whereas g-xTB energies are near the all-electron scale (about -178,500 kcal/mol).
def _is_gfn2_scale(energy_kcal_mol: float) -> bool:
    return -20_000 < energy_kcal_mol < 0


@pytest.mark.parametrize(
    ("gas_calculator", "freq_calculator"),
    [
        ("gfn2", "gxtb"),
        ("gxtb", "gfn2"),
    ],
)
def test_mix_and_match_xtb_calculators(tmp_path: Path, gas_calculator: str, freq_calculator: str):
    result = peace.solvate(
        GLYCINE_SMILES,
        gas_calculator=gas_calculator,
        freq_calculator=freq_calculator,
        scratch_root=tmp_path,
    )
    df = result.dataframe
    assert set(df["workflow_status"]) == {"conformer_refined"}
    for gas_energy, freq_energy in zip(df["gas_sp_energy_kcal_mol"], df["freq_electronic_energy_kcal_mol"]):
        assert _is_gfn2_scale(float(gas_energy)) == (gas_calculator == "gfn2")
        assert _is_gfn2_scale(float(freq_energy)) == (freq_calculator == "gfn2")
    assert _f_zwit(result) >= 0.99
