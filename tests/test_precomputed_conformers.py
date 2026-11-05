"""CREST-style ensembles can drive PEACE refinement without a KDG search."""

from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from peace.protomer import Protomer
from peace.solvation import (
    ConformerEnergyTerms,
    ConformerWorkflowResult,
    _load_precomputed_conformer_ensemble,
    run_protomer_solvation,
)


def _water_xyz(path: Path) -> str:
    mol = Chem.AddHs(Chem.MolFromSmiles("O"))
    params = AllChem.ETKDGv3()
    params.randomSeed = 1
    assert AllChem.EmbedMolecule(mol, params) >= 0
    block = Chem.MolToXYZBlock(mol)
    path.write_text(block)
    return block


def test_loader_keeps_smiles_atom_order_and_skips_mismatches(tmp_path: Path):
    good = tmp_path / "water.xyz"
    block = _water_xyz(good)
    bad = tmp_path / "bad.xyz"
    bad.write_text(block.replace("O", "N", 1))

    mol_h, conf_ids = _load_precomputed_conformer_ensemble("O", [good, bad])

    assert len(conf_ids) == 1
    assert mol_h.GetNumConformers() == 1
    assert [atom.GetSymbol() for atom in mol_h.GetAtoms()][0] == "O"


def test_precomputed_ensemble_skips_kdg_and_aggregates_energies(tmp_path: Path, monkeypatch):
    first = tmp_path / "a.xyz"
    second = tmp_path / "b.xyz"
    block = _water_xyz(first)
    second.write_text(block)
    calls: list[int] = []

    def fake_workflow(_protomer, mol, *, scratch_dir, **_kwargs):
        calls.append(1)
        scratch_dir.mkdir(parents=True, exist_ok=True)
        xyz_path = scratch_dir / "solvent_opt.xyz"
        xyz_path.write_text(Chem.MolToXYZBlock(mol))
        terms = ConformerEnergyTerms(
            gas_sp_energy_kcal_mol=-100.0,
            freq_electronic_energy_kcal_mol=-100.0,
            solvation_free_energy_kcal_mol=-5.0,
            rrho_contribution_kcal_mol=10.0,
            solution_phase_free_energy_kcal_mol=-95.0,
            workflow_status="ok",
        )
        return ConformerWorkflowResult(terms=terms, opt_xyz_path=xyz_path)

    monkeypatch.setattr("peace.solvation._run_single_conformer_workflow", fake_workflow)
    protomer = Protomer.from_smiles("O")
    result = run_protomer_solvation(
        protomer,
        protomer_id=0,
        scratch_root=tmp_path / "scratch",
        precomputed_xyz_paths=[first, second],
        max_qm_conformers=10,
        keep_scratch=True,
        keep_logs=True,
    )

    assert calls == [1, 1]
    assert result.solution_phase_free_energy_kcal_mol == pytest.approx(-95.0)
    assert protomer.mol is not None
    assert protomer.mol.GetProp("conformer_source") == "precomputed_xyz"
    assert protomer.mol.GetProp("workflow_status") == "conformer_refined"
    # Identical geometries are PRISM-pruned down to one conformer.
    assert int(protomer.mol.GetProp("conformer_qm_count")) == 1
