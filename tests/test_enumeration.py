"""Unit tests for tautomer/protomer enumeration, charge seeding, and stereochemistry."""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import AllChem

from peace.common import canon_smiles
from peace.engine import ChargeEngine
from peace.main import (
    _enumerate_species_protomers,
    _make_species,
    _seed_adjacent_charge_species,
)
from peace.protomer import Species

EDTA_SMILES = "O=C(O)CN(CCN(CC(=O)O)CC(=O)O)CC(=O)O"
TETRACYCLINE_SMILES = (
    "C[C@@]1(O)c2cccc(O)c2C(=O)C2=C(O)[C@]3(O)C(=O)C(C(N)=O)=C(O)"
    "[C@@H](N(C)C)[C@@H]3C[C@H]12"
)

EXPECTED_EDTA_BASE_TAUTOMERS_CHARGE_0 = [
    "O=C(O)CN(CCN(CC(=O)O)CC(=O)O)CC(=O)O",
    "O=C(O)CN(C=C(O)O)CCN(C=C(O)O)C=C(O)O",
    "O=C(O)CN(C=C(O)O)CCN(C=C(O)O)CC(=O)O",
    "O=C(O)CN(C=C(O)O)CCN(CC(=O)O)CC(=O)O",
    "O=C(O)CN(CCN(C=C(O)O)C=C(O)O)CC(=O)O",
]

EXPECTED_EDTA_PER_TAUTOMER_PROTOMER_COUNTS: dict[int, dict[int, int]] = {
    -4: {0: 1, 1: 23, 2: 10, 3: 5, 4: 9},
    -3: {0: 2, 1: 30, 2: 15, 3: 12, 4: 14},
    -2: {0: 5, 1: 30, 2: 20, 3: 18, 4: 16},
    -1: {0: 5, 1: 23, 2: 15, 3: 18, 4: 14},
    0: {0: 5, 1: 13, 2: 10, 3: 12, 4: 9},
    1: {0: 2, 1: 5, 2: 3, 3: 5, 4: 4},
    2: {0: 1, 1: 1, 2: 1, 3: 1, 4: 1},
}

EXPECTED_EDTA_TAUTOMER_0_PROTOMERS: dict[int, list[str]] = {
    -4: [
        "O=C([O-])CN(CCN(CC(=O)[O-])CC(=O)[O-])CC(=O)[O-]",
    ],
    -3: [
        "O=C([O-])CN(CCN(CC(=O)[O-])CC(=O)O)CC(=O)[O-]",
        "O=C([O-])CN(CC[NH+](CC(=O)[O-])CC(=O)[O-])CC(=O)[O-]",
    ],
    -2: [
        "O=C([O-])CN(CCN(CC(=O)[O-])CC(=O)O)CC(=O)O",
        "O=C([O-])CN(CCN(CC(=O)O)CC(=O)O)CC(=O)[O-]",
        "O=C([O-])CN(CC[NH+](CC(=O)[O-])CC(=O)[O-])CC(=O)O",
        "O=C([O-])CN(CC[NH+](CC(=O)[O-])CC(=O)O)CC(=O)[O-]",
        "O=C([O-])C[NH+](CC[NH+](CC(=O)[O-])CC(=O)[O-])CC(=O)[O-]",
    ],
    -1: [
        "O=C([O-])CN(CCN(CC(=O)O)CC(=O)O)CC(=O)O",
        "O=C([O-])CN(CC[NH+](CC(=O)[O-])CC(=O)O)CC(=O)O",
        "O=C([O-])C[NH+](CCN(CC(=O)O)CC(=O)O)CC(=O)[O-]",
        "O=C([O-])CN(CC[NH+](CC(=O)O)CC(=O)O)CC(=O)[O-]",
        "O=C([O-])C[NH+](CC[NH+](CC(=O)[O-])CC(=O)O)CC(=O)[O-]",
    ],
    0: [
        "O=C(O)CN(CCN(CC(=O)O)CC(=O)O)CC(=O)O",
        "O=C([O-])C[NH+](CCN(CC(=O)O)CC(=O)O)CC(=O)O",
        "O=C([O-])CN(CC[NH+](CC(=O)O)CC(=O)O)CC(=O)O",
        "O=C([O-])C[NH+](CC[NH+](CC(=O)[O-])CC(=O)O)CC(=O)O",
        "O=C([O-])C[NH+](CC[NH+](CC(=O)O)CC(=O)O)CC(=O)[O-]",
    ],
    1: [
        "O=C(O)CN(CC[NH+](CC(=O)O)CC(=O)O)CC(=O)O",
        "O=C([O-])C[NH+](CC[NH+](CC(=O)O)CC(=O)O)CC(=O)O",
    ],
    2: [
        "O=C(O)C[NH+](CC[NH+](CC(=O)O)CC(=O)O)CC(=O)O",
    ],
}


def _enumerate_charge_range(
    smiles: str,
    *,
    charge_min: int,
    charge_max: int,
) -> dict[int, Species]:
    engine = ChargeEngine()
    seed_spec = _make_species(smiles, engine=engine)
    seed_charge = int(AllChem.GetFormalCharge(seed_spec.tautomers[0].protomers[0].mol))
    _enumerate_species_protomers(seed_spec, engine=engine)

    species_by_charge: dict[int, Species] = {seed_charge: seed_spec}

    current_charge = seed_charge
    current_spec = seed_spec
    while current_charge - 1 >= charge_min:
        target_charge = current_charge - 1
        next_spec = _seed_adjacent_charge_species(
            current_spec,
            engine=engine,
            charge_step=-1,
            site_search_mode=None,
        )
        if next_spec is None:
            break
        _enumerate_species_protomers(
            next_spec,
            engine=engine,
            context="inter_charge",
        )
        species_by_charge[target_charge] = next_spec
        current_spec = next_spec
        current_charge = target_charge

    current_charge = seed_charge
    current_spec = seed_spec
    while current_charge + 1 <= charge_max:
        target_charge = current_charge + 1
        next_spec = _seed_adjacent_charge_species(
            current_spec,
            engine=engine,
            charge_step=1,
            site_search_mode=None,
        )
        if next_spec is None:
            break
        _enumerate_species_protomers(
            next_spec,
            engine=engine,
            context="inter_charge",
        )
        species_by_charge[target_charge] = next_spec
        current_spec = next_spec
        current_charge = target_charge

    return species_by_charge


def test_edta_tautomer_and_protomer_enumeration_all_charge_states():
    """Verify EDTA tautomer and protomer enumeration across charge states -4 to +2."""
    species_by_charge = _enumerate_charge_range(
        EDTA_SMILES,
        charge_min=-4,
        charge_max=3,
    )

    # Upward seeding stops at +2 because both tertiary amines are protonated.
    assert sorted(species_by_charge.keys()) == [-4, -3, -2, -1, 0, 1, 2]

    # Check initial neutral base tautomers on charge 0.
    charge_0_base_smiles = [
        taut.protomers[0].smiles for taut in species_by_charge[0].tautomers.values()
    ]
    assert charge_0_base_smiles == EXPECTED_EDTA_BASE_TAUTOMERS_CHARGE_0

    for charge, expected_counts in EXPECTED_EDTA_PER_TAUTOMER_PROTOMER_COUNTS.items():
        spec = species_by_charge[charge]
        actual_counts = {
            taut_idx: len(taut.protomers)
            for taut_idx, taut in spec.tautomers.items()
        }
        assert actual_counts == expected_counts, f"Mismatch at charge {charge}"

        taut_0_smiles = [p.smiles for p in spec.tautomers[0].protomers.values()]
        assert taut_0_smiles == EXPECTED_EDTA_TAUTOMER_0_PROTOMERS[charge]

        # Every protomer in the species must be unique and have the target formal charge.
        seen_smiles: set[str] = set()
        for taut in spec.tautomers.values():
            for prot in taut.protomers.values():
                assert prot.mol is not None
                assert AllChem.GetFormalCharge(prot.mol) == charge
                canonical = canon_smiles(prot.smiles)
                assert canonical is not None
                assert canonical not in seen_smiles
                seen_smiles.add(canonical)


def test_tetracycline_stereochemistry_preserved_in_tautomers_and_protomers():
    """Verify tetracycline tautomer and protomer enumeration preserves all 5 chiral centers."""
    engine = ChargeEngine()
    ref_mol = Chem.MolFromSmiles(TETRACYCLINE_SMILES)
    assert ref_mol is not None
    AllChem.AssignStereochemistry(ref_mol, cleanIt=True, force=True)
    ref_centers = AllChem.FindMolChiralCenters(
        ref_mol, includeUnassigned=False, useLegacyImplementation=False
    )
    assert len(ref_centers) == 5

    spec = _make_species(TETRACYCLINE_SMILES, engine=engine)
    assert spec.key == "OFVLGDICTFRJMM-WESIUVDSSA-N"
    assert len(spec.tautomers) == 72

    for taut_idx, taut in spec.tautomers.items():
        base_prot = taut.protomers[0]
        assert base_prot.mol is not None
        AllChem.AssignStereochemistry(base_prot.mol, cleanIt=True, force=True)
        centers = AllChem.FindMolChiralCenters(
            base_prot.mol, includeUnassigned=False, useLegacyImplementation=False
        )
        assert len(centers) == 5, (
            f"Tautomer {taut_idx} ({base_prot.smiles}) lost stereocenters: {centers}"
        )
        assert "@" in base_prot.smiles

    # Also verify protomer enumeration on the input tetracycline tautomer preserves stereo.
    prot_spec = _make_species(
        TETRACYCLINE_SMILES,
        engine=engine,
        only_protomer_search=True,
    )
    _enumerate_species_protomers(prot_spec, engine=engine)
    assert len(prot_spec.tautomers) == 1
    assert len(prot_spec.tautomers[0].protomers) == 22

    for prot_idx, prot in prot_spec.tautomers[0].protomers.items():
        assert prot.mol is not None
        assert AllChem.GetFormalCharge(prot.mol) == 0
        AllChem.AssignStereochemistry(prot.mol, cleanIt=True, force=True)
        centers = AllChem.FindMolChiralCenters(
            prot.mol, includeUnassigned=False, useLegacyImplementation=False
        )
        assert len(centers) == 5, (
            f"Protomer {prot_idx} ({prot.smiles}) lost stereocenters: {centers}"
        )
