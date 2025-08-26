from .protomer import Tautomer, Species
from rdkit.Chem import AllChem, Mol, Atom
from rdkit.Chem.MolStandardize.rdMolStandardize import TautomerEnumerator

import warnings

class ChargeEngine:
    """
    Engine for searching for specific acid/base substructures.
    Calculations are performed in here to allow the SMARTS substructures to be cached.
    """
    def __init__(self):
        self.tautomer_enumerator = TautomerEnumerator()
        self.SMARTS_DICT = {
                "strong_basic": 
                    {"groups": ["[#7+0]", "[#6-]"], "sites": [0, 1]}, # TODO: exclude NH acids.
                "weak_basic":  # TODO: =O (etc) groups? can be C=O, P=O etc.
                    {"groups": [], "sites": []}, 
                "strong_acidic": # Acid-type groups (e.g., -ate acids, NH+ acids)
                    {"groups": ["[#6+0,#16+0](=O+0)[OX2H]", "[#7+;!H0]"], "sites": [2, 0]},
                "weak_acidic": # TODO: CH/NH acids. C=CCC=C (aromatic or non),C(=O)CC(=O), N#[CH], N2O-CHn. See IUPAC dataset for examples.
                    {"groups": [], "sites": []} 
                }
        
        for acidity_type, values in self.SMARTS_DICT.items():
            self.SMARTS_DICT[acidity_type]["cached_mols"] = [AllChem.MolFromSmarts(x) for x in values['groups']]

    def search_ionization_centers(self, taut: Tautomer, search_type: str) -> list[int]:
        """ Given a Tautomer, returns a list of atom indices corresponding to the query acidity/basicity"""
        if search_type not in self.SMARTS_DICT.keys():
            raise ValueError(f"Search type must be in: {self.SMARTS_DICT.keys()}")
        
        smarts_collection = self.SMARTS_DICT[search_type]
        return taut.find_ionization_sites(smarts_collection["cached_mols"], smarts_collection["sites"])

    def search_for_tautomers(self, spec: Species) -> list[str]:
        """
        Search a species for tautomers, given a reference tautomer (which has a reference protomer).
        """
        ref_mol = spec.tautomers[0].protomers[0].mol
        results = self.tautomer_enumerator.Enumerate(mol = ref_mol)

        # Find atoms that correspond to forbidden groups
        # TODO: move this code to engine
        # see https://github.com/rdkit/rdkit/discussions/6822
        # Exclude also if it falls into an [aromatic - aromatic(OH/NH) - aromatic]

        atom_ids = []
        for smarts in ["[CX3](=[OX1])O "]:
            smarts_substructure = AllChem.MolFromSmarts(smarts)
            matches = ref_mol.GetSubstructMatches(smarts_substructure)
            for match in matches:
                atom_ids.extend(match[0:2]) # only take the C, =O groups

        atom_ids = list(set(atom_ids))
        candidate_smiles = [spec.tautomers[0].protomers[0].smiles]

        def check_atom_for_equivalence(atom_1: Atom, atom_2: Atom) -> bool:
            for m in ['GetFormalCharge', 'GetNumImplicitHs', 'GetAtomicNum', 'GetDegree','GetHybridization',
                      'GetExplicitValence']:
                if getattr(atom_1, m)() != getattr(atom_2, m)():
                    return False
            for idx, bond_1 in enumerate(atom_1.GetBonds()):
                bond_2 = atom_2.GetBonds()[idx]
                for n in ['GetBondType']:
                    if getattr(bond_1, n)() != getattr(bond_2, n)():
                        return False
            return True


        ref_mol_atoms = ref_mol.GetAtoms()
        for smiles in results.smiles:
            tautomerized_atoms = []
            analyte_mol = AllChem.MolFromSmiles(smiles)
            for atom in analyte_mol.GetAtoms():
                idx = atom.GetIdx()
                if check_atom_for_equivalence(atom, ref_mol_atoms[idx]) == False:
                    tautomerized_atoms.append(idx)

            if not all([x in tautomerized_atoms for x in atom_ids]) or len(atom_ids) == 0:
                candidate_smiles.append(smiles)
                

        return candidate_smiles