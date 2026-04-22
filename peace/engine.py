from .protomer import Tautomer, Species
from rdkit.Chem import AllChem, Mol, Atom, ValenceType
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
                "strong_acidic": # Acid-type groups (e.g., -ate acids, NH+, acidic carbons/nitrogens [(CH/C-OH or NH/N-OH) connected to C#N, or to nitrate, or C=O group)]
                    {"groups": ["[#6,#15,#16](=O)[OX2H]", "[#7+;!H0]", "[#6,#7;!H0]C#N", "[#6,#7;!H0][N+](=O)[O-]", "[#6,#7;H!0][#6,#7;H0]=O", "N#[C;H1]"], "sites": [2, 0, 0, 0, 0, 1]},
                "weak_acidic": # -OH, -NH, -SH acids TODO: C=CCC=C (aromatic or non), N#[CH], N2O-CHn. See IUPAC dataset for examples.
                    {"groups": ["[#7,#8,#16;!H0]"], "sites": [0]} 
                }
        
        for acidity_type, values in self.SMARTS_DICT.items():
            self.SMARTS_DICT[acidity_type]["cached_mols"] = [AllChem.MolFromSmarts(x) for x in values['groups']]

    def search_ionization_centers(self, taut: Tautomer, search_type: str) -> list[int]:
        """ Given a Tautomer, returns a list of atom indices corresponding to the query acidity/basicity"""
        if search_type == "acidic":
            smarts_collection = self.SMARTS_DICT["strong_acidic"]
            collection = taut.find_ionization_sites(smarts_collection["cached_mols"], smarts_collection["sites"])
            if len(collection) == 0:
                warnings.warn(f"No strong acidic sites found for {taut.protomers[0].smiles}, searching for weak acidic sites.")
                smarts_collection = self.SMARTS_DICT["weak_acidic"]
                collection = taut.find_ionization_sites(smarts_collection["cached_mols"], smarts_collection["sites"])
            return collection
        elif search_type == "basic":
            smarts_collection = self.SMARTS_DICT["strong_basic"]
            collection = taut.find_ionization_sites(smarts_collection["cached_mols"], smarts_collection["sites"])
            if len(collection) == 0:
                warnings.warn(f"No strong basic sites found for {taut.protomers[0].smiles}, searching for weak basic sites.")
                smarts_collection = self.SMARTS_DICT["weak_basic"]
                collection = taut.find_ionization_sites(smarts_collection["cached_mols"], smarts_collection["sites"])
            return collection
        elif search_type not in self.SMARTS_DICT.keys():
            raise ValueError(f"Search type must be in: {self.SMARTS_DICT.keys()}, or 'acidic' or 'basic'")
        
        smarts_collection = self.SMARTS_DICT[search_type]
        return taut.find_ionization_sites(smarts_collection["cached_mols"], smarts_collection["sites"])

    def search_for_tautomers(self, spec: Species) -> list[str]:
        """
        Search a species for tautomers, given a reference tautomer (which has a reference protomer).
        """
        ref_mol = spec.tautomers[0].protomers[0].mol
        results = self.tautomer_enumerator.Enumerate(mol = ref_mol)

        # TODO: Find atoms that correspond to forbidden groups
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
            for m in ['GetFormalCharge', 'GetNumImplicitHs', 'GetAtomicNum', 'GetDegree','GetHybridization']:
                if getattr(atom_1, m)() != getattr(atom_2, m)():
                    return False
                if getattr(atom_1, 'GetValence')(ValenceType.EXPLICIT) != getattr(atom_2, 'GetValence')(ValenceType.EXPLICIT):
                    return False
            for idx, bond_1 in enumerate(atom_1.GetBonds()):
                bond_2 = atom_2.GetBonds()[idx]
                for n in ['GetBondType']:
                    if getattr(bond_1, n)() != getattr(bond_2, n)():
                        return False
            return True


        ref_mol_atoms = ref_mol.GetAtoms()

        # exclude duplicate tautomers
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