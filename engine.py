from protomer import Tautomer, Species
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize.rdMolStandardize import TautomerEnumerator

class ChargeEngine:
    """
    Engine for searching for specific acid/base substructures.
    Calculations are performed in here to allow the SMARTS substructures to be cached.
    """
    def __init__(self):
        self.tautomer_enumerator = TautomerEnumerator()
        self.SMARTS_DICT = {
                "strong_basic": 
                    {"groups": ["[#7]", "[#6-]"], "sites": [0, 1]}, # TODO: exclude NH acids.
                "weak_basic":  # TODO: =O (etc) groups? can be C=O, P=O etc.
                    {"groups": [], "sites": []}, 
                "strong_acidic": # non-CH acids, NHx+
                    {"groups": ["[!#6&!#7;!H0]","[#7+;!H0]"], "sites": [0, 0]},
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

    def search_for_tautomers(self, spec: Species):
        """
        Search a species for tautomers, given a reference tautomer (which has a reference protomer).
        """
        ref_mol = spec.tautomers[0].protomers[0].mol
        results = self.tautomer_enumerator.Enumerate(mol = ref_mol)

        screened_smiles = []

        # Find atoms that correspond to forbidden groups
        # TODO: move this code to engine
        for smarts in ["[CX3](=[OX1])O "]:
            smarts_substructure = pass

        for smiles in results.smiles:
            screened_smiles.append(smiles)

        # TODO: exclude carboxylic acid groups and esters and ?
        # see https://github.com/rdkit/rdkit/discussions/6822
        # can analyze if changed atoms from Result correspond to carboxylic acid or ester group
        # can search for atom indices where we have a match for these
        # Exclude also if it falls into an [aromatic - aromatic(OH/NH) - aromatic]

        return screened_smiles