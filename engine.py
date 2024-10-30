from protomer import ProtomerCollection
from rdkit.Chem import AllChem

class ChargeEngine:
    """
    Engine for searching for specific acid/base substructures.
    Calculations are performed in here to allow the SMARTS substructures to be cached.
    """
    def __init__(self):
        self.SMARTS_DICT = {
                "strong_basic": 
                    {"groups": ["[#7]"], "sites": [0]}, # TODO: exclude NH acids
                "weak_basic":  # TODO: =O (etc) groups? can be C=O, P=O etc.
                    {"groups": [], "sites": []}, 
                "strong_acidic": # non-CH acids
                    {"groups": ["[!#6&!#7;!H0]"], "sites": [0]},
                "weak_acidic": # TODO: CH/NH acids. C=CCC=C (aromatic or non),C(=O)CC(=O), N#[CH], N2O-CHn. See IUPAC dataset for examples.
                    {"groups": [], "sites": []} 
                }
        
        for acidity_type, values in self.SMARTS_DICT.items():
            self.SMARTS_DICT[acidity_type]["cached_mols"] = [AllChem.MolFromSmarts(x) for x in values['groups']]

    def search_acidity_centers(self, pc: ProtomerCollection, search_type: str):
        if search_type not in self.SMARTS_DICT.keys():
            raise ValueError(f"Search type must be in: {self.SMARTS_DICT.keys()}")
        
        smarts_collection = self.SMARTS_DICT[search_type]
        return pc.find_ionization_sites(smarts_collection["cached_mols"], smarts_collection["sites"])

    