from rdkit.Chem import AllChem, Mol
from protomer import Protomer, ProtomerCollection
from engine import ChargeEngine
from pathlib import Path
from rdkit.Chem import Draw

if __name__ == "__main__":
    ref_protomer = Protomer("C(=O)CCCCO")
    collec = ProtomerCollection(ref_protomer)
    engine = ChargeEngine()
    test_mol = collec.ref_protomer.mol
    print(engine.search_acidity_centers(collec, "strong_acidic"))
#    im = Draw.MolToImage(test_mol)
#    im.show()
    