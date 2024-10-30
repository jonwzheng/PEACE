from rdkit.Chem import AllChem, Mol
from protomer import Protomer, ProtomerCollection
from pathlib import Path
from rdkit.Chem import Draw

if __name__ == "__main__":
    ref_protomer = Protomer("CCO")
    collec = ProtomerCollection(ref_protomer)
    test_mol = collec.ref_protomer.mol
    print(collec.find_ionization_sites("strong_acidic"))
#    im = Draw.MolToImage(test_mol)
#    im.show()
    