from protomer import Protomer, ProtomerCollection
from engine import ChargeEngine
from rdkit.Chem import Draw

if __name__ == "__main__":
#    collec = ProtomerCollection(Protomer.from_smiles("NCC(=O)C(O)CCC(O)N"))
    collec = ProtomerCollection(Protomer.from_smiles("NCC(=O)CCCC(O)CC(O)CC(N)(N)CN"))
    engine = ChargeEngine()
    test_mol = collec.ref_protomer.mol
    acid_sites = engine.search_acidity_centers(collec, "strong_acidic")
    basic_sites = engine.search_acidity_centers(collec, "strong_basic")

    collec.generate_protomers_from_ref(acid_sites, basic_sites)
    print(collec.protomers)