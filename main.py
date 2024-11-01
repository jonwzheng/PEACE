from protomer import Protomer, Tautomer, Species
from engine import ChargeEngine
from rdkit.Chem import Draw

### TODO: graphic art should be a PEACE sign with a tetrahedral carbon svg

if __name__ == "__main__":
#    collec = ProtomerCollection(Protomer.from_smiles("NCC(=O)C(O)CCC(O)N"))
#    prot = Protomer.from_smiles("NCC(=O)CCCC(O)CC(O)CC(N)(N)CN")
#    taut = Tautomer(prot)
    spec = Species.from_smiles("NCC(=O)CCCC(O)CC(O)CC(N)(N)CN")
    engine = ChargeEngine()
    
    list_of_tautomers = engine.search_for_tautomers(spec)
    spec.add_tautomers_from_list_of_smiles(list_of_tautomers)
#    print(spec.tautomers)
    for taut in spec.tautomers:
        acid_sites = engine.search_acidity_centers(taut, "strong_acidic")
        basic_sites = engine.search_acidity_centers(taut, "strong_basic")
        taut.generate_protomers_from_ref(acid_sites, basic_sites)
#        print(taut.protomers)

    # TODO: There appears to be a bug with this not applying to ref_tautomer or ref_protomer, investigate?
    print(spec)
