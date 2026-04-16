from peace.protomer import Species
from peace.engine import ChargeEngine
from peace.common import show_images
from rdkit.Chem import Draw

if __name__ == "__main__":
#    spec = Species.from_smiles("NCC(=O)CCCC(O)CC(O)CC(N)(N)CN")
#    spec = Species.from_smiles("C(CCN)C[C@@H](C(=O)O)N")
#    spec = Species.from_smiles("OC(=O)CN(CCN(CC(O)=O)CC(O)=O)CC(O)=O")
#    spec = Species.from_smiles("Oc1cc(N)ccc1")
    spec = Species.from_smiles("NCCCC(=O)CCC(=O)O")
    engine = ChargeEngine()
    
    tautomers = engine.search_for_tautomers(spec)
    print(tautomers) # just a list of SMILES strings. Could be provided manually if desired.

    spec.embed_tautomers_from_list_of_smiles(tautomers)
    print(spec)

    # generate protomers for each tautomer based on available acid/base sites.
    for taut in spec.tautomers.values():
        acid_sites = engine.search_ionization_centers(taut, "strong_acidic")
        basic_sites = engine.search_ionization_centers(taut, "strong_basic")
        taut.generate_protomers_from_base_protomer(acid_sites, basic_sites)

    imgs = spec.generate_protomer_plot(n_columns = 5)
    show_images(imgs, mode="vertical")
    print(spec.to_dataframe())