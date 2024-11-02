from protomer import Protomer, Tautomer, Species
from engine import ChargeEngine
from common import show_images
from rdkit.Chem import Draw

### TODO: graphic art should be a PEACE sign with a tetrahedral carbon svg

if __name__ == "__main__":
#    spec = Species.from_smiles("NCC(=O)CCCC(O)CC(O)CC(N)(N)CN")
    spec = Species.from_smiles("Oc1cc(N)ccc1")
#    spec = Species.from_smiles("NCC(=O)O")
    engine = ChargeEngine()
    
    list_of_tautomers = engine.search_for_tautomers(spec)
    spec.embed_tautomers_from_list_of_smiles(list_of_tautomers)
    for taut in spec.tautomers.values():
        acid_sites = engine.search_ionization_centers(taut, "strong_acidic")
        basic_sites = engine.search_ionization_centers(taut, "strong_basic")
        taut.generate_protomers_from_base_protomer(acid_sites, basic_sites)

    print(spec)
    imgs = spec.generate_protomer_plot(n_columns = 5)
    show_images(imgs, mode="vertical")
    print(spec.to_dataframe())