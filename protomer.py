from rdkit.Chem import AllChem, Mol
import warnings

class Protomer:
    # This module should be compatible with any charge object.
    def __init__(self, smiles: str):
        self.smiles = smiles
        mol = AllChem.MolFromSmiles(smiles)
#        [atom.SetAtomMapNum(atom.GetIdx()+1) for atom in mol.GetAtoms()]
        self.mol = mol
    
class ProtomerCollection:
    def __init__(self, ref_protomer : Protomer):
        """
        Species should be of the neutral uncharged form OR a zwitterionic form. 
        Should keep the uncharged form in a separate collection to the zwitterionic forms. 
        Must be instantiated with an uncharged protomer to provide reference state.
        """
        self.protomers = {}
        self.forbidden_atoms = []

        if AllChem.GetFormalCharge(ref_protomer.mol) == 0:
            self.ref_protomer = self.generate_uncharged_protomer(ref_protomer)
        else:
            self.ref_protomer = ref_protomer
            # TODO: get the protonated and deprotonated groups and EXCLUDE those
            self.forbidden_atoms = [] # TODO 
            

#        self.basic_sites = self.find_basic_sites(ref_protomer)
#        self.acidic_sites = self.find_acidic_sites(ref_protomer)

    def extract_matches_from_smarts_collection(self, ref_mol: Mol, groups: list, sites: list):
        """
        Given a reference mol and a list of groups and group of acidity centers corresponding to those groups,
        returns the matching atom indices matching those substructures.
        """
        matching_sites = []

        for idx, substruct in enumerate(groups):
            matches = ref_mol.GetSubstructMatch(substruct)
            if len(matches) == 1:
                matches = [matches]
            for match in matches:
                site = sites[idx]
                atom_match = match[site]
            matching_sites.append(atom_match)

        return matching_sites

    def find_ionization_sites(self, query_substructs: list, query_sites: list):

        sites = []
        ref_mol = self.ref_protomer.mol

        sites = self.extract_matches_from_smarts_collection(ref_mol, 
                                                            query_substructs,
                                                            query_sites,
        )

        if len(self.forbidden_atoms) > 0:        
            sites = [x for x in sites if x not in self.forbidden_atoms]
        
        return sites

    def generate_uncharged_protomer(self, protomer: Protomer) -> Protomer:
        """
        Given a protomer, finds the uncharged variant as a mol object.
        TODO: get this to work for non-zero charge
        """
        
        # TODO: assert number of N[H1,H2,H3]+ groups MINUS the  number of [X-] groups is equal to the overall charge.
        # If they aren't equal, raise an error.
        return protomer   # TODO

    def add_protomer(self, protomer: Protomer):
        # TODO: link protomer to specific ID. Add to self.protomers. 
        # Make self-consistent with uncharged protomer
        pass

    def search_for_protomers(self):
        """
        Using the base protomer, search and enumerate a list of protomers.
        """
        pass
        # TODO: return not only the protomers, but the acid sites that were modified.
        # Either here or separately, pass a new unique ID for this as well.


def protomer_collection_from_smiles(smiles: str):
    # helper function for generating protomer collection from SMILES
    protomer = Protomer(smiles = smiles)
    return ProtomerCollection(ref_protomer = protomer)