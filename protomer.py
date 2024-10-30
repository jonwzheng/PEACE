from rdkit.Chem import AllChem, Mol
from common import protonate_at_site, deprotonate_at_site

import copy
import itertools
import warnings

class Protomer:
    # This module should be compatible with any charge object.
    def __init__(self, smiles: str = "", mol: Mol = None):
        self.smiles = smiles
        self.mol = mol

    def __repr__(self):
        return f"Protomer {self.smiles}"

    @classmethod
    def from_smiles(cls, smiles):
        return cls(smiles, AllChem.MolFromSmiles(smiles))

    @classmethod
    def from_mol(cls, mol):
        return cls(AllChem.MolToSmiles(mol), mol )
    
    
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
            self.forbidden_atoms = [None] # TODO: get the protonated and deprotonated groups and EXCLUDE those

        self.acidic_sites = []
        self.basic_sites = []

    def extract_matches_from_smarts_collection(self, mol: Mol, groups: list[Mol], sites: list[int]) -> list[int]:
        """
        Given any mol and a list of groups and group of acidity centers corresponding to those groups,
        returns the matching atom indices matching those substructures.
        Args:
            mol:  mol object to find the matching indices
            groups: list of substructures (mol)
            sites: list of acidity center indices (of the substructures) where the H atom is attached to
        Returns:
            matching_sites: list of atom indices that match the acidic or basic site
        """
        matching_sites = []

        for idx, substruct in enumerate(groups):
            matches = mol.GetSubstructMatches(substruct)
            for match in matches:
                site = sites[idx]
                atom_match = match[site]
                matching_sites.append(atom_match)

        return matching_sites

    def find_ionization_sites(self, query_substructs: list[Mol], query_sites: list[int]) -> list[int]:
        """
        Takes the ref mol and tries to find the acidic or basic sites on it matching query. 
        """
        sites = []
        ref_mol = copy.deepcopy(self.ref_protomer.mol)

        sites = self.extract_matches_from_smarts_collection(ref_mol, 
                                                            query_substructs,
                                                            query_sites,
        )

        if len(self.forbidden_atoms) > 0:        
            sites = [x for x in sites if x not in self.forbidden_atoms]
        
        return sites
    
    def generate_protomers_from_ref(self, acidic_sites: list[int], basic_sites: list[int]):
        """
        Takes the ref protomer and enumerates other protomers given possible given the acid/base sites.
        Most often used in combination with find_ionization_sites.

        Args:
            acid_sites: list of acidity centers for mol
            basic_sites: list of basic centers for mol
        """
        acid_base_pairs = [r for r in itertools.product(acidic_sites, basic_sites)]
        for acid_base_pair in acid_base_pairs:
            mol = copy.deepcopy(self.ref_protomer.mol)

            acidic_idx = acid_base_pair[0]
            basic_idx = acid_base_pair[1]
            protonate_at_site(mol, basic_idx)
            deprotonate_at_site(mol, acidic_idx)
            new_protomer = Protomer.from_mol(mol)
            self.add_protomer(new_protomer)


    def generate_uncharged_protomer(self, protomer: Protomer) -> Protomer:
        """
        Given a protomer, finds the uncharged variant as a mol object.
        TODO: get this to work for non-zero charge. TODO is to do this at all.
        """
        
        # TODO: assert number of N[H1,H2,H3]+ groups MINUS the  number of [X-] groups is equal to the overall charge.
        return protomer

    def add_protomer(self, protomer: Protomer):
        """
        Adds a protomer to the ProtomerCollection.
        Args:
            protomer: The protomer to add
            idx: the id of the protomer to label.
        """
        #TODO: check if generated protomers are isomorphic compared to existing protomers!

        if len(self.protomers) == 0:
            idx = 0
        else:
            idx = list(self.protomers.keys())[-1] + 1

        # Check for isomorphic
        existing_smiles = [AllChem.CanonSmiles(p.smiles) for p in self.protomers.values()]
        if any([AllChem.CanonSmiles(protomer.smiles) == x for x in existing_smiles]):
            warnings.warn(f"Protomer {protomer.smiles} not added due to degeneracy.")
            # TODO: include degeneracy for this species. May need to rewrite code, maybe 2 dicts?
        else:
            self.protomers[idx] = protomer

    def search_for_protomers(self):
        """
        Using the base protomer, search and enumerate a list of protomers.
        """
        # TODO: return not only the protomers, but the acid sites that were modified.
        # Either here or separately, pass a new unique ID for this as well.
        pass
