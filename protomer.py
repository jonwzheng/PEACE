from rdkit.Chem import AllChem, Mol, Draw
from common import protonate_at_site, deprotonate_at_site, extract_matches_from_smarts_collection

import copy
import itertools
import warnings
import numpy as np
import pandas as pd

class Protomer:
    # TODO: This module should be compatible with any charge object.
    def __init__(self, smiles: str = "", mol: Mol = None):
        self.smiles = AllChem.CanonSmiles(smiles)
        self.mol = mol
        self.ionization_sites = []

    def __repr__(self):
        return f"Protomer {self.smiles}"

    @classmethod
    def from_smiles(cls, smiles: str):
        return cls(smiles, AllChem.MolFromSmiles(smiles))

    @classmethod
    def from_mol(cls, mol: Mol):
        return cls(AllChem.MolToSmiles(mol), mol )
    
    def highlight_ionization_sites(self):
        self.mol.__sssAtoms = self.ionization_sites
    
class Tautomer:
    def __init__(self, base_protomer : Protomer = None):
        """
        Species should be of the neutral uncharged form OR a zwitterionic form. 
        Should keep the uncharged form in a separate collection to the zwitterionic forms. 
        Must be instantiated with an uncharged protomer to provide reference state.
        """
        self.protomers = {0: base_protomer}
        self.forbidden_atoms = []

        if AllChem.GetFormalCharge(base_protomer.mol) == 0:
            self.protomers[0] = self.generate_uncharged_protomer(base_protomer)

        self.acidic_sites = []
        self.basic_sites = []

    def __repr__(self):
        return f"Tautomer with {self.protomers}"

    @classmethod
    def from_smiles(cls, smiles: str):
        return cls(Protomer.from_smiles(smiles))

    @classmethod
    def from_mol(cls, mol: Mol):
        return cls(Protomer.from_mol(mol))

    def find_ionization_sites(self, query_substructs: list[Mol], query_sites: list[int]) -> list[int]:
        """
        Takes the base protomer mol and tries to find the acidic or basic sites on it matching query.
        Retursn a list of atom indices. 
        """
        sites = []
        base_mol = copy.deepcopy(self.protomers[0].mol)

        sites = extract_matches_from_smarts_collection(base_mol, 
                                                        query_substructs,
                                                        query_sites,
        )

        if len(self.forbidden_atoms) > 0:        
            sites = [x for x in sites if x not in self.forbidden_atoms]
        
        return sites
    
    def generate_protomers_from_base_protomer(self, acidic_sites: list[int], basic_sites: list[int]):
        """
        Takes the ref protomer and enumerates other protomers given possible given the acid/base sites.
        Most often used in combination with find_ionization_sites.

        Args:
            acid_sites: list of acidity centers for mol
            basic_sites: list of basic centers for mol
        """
        acid_base_pairs = [r for r in itertools.product(acidic_sites, basic_sites)]
        for acid_base_pair in acid_base_pairs:
            mol = copy.deepcopy(self.protomers[0].mol)

            acidic_idx = acid_base_pair[0]
            basic_idx = acid_base_pair[1]
            protonate_at_site(mol, basic_idx)
            deprotonate_at_site(mol, acidic_idx)
            new_protomer = Protomer.from_mol(mol)
            new_protomer.ionization_sites = [basic_idx, acidic_idx]
            self.embed_protomer(new_protomer)


    def generate_uncharged_protomer(self, protomer: Protomer) -> Protomer:
        """
        Given a protomer, finds the uncharged variant as a mol object.
        TODO: get this to work for non-zero charge. TODO is to do this at all.
        """
        
        # TODO: assert number of N[H1,H2,H3]+ groups MINUS the  number of [X-] groups is equal to the overall charge.
        return protomer

    def embed_protomer(self, protomer: Protomer):
        """
        Embeds a protomer to the Tautomer.
        Args:
            protomer: The protomer to add
            idx: the id of the protomer to label.
        """
        #TODO: check if generated protomers have same # of heavy atoms to base
        idx = list(self.protomers.keys())[-1] + 1

        # Check for isomorphic
        existing_smiles = [AllChem.CanonSmiles(p.smiles) for p in self.protomers.values()]
        if any([AllChem.CanonSmiles(protomer.smiles) == x for x in existing_smiles]):
            warnings.warn(f"Protomer {protomer.smiles} not added due to degeneracy.")
            # TODO: include degeneracy for this species. May need to rewrite code, maybe 2 dicts?
        else:
            self.protomers[idx] = protomer

    def generate_protomer_plot(self, n_columns : int):
        """ Plots up to n_columns showing the mol objects. Returns an image."""
        mols = [p.mol for p in self.protomers.values()]
        [p.highlight_ionization_sites() for p in self.protomers.values()]
        n_rows = int(np.ceil(len(mols) / n_columns))
        n_padding = n_rows * n_columns - len(mols)
        mols.extend([None]*n_padding)

        legends = [f"ID: {k} | SMILES: {v.smiles}" for k, v in self.protomers.items()]
        legends.extend([""]*n_padding)

        highlights = [protomer.ionization_sites for protomer in self.protomers.values()]
        highlights.extend([[] for _ in range(n_padding)])
        highlights_array = np.array(highlights, dtype=object)

        mols = np.reshape(mols, (n_rows, n_columns))
        legends = np.reshape(legends, (n_rows, n_columns))
        if highlights_array.size is not 0:
            highlights = highlights_array.reshape(n_rows, n_columns)
        
        img = Draw.MolsMatrixToGridImage(molsMatrix=mols.tolist(), legendsMatrix=legends.tolist(),
                                         subImgSize=(300, 200), highlightAtomListsMatrix=highlights.tolist())
        return img 

class Species:
    """
    Contains enumerations of tautomers for a given compound.
    """
    def __init__(self, base_tautomer : Tautomer = None):
        self.tautomers = {0: base_tautomer}
        self.key = AllChem.MolToInchiKey(base_tautomer.protomers[0].mol)

    def __repr__(self):
        return f"Species with {self.tautomers}"

    @classmethod
    def from_smiles(cls, smiles: str):
        return cls(Tautomer.from_smiles(smiles))

    @classmethod
    def from_mol(cls, mol: Mol):
        return cls(Tautomer.from_mol(mol))

    def embed_tautomer(self, taut: Tautomer):
        # TODO: check that the number of atoms is same as the reference tautomer
        idx = list(self.tautomers.keys())[-1] + 1
        self.tautomers[idx] = taut        
    
    def get_all_smiles(self):
        smiles = []
        for tautomer in self.tautomers.values():
            for protomer in tautomer.protomers.values():
                smiles.append(protomer.smiles)
        return list(set(smiles))

    def embed_tautomers_from_list_of_smiles(self, tautomer_smiles: list[str]):
        """ Embed tautomers from a list of SMILES strings."""
        for smiles in tautomer_smiles:
            if smiles not in self.get_all_smiles():
                tautomer = Tautomer.from_smiles(smiles)
                self.embed_tautomer(tautomer)
            else:
                warnings.warn(f"Tautomer with SMILES {smiles} not added as already embedded in species.")
        
    def to_dataframe(self):
        spec_ids = []
        taut_ids = []
        prot_ids = []
        prot_smiles = []

        for taut_idx, tautomer in self.tautomers.items():
            for prot_idx, protomer in tautomer.protomers.items():
                spec_ids.append(self.key)
                taut_ids.append(taut_idx)
                prot_ids.append(prot_idx)
                prot_smiles.append(protomer.smiles)
                
        return pd.DataFrame.from_dict({"species_id" : spec_ids,
                                       "tautomer_id": taut_ids,
                                       "protomer_id": prot_ids,
                                       "protomer_smiles": prot_smiles,
                                       }
                                    )
    
    def generate_protomer_plot(self, n_columns: int = 5):
        imgs = []
        for tautomer in self.tautomers.values():
            imgs.append(tautomer.generate_protomer_plot(n_columns))
        return imgs