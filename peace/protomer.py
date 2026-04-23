from rdkit.Chem import AllChem, Mol, Draw
from .common import protonate_at_site, deprotonate_at_site, extract_matches_from_smarts_collection, canon_smiles

import copy
import itertools
import warnings
import numpy as np
import pandas as pd

class Protomer:
    # TODO: This module should be compatible with any charge object.

    def __init__(self, smiles: str = "", mol: Mol = None):
        self.smiles = canon_smiles(smiles)
        self.mol = mol
        # Keep a copy of the pre-optimization/input molecular graph for display/export.
        self.input_mol = copy.deepcopy(mol) if mol is not None else None
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
        Returns a list of atom indices. 
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

        # simultaneously consider all acid-base pairs
        # basically this gives us zwitterions
        acid_base_pairs = [r for r in itertools.product(acidic_sites, basic_sites)] 
        for acid_base_pair in acid_base_pairs:
            mol = copy.deepcopy(self.protomers[0].mol)

            acidic_idx = acid_base_pair[0]
            basic_idx = acid_base_pair[1]
            protonate_at_site(mol, basic_idx)
            deprotonate_at_site(mol, acidic_idx)

            new_smiles = canon_smiles(AllChem.MolToSmiles(mol))
            new_protomer = Protomer.from_mol(mol)
            if new_protomer.smiles != new_smiles:
                warnings.warn(
                    f"Protomer SMILES mismatch after protonation/deprotonation: "
                    f"expected={new_smiles}, actual={new_protomer.smiles}. "
                    "Replacing stored SMILES with actual value."
                )
                # Keep running, but force a smiles value that reflects the transformed mol.
                new_protomer.smiles = new_smiles

            new_protomer.ionization_sites = [basic_idx, acidic_idx]
            self.embed_protomer(new_protomer)


    def generate_uncharged_protomer(self, protomer: Protomer) -> Protomer:
        """
        Given a protomer, finds the uncharged variant as a mol object.
        TODO: get this to work for non-zero charge. 
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
        existing_smiles = [canon_smiles(p.smiles) for p in self.protomers.values()]
        if any([canon_smiles(protomer.smiles) == x for x in existing_smiles]):
            warnings.warn(f"Protomer {protomer.smiles} not added due to degeneracy.")
        else:
            self.protomers[idx] = protomer

    def generate_protomer_plot(self, n_columns : int):
        """ Plots up to n_columns showing the mol objects. Returns an image."""
        # Prefer the original input molecular graph for display because optimized xyz
        # geometries can lack bond assignment.
        mols = []
        for p in self.protomers.values():
            display_mol = p.input_mol if p.input_mol is not None else p.mol
            if display_mol is not None:
                display_mol = copy.deepcopy(display_mol)
                display_mol.__sssAtoms = p.ionization_sites
            mols.append(display_mol)
        n_rows = int(np.ceil(len(mols) / n_columns))
        n_padding = n_rows * n_columns - len(mols)
        mols.extend([None]*n_padding)

        # show also the microstate populations if available
        legends = []
        for k, v in self.protomers.items():
            if v.mol.HasProp('peace_boltzmann_fraction'):
                f_i = f"f_i: {float(v.mol.GetProp('peace_boltzmann_fraction')):.4f}"
            else:
                f_i = ""
            legends.append(f"ID: {k} | SMILES: {v.smiles}\n {f_i}")
        legends.extend([""]*n_padding)

        highlights = [protomer.ionization_sites for protomer in self.protomers.values()]
        highlights.extend([[] for _ in range(n_padding)])
        highlights_array = np.array(highlights, dtype=object)

        mols = np.reshape(mols, (n_rows, n_columns))
        legends = np.reshape(legends, (n_rows, n_columns))
        if highlights_array.size != 0:
            highlights = highlights_array.reshape(n_rows, n_columns).tolist()            
            img = Draw.MolsMatrixToGridImage(molsMatrix=mols.tolist(), legendsMatrix=legends.tolist(),
                                            subImgSize=(300, 200), highlightAtomListsMatrix=highlights)
        else:
            img = Draw.MolsMatrixToGridImage(molsMatrix=mols.tolist(), legendsMatrix=legends.tolist(),
                                            subImgSize=(300, 200))
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
        """ Embeds tautomers from a list of SMILES strings."""
        for smiles in tautomer_smiles:
            if smiles not in self.get_all_smiles():
                tautomer = Tautomer.from_smiles(smiles)
                self.embed_tautomer(tautomer)

    def assign_boltzmann_microstate_populations(
        self,
        *,
        temperature_k: float = 298.15,
        energy_prop: str = "peace_solution_phase_free_energy_kcal_mol",
    ) -> pd.DataFrame:
        """
        Compute and assign Boltzmann populations across all protomers in all tautomers.

        Uses:
            DGi = Gi - Gref
            Q = sum_i exp(-DGi/RT)
            fi = exp(-DGi/RT) / Q

        Energies are read from `energy_prop` on each protomer mol.
        The lowest-energy protomer across ALL tautomers is used as reference.
        Assigned properties:
            - peace_delta_g_kcal_mol
            - peace_boltzmann_fraction
        """
        if temperature_k <= 0:
            raise ValueError("temperature_k must be > 0.")

        # kcal/mol/K
        GAS_CONSTANT_KCAL = 0.00198720425864083
        rt = GAS_CONSTANT_KCAL * float(temperature_k)

        entries = []
        for taut_idx, tautomer in self.tautomers.items():
            for prot_idx, protomer in tautomer.protomers.items():
                if protomer.mol is None or not protomer.mol.HasProp(energy_prop):
                    continue
                try:
                    g_i = float(protomer.mol.GetProp(energy_prop))
                except ValueError:
                    warnings.warn(
                        f"Could not parse {energy_prop} for tautomer_id={taut_idx}, protomer_id={prot_idx}."
                    )
                    continue
                entries.append((taut_idx, prot_idx, protomer, g_i))

        if len(entries) == 0:
            warnings.warn(
                f"No protomers found with property '{energy_prop}'. "
                "Boltzmann populations were not assigned."
            )

        g_ref = min(g_i for _, _, _, g_i in entries)
        reduced = [-(g_i - g_ref) / rt for _, _, _, g_i in entries]
        weights = np.exp(np.array(reduced, dtype=float))
        partition_q = float(np.sum(weights))

        rows = []
        for idx, (taut_idx, prot_idx, protomer, g_i) in enumerate(entries):
            delta_g = g_i - g_ref
            frac = float(weights[idx] / partition_q) if partition_q > 0 else 0.0
            protomer.mol.SetDoubleProp("peace_delta_g_kcal_mol", float(delta_g))
            protomer.mol.SetDoubleProp("peace_boltzmann_fraction", float(frac))

    def to_dataframe(self):
        rows = []
        solvation_props = [
            "peace_conformer_energy_kcal_mol",
            "peace_solvation_free_energy_kcal_mol",
            "peace_gas_sp_energy_kcal_mol",
            "peace_frequency_contribution_kcal_mol",
            "peace_solution_phase_free_energy_kcal_mol",
            "peace_delta_g_kcal_mol",
            "peace_boltzmann_fraction",
        ]
        for taut_idx, tautomer in self.tautomers.items():
            for prot_idx, protomer in tautomer.protomers.items():
                row = {
                    "species_id": self.key,
                    "tautomer_id": taut_idx,
                    "protomer_id": prot_idx,
                    "protomer_smiles": protomer.smiles,
                }
                if protomer.mol is not None:
                    for prop in solvation_props:
                        if protomer.mol.HasProp(prop):
                            row[prop] = protomer.mol.GetProp(prop)
                rows.append(row)

        return pd.DataFrame(rows)
    
    def generate_protomer_plot(self, n_columns: int = 5):
        imgs = []
        for tautomer in self.tautomers.values():
            imgs.append(tautomer.generate_protomer_plot(n_columns))
        return imgs