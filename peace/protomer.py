from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import AllChem, Mol
from .common import (
    protonate_at_site,
    deprotonate_at_site,
    extract_matches_from_smarts_collection,
    canon_smiles,
    canonicalize_atom_order,
)

import copy
import itertools
import json
import warnings

import numpy as np
import pandas as pd


def _increment_degeneracy(protomer: "Protomer") -> int:
    """Increment and return the degeneracy count on a protomer mol."""
    if protomer.mol is None:
        return 1
    if protomer.mol.HasProp("degeneracy"):
        new_count = int(protomer.mol.GetProp("degeneracy")) + 1
    else:
        new_count = 2
    protomer.mol.SetIntProp("degeneracy", new_count)
    return new_count


def _sync_alternate_tautomer_ids_prop(protomer: "Protomer") -> None:
    if protomer.mol is None or not protomer.alternate_tautomer_ids:
        return
    protomer.mol.SetProp(
        "alternate_tautomer_ids",
        ",".join(str(taut_id) for taut_id in sorted(set(protomer.alternate_tautomer_ids))),
    )


def _sync_resonance_charge_forms_prop(protomer: "Protomer") -> None:
    if protomer.mol is None or not protomer.resonance_charge_forms:
        return
    protomer.mol.SetProp(
        "resonance_charge_forms",
        json.dumps(protomer.resonance_charge_forms, sort_keys=True),
    )


def _resonance_charge_form_label(
    *,
    tautomer_id: int | None,
    protomer_id: int | None,
) -> str:
    taut_label = f"tautomer_{tautomer_id}" if tautomer_id is not None else "tautomer_unknown"
    prot_label = f"id_{protomer_id}" if protomer_id is not None else "id_unknown"
    return f"{taut_label}_{prot_label}"


def _explicit_h_neutral_connectivity_graph(mol: Mol | None) -> Mol | None:
    """
    Return a graph for resonance-charge duplicate detection.

    Hydrogens are explicit graph nodes, formal charges are ignored, and all bonds
    are treated as single/non-aromatic. The result is intentionally unsanitized so
    RDKit does not reinterpret valence or implicit hydrogens after neutralization.
    """
    if mol is None:
        return None
    graph = Chem.AddHs(Chem.Mol(mol))
    rw = Chem.RWMol(graph)
    for atom in rw.GetAtoms():
        atom.SetFormalCharge(0)
        atom.SetNoImplicit(True)
        atom.SetIsAromatic(False)
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    for bond in rw.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
        bond.SetStereo(Chem.BondStereo.STEREONONE)
    return rw.GetMol()


def _connectivity_graphs_are_isomorphic(left: Mol | None, right: Mol | None) -> bool:
    if left is None or right is None:
        return False
    if left.GetNumAtoms() != right.GetNumAtoms():
        return False
    if left.GetNumBonds() != right.GetNumBonds():
        return False
    return left.HasSubstructMatch(right, useChirality=False) and right.HasSubstructMatch(
        left, useChirality=False
    )


def _record_duplicate_skip(
    canonical_protomer: "Protomer",
    *,
    skipped_tautomer_id: int,
    canonical_tautomer_id: int,
) -> int:
    """Record alternate tautomer membership and bump degeneracy on the canonical protomer."""
    if (
        skipped_tautomer_id != canonical_tautomer_id
        and skipped_tautomer_id not in canonical_protomer.alternate_tautomer_ids
    ):
        canonical_protomer.alternate_tautomer_ids.append(skipped_tautomer_id)
        _sync_alternate_tautomer_ids_prop(canonical_protomer)
    return _increment_degeneracy(canonical_protomer)


def _record_resonance_charge_skip(
    canonical_protomer: "Protomer",
    skipped_protomer: "Protomer",
    *,
    skipped_tautomer_id: int | None,
    skipped_protomer_id: int | None,
    canonical_tautomer_id: int,
) -> int:
    """Record a discarded resonance-charge form on the kept protomer."""
    skipped_smiles = canon_smiles(skipped_protomer.smiles) or skipped_protomer.smiles
    if skipped_smiles:
        canonical_protomer.resonance_charge_forms[skipped_smiles] = _resonance_charge_form_label(
            tautomer_id=skipped_tautomer_id,
            protomer_id=skipped_protomer_id,
        )
        _sync_resonance_charge_forms_prop(canonical_protomer)
    if (
        skipped_tautomer_id is not None
        and skipped_tautomer_id != canonical_tautomer_id
        and skipped_tautomer_id not in canonical_protomer.alternate_tautomer_ids
    ):
        canonical_protomer.alternate_tautomer_ids.append(skipped_tautomer_id)
        _sync_alternate_tautomer_ids_prop(canonical_protomer)
    return _increment_degeneracy(canonical_protomer)


class SpeciesProtomerRegistry:
    """
    Track canonical protomer representatives across all tautomers in a Species.
    Supports tracking of duplicate protomers across a Species, which 
    is useful for deduplication + keeping track of ion pairs.
    """

    def __init__(self) -> None:
        self._canonical: dict[str, tuple[int, int, Protomer]] = {}
        self._resonance_graphs: list[tuple[int, int, Protomer, Mol]] = []
        self.skipped_count = 0
        self.resonance_skipped_count = 0

    def is_duplicate(self, smiles: str) -> bool:
        canonical = canon_smiles(smiles)
        return canonical is not None and canonical in self._canonical

    def canonical_for(self, smiles: str) -> tuple[int, int, Protomer] | None:
        canonical = canon_smiles(smiles)
        if canonical is None:
            return None
        return self._canonical.get(canonical)

    def resonance_for(self, protomer: "Protomer") -> tuple[int, int, Protomer] | None:
        graph = _explicit_h_neutral_connectivity_graph(protomer.mol)
        if graph is None:
            return None
        for tautomer_id, protomer_id, canonical_protomer, canonical_graph in self._resonance_graphs:
            if _connectivity_graphs_are_isomorphic(graph, canonical_graph):
                return tautomer_id, protomer_id, canonical_protomer
        return None

    def register(self, tautomer_id: int, protomer_id: int, protomer: Protomer) -> None:
        canonical = canon_smiles(protomer.smiles)
        if canonical is not None:
            self._canonical[canonical] = (tautomer_id, protomer_id, protomer)
        graph = _explicit_h_neutral_connectivity_graph(protomer.mol)
        if graph is not None:
            self._resonance_graphs.append((tautomer_id, protomer_id, protomer, graph))

    def seed_from_species(self, spec: "Species") -> int:
        """Register existing protomers and remove cross-tautomer duplicates."""
        removed = 0
        for taut_idx in sorted(spec.tautomers.keys()):
            taut = spec.tautomers[taut_idx]
            for prot_idx in list(taut.protomers.keys()):
                protomer = taut.protomers[prot_idx]
                canonical = canon_smiles(protomer.smiles)
                if canonical is None:
                    continue
                existing = self._canonical.get(canonical)
                if existing is not None:
                    canon_taut_idx, canon_prot_idx, canon_protomer = existing
                    degeneracy = _record_duplicate_skip(
                        canon_protomer,
                        skipped_tautomer_id=taut_idx,
                        canonical_tautomer_id=canon_taut_idx,
                    )
                    warnings.warn(
                        f"Skipping duplicate protomer {protomer.smiles} under tautomer {taut_idx}; "
                        f"canonical entry is tautomer {canon_taut_idx} protomer {canon_prot_idx} "
                        f"(degeneracy={degeneracy})."
                    )
                    del taut.protomers[prot_idx]
                    removed += 1
                    self.skipped_count += 1
                    continue
                resonance_existing = self.resonance_for(protomer)
                if resonance_existing is not None:
                    canon_taut_idx, canon_prot_idx, canon_protomer = resonance_existing
                    degeneracy = _record_resonance_charge_skip(
                        canon_protomer,
                        protomer,
                        skipped_tautomer_id=taut_idx,
                        skipped_protomer_id=prot_idx,
                        canonical_tautomer_id=canon_taut_idx,
                    )
                    warnings.warn(
                        f"Skipping resonance-charge duplicate protomer {protomer.smiles} "
                        f"under tautomer {taut_idx}; canonical entry is tautomer "
                        f"{canon_taut_idx} protomer {canon_prot_idx} "
                        f"(degeneracy={degeneracy})."
                    )
                    del taut.protomers[prot_idx]
                    removed += 1
                    self.skipped_count += 1
                    self.resonance_skipped_count += 1
                    continue
                if protomer.mol is not None and not protomer.mol.HasProp("degeneracy"):
                    protomer.mol.SetIntProp("degeneracy", 1)
                self.register(taut_idx, prot_idx, protomer)
        return removed

    def unique_count(self) -> int:
        return len(self._canonical)


class Protomer:
    def __init__(self, smiles: str = "", mol: Mol = None):
        self.smiles = canon_smiles(smiles)
        if mol is not None:
            mol = canonicalize_atom_order(mol)
        self.mol = mol
        # Keep a copy of the pre-optimization/input molecular graph for display/export.
        self.input_mol = copy.deepcopy(mol) if mol is not None else None
        self.ionization_sites = []
        self.alternate_tautomer_ids: list[int] = []
        self.resonance_charge_forms: dict[str, str] = {}
        self.is_zwitterion = self._is_zwitterion_mol(mol) if mol is not None else False

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

    @staticmethod
    def _is_zwitterion_mol(mol: Mol) -> bool:
        """
        Return True if a molecule contains BOTH:
          1) a positively charged heavy atom bearing at least one hydrogen, and
          2) a negatively charged heavy atom.
        """
        has_positive_heavy_atom_h = False
        has_negative_heavy_atom = False

        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 1:
                continue
            formal_charge = atom.GetFormalCharge()
            if formal_charge > 0 and atom.GetTotalNumHs(includeNeighbors=True) > 0:
                has_positive_heavy_atom_h = True
            if formal_charge < 0:
                has_negative_heavy_atom = True
            if has_positive_heavy_atom_h and has_negative_heavy_atom:
                return True

        return False
    
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

    def reference_protomer(self) -> Protomer | None:
        """Return the lowest-index protomer, or None when the tautomer is empty."""
        if not self.protomers:
            return None
        return self.protomers[min(self.protomers.keys())]

    def find_ionization_sites(
        self,
        query_substructs: list[Mol],
        query_sites: list[int],
        protomer: Protomer | None = None,
    ) -> list[int]:
        """
        Takes the base protomer mol and tries to find the acidic or basic sites on it matching query.
        Returns a list of atom indices. 
        """
        sites = []
        seed_protomer = protomer if protomer is not None else self.protomers[0]
        base_mol = copy.deepcopy(seed_protomer.mol)

        sites = extract_matches_from_smarts_collection(base_mol, 
                                                        query_substructs,
                                                        query_sites,
        )

        if len(self.forbidden_atoms) > 0:        
            sites = [x for x in sites if x not in self.forbidden_atoms]
        
        return sites
    
    def generate_protomers_from_seed_protomer(
        self,
        seed_protomer: Protomer,
        acidic_sites: list[int],
        basic_sites: list[int],
        *,
        species_registry: SpeciesProtomerRegistry | None = None,
        tautomer_id: int | None = None,
    ) -> list[Protomer]:
        """
        Enumerate protomers by applying one protonation/deprotonation pair to a
        provided seed protomer.

        Returns:
            List of newly embedded protomers.
        """
        new_protomers = []

        # simultaneously consider all acid-base pairs.
        acid_base_pairs = [r for r in itertools.product(acidic_sites, basic_sites)]
        for acid_base_pair in acid_base_pairs:
            mol = copy.deepcopy(seed_protomer.mol)

            acidic_idx = acid_base_pair[0]
            basic_idx = acid_base_pair[1]
            if acidic_idx == basic_idx:
                continue
            protonate_at_site(mol, basic_idx)
            deprotonate_at_site(mol, acidic_idx)

            new_smiles = canon_smiles(AllChem.MolToSmiles(mol))
            canon_mol, idx_map = canonicalize_atom_order(mol, return_index_map=True)
            new_protomer = Protomer(new_smiles or AllChem.MolToSmiles(canon_mol), canon_mol)
            if new_protomer.smiles != new_smiles and new_smiles is not None:
                warnings.warn(
                    f"Protomer SMILES mismatch after protonation/deprotonation: "
                    f"expected={new_smiles}, actual={new_protomer.smiles}. "
                    "Replacing stored SMILES with actual value."
                )
                # Keep running, but force a smiles value that reflects the transformed mol.
                new_protomer.smiles = new_smiles

            # keep historical ionization-site highlights across iterative generations.
            prior_sites = seed_protomer.ionization_sites if seed_protomer.ionization_sites else []
            touched_sites = prior_sites + [basic_idx, acidic_idx]
            new_protomer.ionization_sites = list(
                dict.fromkeys(idx_map.get(site, site) for site in touched_sites)
            )
            if self.embed_protomer(
                new_protomer,
                species_registry=species_registry,
                tautomer_id=tautomer_id,
            ):
                new_protomers.append(new_protomer)
        return new_protomers

    def generate_protomers_from_base_protomer(self, acidic_sites: list[int], basic_sites: list[int]):
        """
        Takes the ref protomer and enumerates other protomers given possible given the acid/base sites.
        Most often used in combination with find_ionization_sites.

        Args:
            acid_sites: list of acidity centers for mol
            basic_sites: list of basic centers for mol
        """

        self.generate_protomers_from_seed_protomer(self.protomers[0], acidic_sites, basic_sites)


    def generate_uncharged_protomer(self, protomer: Protomer) -> Protomer:
        """
        Given a protomer, finds the uncharged variant as a mol object.
        TODO: get this to work for non-zero charge. 
        """
        
        # TODO: assert number of N[H1,H2,H3]+ groups MINUS the  number of [X-] groups is equal to the overall charge.
        return protomer

    def embed_protomer(
        self,
        protomer: Protomer,
        *,
        species_registry: SpeciesProtomerRegistry | None = None,
        tautomer_id: int | None = None,
    ) -> bool:
        """
        Embeds a protomer to the Tautomer.
        Args:
            protomer: The protomer to add
            idx: the id of the protomer to label.
        Returns:
            True if the protomer was added, False if it was not.
        """
        canonical_smiles = canon_smiles(protomer.smiles)
        if canonical_smiles is None:
            return False
        idx = list(self.protomers.keys())[-1] + 1

        if species_registry is not None:
            existing = species_registry.canonical_for(protomer.smiles)
            if existing is not None:
                canon_taut_idx, canon_prot_idx, canon_protomer = existing
                skipped_tautomer_id = tautomer_id if tautomer_id is not None else -1
                degeneracy = _record_duplicate_skip(
                    canon_protomer,
                    skipped_tautomer_id=skipped_tautomer_id,
                    canonical_tautomer_id=canon_taut_idx,
                )
                species_registry.skipped_count += 1
                if skipped_tautomer_id == canon_taut_idx:
                    warnings.warn(
                        f"Skipping duplicate protomer {protomer.smiles} within tautomer "
                        f"{skipped_tautomer_id} (degeneracy={degeneracy})."
                    )
                else:
                    warnings.warn(
                        f"Skipping duplicate protomer {protomer.smiles} under tautomer "
                        f"{skipped_tautomer_id}; canonical entry is tautomer {canon_taut_idx} "
                        f"protomer {canon_prot_idx} (degeneracy={degeneracy})."
                    )
                return False
            resonance_existing = species_registry.resonance_for(protomer)
            if resonance_existing is not None:
                canon_taut_idx, canon_prot_idx, canon_protomer = resonance_existing
                skipped_tautomer_id = tautomer_id if tautomer_id is not None else None
                degeneracy = _record_resonance_charge_skip(
                    canon_protomer,
                    protomer,
                    skipped_tautomer_id=skipped_tautomer_id,
                    skipped_protomer_id=idx,
                    canonical_tautomer_id=canon_taut_idx,
                )
                species_registry.skipped_count += 1
                species_registry.resonance_skipped_count += 1
                if skipped_tautomer_id == canon_taut_idx:
                    warnings.warn(
                        f"Skipping resonance-charge duplicate protomer {protomer.smiles} "
                        f"within tautomer {skipped_tautomer_id} (degeneracy={degeneracy})."
                    )
                else:
                    warnings.warn(
                        f"Skipping resonance-charge duplicate protomer {protomer.smiles} "
                        f"under tautomer {skipped_tautomer_id}; canonical entry is "
                        f"tautomer {canon_taut_idx} protomer {canon_prot_idx} "
                        f"(degeneracy={degeneracy})."
                    )
                return False

        # Check for isomorphic within this tautomer when no species registry is used.
        existing_smiles = [canon_smiles(p.smiles) for p in self.protomers.values()]
        if any(canonical_smiles == x for x in existing_smiles):
            for existing_protomer in self.protomers.values():
                if canon_smiles(existing_protomer.smiles) == canonical_smiles:
                    degeneracy = _increment_degeneracy(existing_protomer)
                    if tautomer_id is not None:
                        warnings.warn(
                            f"Skipping duplicate protomer {protomer.smiles} within tautomer "
                            f"{tautomer_id} (degeneracy={degeneracy})."
                        )
                    break
            return False

        if protomer.mol is not None:
            protomer.mol.SetIntProp("degeneracy", 1)
        self.protomers[idx] = protomer
        if species_registry is not None and tautomer_id is not None:
            species_registry.register(tautomer_id, idx, protomer)
        return True

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
        idx = list(self.tautomers.keys())[-1] + 1
        self.tautomers[idx] = taut

    def drop_empty_tautomers(self) -> list[int]:
        """Remove tautomers with no protomers after deduplication."""
        removed: list[int] = []
        for taut_idx in list(self.tautomers.keys()):
            if not self.tautomers[taut_idx].protomers:
                removed.append(taut_idx)
                del self.tautomers[taut_idx]
        return removed

    def reindex_protomers(self) -> dict[int, dict[int, int]]:
        """
        Compact kept protomer IDs within each tautomer after pruning.

        Deduplication removes entries from each tautomer's protomer dictionary.
        Reindexing keeps downstream logs, scratch folders, plots, and CSV exports
        aligned with the retained protomer count.
        """
        remapped: dict[int, dict[int, int]] = {}
        for taut_idx, tautomer in self.tautomers.items():
            old_items = sorted(tautomer.protomers.items())
            mapping = {
                old_idx: new_idx
                for new_idx, (old_idx, _protomer) in enumerate(old_items)
                if old_idx != new_idx
            }
            if not mapping:
                continue
            tautomer.protomers = {
                new_idx: protomer
                for new_idx, (_old_idx, protomer) in enumerate(old_items)
            }
            remapped[taut_idx] = mapping
        return remapped
    
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
        energy_prop: str = "solution_phase_free_energy_kcal_mol",
        exclude_connectivity_mismatch: bool = False,
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
            - delta_g_kcal_mol
            - boltzmann_fraction
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
                if (
                    exclude_connectivity_mismatch
                    and protomer.mol.HasProp("connectivity_mismatch")
                    and protomer.mol.GetProp("connectivity_mismatch").lower() == "true"
                ):
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
            return pd.DataFrame(
                columns=[
                    "tautomer_id",
                    "protomer_id",
                    "delta_g_kcal_mol",
                    "boltzmann_fraction",
                ]
            )

        g_ref = min(g_i for _, _, _, g_i in entries)
        reduced = [-(g_i - g_ref) / rt for _, _, _, g_i in entries]
        weights = np.exp(np.array(reduced, dtype=float))
        partition_q = float(np.sum(weights))

        rows = []
        for idx, (taut_idx, prot_idx, protomer, g_i) in enumerate(entries):
            delta_g = g_i - g_ref
            frac = float(weights[idx] / partition_q) if partition_q > 0 else 0.0
            protomer.mol.SetDoubleProp("delta_g_kcal_mol", float(delta_g))
            protomer.mol.SetDoubleProp("boltzmann_fraction", float(frac))
            rows.append(
                {
                    "tautomer_id": taut_idx,
                    "protomer_id": prot_idx,
                    "delta_g_kcal_mol": float(delta_g),
                    "boltzmann_fraction": float(frac),
                }
            )
        return pd.DataFrame(rows)

    def get_f_zwit(self) -> float:
        """
        Return total zwitterion fraction from assigned Boltzmann populations.

        This sums `boltzmann_fraction` over all protomers tagged as zwitterions.
        """
        f_zwit = 0.0
        for tautomer in self.tautomers.values():
            for protomer in tautomer.protomers.values():
                if (
                    protomer.is_zwitterion
                    and protomer.mol is not None
                    and protomer.mol.HasProp("boltzmann_fraction")
                ):
                    f_zwit += float(protomer.mol.GetProp("boltzmann_fraction"))
        return float(f_zwit)
        
    def to_dataframe(self):
        rows = []
        solvation_props = [
            "conformer_energy_kcal_mol",
            "conformer_delta_kcal_mol",
            "conformer_qm_count",
            "conformer_labels",
            "conformer_gas_sp_energy_kcal_mol_list",
            "conformer_gas_sp_energy_xtb_kcal_mol_list",
            "conformer_solvation_free_energy_kcal_mol_list",
            "conformer_rrho_contribution_kcal_mol_list",
            "conformer_solution_phase_free_energy_kcal_mol_list",
            "screening_solution_phase_free_energy_kcal_mol",
            "screening_solvation_free_energy_kcal_mol",
            "screening_gas_sp_energy_kcal_mol",
            "screening_gas_sp_energy_xtb_kcal_mol",
            "screening_rrho_contribution_kcal_mol",
            "screening_delta_kcal_mol",
            "screening_placeholder_solution_phase_free_energy_kcal_mol",
            "solvation_free_energy_kcal_mol",
            "solvation_free_energy_cpcmx_kcal_mol",
            "gas_sp_energy_kcal_mol",
            "gas_sp_energy_xtb_kcal_mol",
            "gas_sp_energy_gxtb_kcal_mol",
            "frequency_contribution_kcal_mol",
            "rrho_contribution_kcal_mol",
            "solution_phase_free_energy_kcal_mol",
            "delta_g_kcal_mol",
            "boltzmann_fraction",
            "workflow_status",
            "workflow_error",
            "optimization_opt_level",
            "optimization_initial_opt_level",
            "optimization_engine",
            "connectivity_mismatch",
            "solvent",
            "degeneracy",
            "alternate_tautomer_ids",
            "resonance_charge_forms",
#            "connectivity_mismatch_error",
        ]
        for taut_idx, tautomer in self.tautomers.items():
            for prot_idx, protomer in tautomer.protomers.items():
                row = {
                    "species_id": self.key,
                    "tautomer_id": taut_idx,
                    "protomer_id": prot_idx,
                    "protomer_smiles": protomer.smiles,
                    "is_zwitterion": bool(protomer.is_zwitterion),
                }
                if protomer.mol is not None:
                    for prop in solvation_props:
                        if protomer.mol.HasProp(prop):
                            row[prop] = protomer.mol.GetProp(prop)
                if (
                    "alternate_tautomer_ids" not in row
                    and protomer.alternate_tautomer_ids
                ):
                    row["alternate_tautomer_ids"] = ",".join(
                        str(taut_id) for taut_id in sorted(set(protomer.alternate_tautomer_ids))
                    )
                if (
                    "resonance_charge_forms" not in row
                    and protomer.resonance_charge_forms
                ):
                    row["resonance_charge_forms"] = json.dumps(
                        protomer.resonance_charge_forms, sort_keys=True
                    )
                rows.append(row)

        return pd.DataFrame(rows)
