"""Microscopic and macroscopic pKa from PEACE solution-phase free energies.

Macroscopic pKa uses the degeneracy-weighted partition function of each
neighboring charge state, Q = Σ g_i exp(-G_i/RT), with relative g taken from
microscopic n_fwd/n_rev (g_A-/g_AH = n_fwd/n_rev). Microscopic pKa pairs are
the single-proton (de)protonations between those ensembles.

Pair enumeration is intentionally cheap: each microstate is reduced once to a
cached heavy-atom skeleton key plus an H-count vector. Only microstates that
share a skeleton are compared, and the comparison is an O(N_atoms) vector
check (exactly one heavy atom gains/loses a single hydrogen).
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import pandas as pd
from rdkit import Chem

from .calculators.common import DEFAULT_TEMPERATURE_K
from .common import canon_smiles
from .logging_utils import LogLevel, log
from .protomer import Species

# kcal/mol/K; matches Species.assign_boltzmann_microstate_populations
GAS_CONSTANT_KCAL_MOL_K = 0.00198720425864083
ENERGY_PROP = "solution_phase_free_energy_kcal_mol"

PKA_CSV_COLUMNS = [
    "kind",
    "charge_acid",
    "charge_base",
    "acid_smiles",
    "base_smiles",
    "acid_tautomer_id",
    "acid_protomer_id",
    "base_tautomer_id",
    "base_protomer_id",
    "site_atom_idx",
    "site_element",
    "g_acid_kcal_mol",
    "g_base_kcal_mol",
    "g_proton_kcal_mol",
    "delta_g_kcal_mol",
    "pka",
    "pka_stat",
    "n_fwd",
    "n_rev",
    "log10_n_over_m",
    "pair_energy_kcal_mol",
    "pair_rel_energy_kcal_mol",
    "acid_boltzmann_fraction",
    "base_boltzmann_fraction",
    "temperature_k",
    "solvent",
]


@dataclass(frozen=True)
class _SkeletonFeatures:
    """Cached heavy-atom graph features used for micro-pKa pairing."""

    smiles: str
    skeleton_key: str
    skeleton: Chem.Mol
    h_counts: tuple[int, ...]
    atom_indices: tuple[int, ...]
    elements: tuple[str, ...]
    n_hydrogen: int


@dataclass
class _AlignedMicrostate:
    charge: int
    tautomer_id: int
    protomer_id: int
    smiles: str
    energy: float
    boltzmann_fraction: Optional[float]
    mol: Any
    h_counts: tuple[int, ...]
    atom_indices: tuple[int, ...]
    elements: tuple[str, ...]
    ionizable: tuple[bool, ...]
    solvent: str


@dataclass
class MicroPkaRecord:
    charge_acid: int
    charge_base: int
    acid_smiles: str
    base_smiles: str
    acid_tautomer_id: int
    acid_protomer_id: int
    base_tautomer_id: int
    base_protomer_id: int
    site_atom_idx: Optional[int]
    site_element: str
    g_acid: float
    g_base: float
    proton_energy: float
    delta_g: float
    pka: float
    pair_energy: float
    pair_rel_energy: float
    acid_fraction: Optional[float]
    base_fraction: Optional[float]
    n_fwd: int = 1
    n_rev: int = 1
    log10_n_over_m: float = 0.0
    pka_stat: float = 0.0
    acid_mol: Any = None
    base_mol: Any = None
    solvent: str = ""
    temperature_k: float = DEFAULT_TEMPERATURE_K


@dataclass
class MacroPkaRecord:
    charge_acid: int
    charge_base: int
    g_acid: float
    g_base: float
    proton_energy: float
    delta_g: float
    pka: float
    n_acid_microstates: int
    n_base_microstates: int
    n_fwd: int = 1
    n_rev: int = 1
    log10_n_over_m: float = 0.0
    solvent: str = ""
    temperature_k: float = DEFAULT_TEMPERATURE_K


@dataclass
class pkaResult:
    micro: list[MicroPkaRecord] = field(default_factory=list)
    macro: list[MacroPkaRecord] = field(default_factory=list)
    proton_energy: float = 0.0
    temperature_k: float = DEFAULT_TEMPERATURE_K
    solvent: str = ""


def rt_ln10(temperature_k: float) -> float:
    if temperature_k <= 0:
        raise ValueError(f"temperature_k must be > 0 K, got {temperature_k}")
    return GAS_CONSTANT_KCAL_MOL_K * float(temperature_k) * math.log(10.0)


def pka_from_free_energies(
    g_acid: float,
    g_base: float,
    *,
    proton_energy: float = 0.0,
    temperature_k: float = DEFAULT_TEMPERATURE_K,
) -> tuple[float, float]:
    """Return (pKa, DG) for AH <=> A- + H+ with DG = G(A-) + G(H+) - G(AH)."""
    delta_g = float(g_base) + float(proton_energy) - float(g_acid)
    return delta_g / rt_ln10(temperature_k), delta_g


def ensemble_free_energy(
    energies: Iterable[float],
    *,
    temperature_k: float = DEFAULT_TEMPERATURE_K,
    degeneracies: Iterable[float] | None = None,
) -> Optional[float]:
    """Boltzmann-aggregated free energy: G = Gmin - RT ln Σ g_i exp(-(Gi-Gmin)/RT)."""
    numeric = [float(e) for e in energies]
    if not numeric:
        return None
    if degeneracies is None:
        weights = [1.0] * len(numeric)
    else:
        weights = []
        for value in degeneracies:
            weight = float(value)
            if not math.isfinite(weight) or weight <= 0.0:
                weight = 1.0
            weights.append(weight)
        if len(weights) != len(numeric):
            raise ValueError("degeneracies must match energies in length")
    rt = GAS_CONSTANT_KCAL_MOL_K * float(temperature_k)
    g_min = min(numeric)
    log_sum = math.log(
        sum(
            weight * math.exp(-(energy - g_min) / rt)
            for energy, weight in zip(numeric, weights)
        )
    )
    return g_min - rt * log_sum


def neighboring_charge_pairs(charges: Iterable[int]) -> list[tuple[int, int]]:
    """Return (acid_charge, base_charge) pairs with acid = base + 1."""
    present = set(int(c) for c in charges)
    return [(charge + 1, charge) for charge in sorted(present) if (charge + 1) in present]


def _optional_mol_float(mol: Chem.Mol | None, key: str) -> Optional[float]:
    if mol is None or not mol.HasProp(key):
        return None
    try:
        return float(mol.GetProp(key))
    except (TypeError, ValueError):
        return None


def _optional_mol_str(mol: Chem.Mol | None, key: str) -> str:
    if mol is None or not mol.HasProp(key):
        return ""
    return str(mol.GetProp(key)).strip()


def _is_connectivity_mismatch(mol: Chem.Mol | None) -> bool:
    if mol is None or not mol.HasProp("connectivity_mismatch"):
        return False
    return mol.GetProp("connectivity_mismatch").strip().lower() == "true"


def _graph_mol(protomer) -> Any:
    if protomer.input_mol is not None:
        return protomer.input_mol
    return protomer.mol


def _heavy_atom_records(mol: Chem.Mol) -> list[tuple[int, str, int]]:
    records = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        records.append(
            (
                int(atom.GetIdx()),
                atom.GetSymbol(),
                int(atom.GetTotalNumHs(includeNeighbors=True)),
            )
        )
    return records


def _total_hydrogen_count(mol: Chem.Mol) -> int:
    return sum(n_h for _idx, _el, n_h in _heavy_atom_records(mol))


def _stripped_heavy_skeleton(mol: Chem.Mol) -> Chem.Mol | None:
    """Heavy-atom connectivity graph: no hydrogens, charges, or bond orders."""
    if mol is None:
        return None
    try:
        work = Chem.RemoveHs(Chem.Mol(mol), sanitize=False)
    except Exception:
        work = Chem.Mol(mol)
    rw = Chem.RWMol(work)
    for atom in rw.GetAtoms():
        atom.SetFormalCharge(0)
        atom.SetNumExplicitHs(0)
        atom.SetNoImplicit(True)
        atom.SetIsAromatic(False)
        atom.SetIsotope(0)
        atom.SetAtomMapNum(0)
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    for bond in rw.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
        bond.SetStereo(Chem.BondStereo.STEREONONE)
    skeleton = rw.GetMol()
    try:
        Chem.FastFindRings(skeleton)
    except Exception:
        pass
    return skeleton


def _skeleton_key(skeleton: Chem.Mol) -> str:
    try:
        key = Chem.MolToSmiles(skeleton, canonical=True, isomericSmiles=False)
    except Exception:
        key = Chem.MolToSmiles(skeleton, canonical=True)
    if not key:
        atoms = tuple(atom.GetAtomicNum() for atom in skeleton.GetAtoms())
        bonds = tuple(
            sorted(
                (
                    min(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
                    max(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
                )
                for bond in skeleton.GetBonds()
            )
        )
        key = f"fallback:{atoms}:{bonds}"
    return key


def _skeleton_features(mol: Chem.Mol, smiles: str, cache: dict[str, _SkeletonFeatures]) -> _SkeletonFeatures | None:
    if smiles in cache:
        return cache[smiles]
    if mol is None:
        return None
    heavy = _heavy_atom_records(mol)
    if not heavy:
        return None
    skeleton = _stripped_heavy_skeleton(mol)
    if skeleton is None or skeleton.GetNumAtoms() != len(heavy):
        return None
    features = _SkeletonFeatures(
        smiles=smiles,
        skeleton_key=_skeleton_key(skeleton),
        skeleton=skeleton,
        h_counts=tuple(n_h for _idx, _el, n_h in heavy),
        atom_indices=tuple(idx for idx, _el, _n_h in heavy),
        elements=tuple(el for _idx, el, _n_h in heavy),
        n_hydrogen=_total_hydrogen_count(mol),
    )
    cache[smiles] = features
    return features


def _match_to_reference(skeleton: Chem.Mol, reference: Chem.Mol) -> tuple[int, ...] | None:
    """Map reference atom i -> skeleton atom index, or None if graphs differ."""
    if skeleton.GetNumAtoms() != reference.GetNumAtoms():
        return None
    if skeleton.GetNumBonds() != reference.GetNumBonds():
        return None
    match = skeleton.GetSubstructMatch(reference)
    if not match or len(match) != reference.GetNumAtoms():
        match = reference.GetSubstructMatch(skeleton)
        if not match or len(match) != skeleton.GetNumAtoms():
            return None
        inverse = [0] * len(match)
        for query_idx, target_idx in enumerate(match):
            inverse[target_idx] = query_idx
        return tuple(inverse)
    return tuple(int(i) for i in match)


def _align_counts(values: tuple[int, ...], match: tuple[int, ...]) -> tuple[int, ...]:
    aligned = [0] * len(match)
    for ref_idx, skel_idx in enumerate(match):
        aligned[ref_idx] = values[skel_idx]
    return tuple(aligned)


def _align_elements(values: tuple[str, ...], match: tuple[int, ...]) -> tuple[str, ...]:
    aligned = [""] * len(match)
    for ref_idx, skel_idx in enumerate(match):
        aligned[ref_idx] = values[skel_idx]
    return tuple(aligned)


def _align_bools(values: tuple[bool, ...], match: tuple[int, ...]) -> tuple[bool, ...]:
    aligned = [False] * len(match)
    for ref_idx, skel_idx in enumerate(match):
        aligned[ref_idx] = values[skel_idx]
    return tuple(aligned)


def find_symmetry_classes(mol: Chem.Mol | None) -> set[tuple[int, ...]]:
    """Atom-index orbits under the molecular automorphism group.

    Adapted from Greg Landrum (2011): un-uniquified self-substructure matches
    are automorphisms; atoms that map onto each other are equivalent.
    """
    if mol is None or mol.GetNumAtoms() == 0:
        return set()
    equivs: dict[int, set[int]] = defaultdict(set)
    try:
        matches = mol.GetSubstructMatches(mol, uniquify=False)
    except Exception:
        return {tuple([idx]) for idx in range(mol.GetNumAtoms())}
    if not matches:
        return {tuple([idx]) for idx in range(mol.GetNumAtoms())}
    for match in matches:
        for idx1, idx2 in enumerate(match):
            equivs[idx1].add(int(idx2))
    return {tuple(sorted(group)) for group in equivs.values()}


def _is_ionizable_heavy_atom(atom) -> bool:
    """N/O/S that currently bears H or a formal charge."""
    if atom.GetAtomicNum() not in (7, 8, 16):
        return False
    n_h = int(atom.GetTotalNumHs(includeNeighbors=True))
    charge = int(atom.GetFormalCharge())
    return n_h > 0 or charge != 0


def _equivalent_site_count(mol: Chem.Mol | None, atom_idx: int, h_count: int) -> int:
    """Count chemically equivalent (de)ionization sites on one Lewis structure.

    Equivalence uses the molecule as drawn (bond orders and charges kept), so
    COOH is not mixed with a geminal enol, and carboxylate O- is not mixed with
    carbonyl O. Atoms must match element, hydrogen count, and formal charge.
    """
    if mol is None:
        return 1
    try:
        atom = mol.GetAtomWithIdx(int(atom_idx))
    except Exception:
        return 1
    atomic_num = atom.GetAtomicNum()
    charge = int(atom.GetFormalCharge())
    classes = find_symmetry_classes(mol)
    cls = next((c for c in classes if int(atom_idx) in c), (int(atom_idx),))
    n_eq = 0
    for idx in cls:
        try:
            other = mol.GetAtomWithIdx(int(idx))
        except Exception:
            continue
        if other.GetAtomicNum() != atomic_num:
            continue
        if int(other.GetFormalCharge()) != charge:
            continue
        if int(other.GetTotalNumHs(includeNeighbors=True)) != int(h_count):
            continue
        n_eq += 1
    return max(1, n_eq)


def _log10_n_over_m(n: float, m: float) -> float:
    n_val = max(float(n), 1.0)
    m_val = max(float(m), 1.0)
    return math.log10(n_val / m_val)


def _pair_site_counts(
    acid: _AlignedMicrostate,
    base: _AlignedMicrostate,
    aligned_site_idx: int,
) -> tuple[int, int]:
    """n_fwd / n_rev for AH ⇌ A- + H+ from equivalent ionizable groups.

    n_fwd: equivalent acidic sites on AH that deprotonate to this A-
           (read from the more protonated Lewis structure).
    n_rev: equivalent ionized sites on A- that protonate back to this AH.
    """
    if aligned_site_idx < 0 or aligned_site_idx >= len(acid.atom_indices):
        return 1, 1
    n_fwd = _equivalent_site_count(
        acid.mol,
        int(acid.atom_indices[aligned_site_idx]),
        int(acid.h_counts[aligned_site_idx]),
    )
    n_rev = _equivalent_site_count(
        base.mol,
        int(base.atom_indices[aligned_site_idx]),
        int(base.h_counts[aligned_site_idx]),
    )
    return n_fwd, n_rev


def _propagate_transition_degeneracies(
    acid_states: list[tuple],
    base_states: list[tuple],
    pair_micro: list[MicroPkaRecord],
) -> tuple[list[float], list[float]]:
    """Assign relative g from g_A-/g_AH = n_fwd/n_rev on the micro-pair graph.

    Gauge: g = 1 on one node per connected component (acids first). Unpaired
    unique structures keep g = 1. Absolute scale within a component cancels
    in Q(A-)/Q(AH).
    """
    adj: dict[tuple[str, int, int], list[tuple[tuple[str, int, int], float]]] = defaultdict(list)

    def add_edge(
        src: tuple[str, int, int],
        dst: tuple[str, int, int],
        factor: float,
    ) -> None:
        if not math.isfinite(factor) or factor <= 0.0:
            return
        adj[src].append((dst, factor))
        adj[dst].append((src, 1.0 / factor))

    for rec in pair_micro:
        acid_node = ("a", int(rec.acid_tautomer_id), int(rec.acid_protomer_id))
        base_node = ("b", int(rec.base_tautomer_id), int(rec.base_protomer_id))
        add_edge(acid_node, base_node, float(rec.n_fwd) / float(rec.n_rev))

    degeneracy: dict[tuple[str, int, int], float] = {}

    def seed_component(start: tuple[str, int, int]) -> None:
        degeneracy[start] = 1.0
        queue = [start]
        while queue:
            node = queue.pop(0)
            for neighbor, factor in adj[node]:
                predicted = degeneracy[node] * factor
                if neighbor in degeneracy:
                    current = degeneracy[neighbor]
                    if current > 0.0 and predicted > 0.0:
                        if abs(math.log(current / predicted)) > 1e-6:
                            log(
                                f"Inconsistent transition degeneracy for {neighbor}: "
                                f"{current:g} vs {predicted:g}; keeping {current:g}",
                                level=LogLevel.VERBOSE,
                            )
                    continue
                degeneracy[neighbor] = predicted
                queue.append(neighbor)

    acid_nodes = [("a", int(row[0]), int(row[1])) for row in acid_states]
    base_nodes = [("b", int(row[0]), int(row[1])) for row in base_states]
    for node in acid_nodes + base_nodes:
        if node not in degeneracy:
            seed_component(node)

    def weight_for(node: tuple[str, int, int]) -> float:
        value = float(degeneracy.get(node, 1.0))
        if not math.isfinite(value) or value <= 0.0:
            return 1.0
        return value

    acid_weights = [weight_for(("a", int(row[0]), int(row[1]))) for row in acid_states]
    base_weights = [weight_for(("b", int(row[0]), int(row[1]))) for row in base_states]
    return acid_weights, base_weights


def _single_hydrogen_site(
    acid: _AlignedMicrostate,
    base: _AlignedMicrostate,
) -> tuple[int, int, str] | None:
    """Return (aligned_idx, original acid atom index, element) for a single-H pair."""
    if len(acid.h_counts) != len(base.h_counts):
        return None
    site_ref: Optional[int] = None
    for idx, (n_acid, n_base) in enumerate(zip(acid.h_counts, base.h_counts)):
        delta = n_acid - n_base
        if delta == 0:
            continue
        if delta != 1 or site_ref is not None:
            return None
        site_ref = idx
    if site_ref is None:
        return None
    return site_ref, acid.atom_indices[site_ref], acid.elements[site_ref]


def _iter_energy_microstates(
    spec: Species,
    *,
    exclude_connectivity_mismatch: bool,
):
    for taut_idx, tautomer in spec.tautomers.items():
        for prot_idx, protomer in tautomer.protomers.items():
            energy = _optional_mol_float(protomer.mol, ENERGY_PROP)
            if energy is None:
                continue
            if exclude_connectivity_mismatch and _is_connectivity_mismatch(protomer.mol):
                continue
            smiles = canon_smiles(protomer.smiles) or protomer.smiles
            yield taut_idx, prot_idx, protomer, smiles, energy


def compute_pka_results(
    species_by_charge: dict[int, Species],
    *,
    temperature_k: float = DEFAULT_TEMPERATURE_K,
    proton_energy: float = 0.0,
    exclude_connectivity_mismatch: bool = False,
    solvent: str = "",
) -> pkaResult:
    """Compute macro- and micro-pKa for every neighboring charge pair."""
    result = pkaResult(
        proton_energy=float(proton_energy),
        temperature_k=float(temperature_k),
        solvent=solvent,
    )
    pairs = neighboring_charge_pairs(species_by_charge.keys())
    if not pairs:
        return result

    graph_cache: dict[str, _SkeletonFeatures] = {}

    for charge_acid, charge_base in pairs:
        acid_spec = species_by_charge[charge_acid]
        base_spec = species_by_charge[charge_base]
        acid_states = list(
            _iter_energy_microstates(
                acid_spec,
                exclude_connectivity_mismatch=exclude_connectivity_mismatch,
            )
        )
        base_states = list(
            _iter_energy_microstates(
                base_spec,
                exclude_connectivity_mismatch=exclude_connectivity_mismatch,
            )
        )
        acid_energies = [row[4] for row in acid_states]
        base_energies = [row[4] for row in base_states]
        pair_solvent = solvent
        if not pair_solvent:
            for _taut, _prot, protomer, _smi, _e in acid_states + base_states:
                pair_solvent = _optional_mol_str(protomer.mol, "solvent")
                if pair_solvent:
                    break

        grouped: dict[str, dict[str, list[_AlignedMicrostate]]] = {}
        ref_skeletons: dict[str, Chem.Mol] = {}

        def _register(charge: int, taut_idx: int, prot_idx: int, protomer, smiles: str, energy: float) -> None:
            mol = _graph_mol(protomer)
            features = _skeleton_features(mol, smiles, graph_cache)
            if features is None:
                return
            reference = ref_skeletons.setdefault(features.skeleton_key, features.skeleton)
            match = _match_to_reference(features.skeleton, reference)
            if match is None:
                return
            ionizable = tuple(
                _is_ionizable_heavy_atom(mol.GetAtomWithIdx(int(idx)))
                for idx in features.atom_indices
            )
            aligned = _AlignedMicrostate(
                charge=charge,
                tautomer_id=int(taut_idx),
                protomer_id=int(prot_idx),
                smiles=smiles,
                energy=float(energy),
                boltzmann_fraction=_optional_mol_float(protomer.mol, "boltzmann_fraction"),
                mol=mol,
                h_counts=_align_counts(features.h_counts, match),
                atom_indices=_align_counts(features.atom_indices, match),
                elements=_align_elements(features.elements, match),
                ionizable=_align_bools(ionizable, match),
                solvent=_optional_mol_str(protomer.mol, "solvent") or pair_solvent,
            )
            bucket = grouped.setdefault(features.skeleton_key, {"acid": [], "base": []})
            if charge == charge_acid:
                bucket["acid"].append(aligned)
            else:
                bucket["base"].append(aligned)

        for taut_idx, prot_idx, protomer, smiles, energy in acid_states:
            _register(charge_acid, taut_idx, prot_idx, protomer, smiles, energy)
        for taut_idx, prot_idx, protomer, smiles, energy in base_states:
            _register(charge_base, taut_idx, prot_idx, protomer, smiles, energy)

        all_pair_energies = acid_energies + base_energies
        g_ref = min(all_pair_energies) if all_pair_energies else 0.0
        n_compared = 0
        pair_micro: list[MicroPkaRecord] = []
        for bucket in grouped.values():
            acids = bucket["acid"]
            bases = bucket["base"]
            if not acids or not bases:
                continue
            n_compared += len(acids) * len(bases)
            for acid in acids:
                for base in bases:
                    site = _single_hydrogen_site(acid, base)
                    if site is None:
                        continue
                    aligned_site_idx, site_idx, site_element = site
                    n_fwd, n_rev = _pair_site_counts(acid, base, aligned_site_idx)
                    pka, delta_g = pka_from_free_energies(
                        acid.energy,
                        base.energy,
                        proton_energy=proton_energy,
                        temperature_k=temperature_k,
                    )
                    log_nm = _log10_n_over_m(n_fwd, n_rev)
                    pair_energy = min(acid.energy, base.energy)
                    pair_micro.append(
                        MicroPkaRecord(
                            charge_acid=charge_acid,
                            charge_base=charge_base,
                            acid_smiles=acid.smiles,
                            base_smiles=base.smiles,
                            acid_tautomer_id=acid.tautomer_id,
                            acid_protomer_id=acid.protomer_id,
                            base_tautomer_id=base.tautomer_id,
                            base_protomer_id=base.protomer_id,
                            site_atom_idx=site_idx,
                            site_element=site_element,
                            g_acid=acid.energy,
                            g_base=base.energy,
                            proton_energy=float(proton_energy),
                            delta_g=delta_g,
                            pka=pka,
                            pka_stat=pka - log_nm,
                            pair_energy=pair_energy,
                            pair_rel_energy=pair_energy - g_ref,
                            acid_fraction=acid.boltzmann_fraction,
                            base_fraction=base.boltzmann_fraction,
                            n_fwd=n_fwd,
                            n_rev=n_rev,
                            log10_n_over_m=log_nm,
                            acid_mol=acid.mol,
                            base_mol=base.mol,
                            solvent=acid.solvent or base.solvent or pair_solvent,
                            temperature_k=float(temperature_k),
                        )
                    )
        result.micro.extend(pair_micro)

        acid_weights, base_weights = _propagate_transition_degeneracies(
            acid_states, base_states, pair_micro
        )
        g_acid = ensemble_free_energy(
            acid_energies,
            temperature_k=temperature_k,
            degeneracies=acid_weights,
        )
        g_base = ensemble_free_energy(
            base_energies,
            temperature_k=temperature_k,
            degeneracies=base_weights,
        )
        g_acid_raw = ensemble_free_energy(acid_energies, temperature_k=temperature_k)
        g_base_raw = ensemble_free_energy(base_energies, temperature_k=temperature_k)
        n_fwd_macro, n_rev_macro = 1, 1
        if pair_micro:
            dominant = min(
                pair_micro,
                key=lambda rec: (rec.delta_g, rec.pka, rec.acid_smiles, rec.base_smiles),
            )
            n_fwd_macro, n_rev_macro = dominant.n_fwd, dominant.n_rev
        if g_acid is not None and g_base is not None:
            pka, delta_g = pka_from_free_energies(
                g_acid,
                g_base,
                proton_energy=proton_energy,
                temperature_k=temperature_k,
            )
            log_nm_macro = 0.0
            if g_acid_raw is not None and g_base_raw is not None:
                pka_raw, _delta_raw = pka_from_free_energies(
                    g_acid_raw,
                    g_base_raw,
                    proton_energy=proton_energy,
                    temperature_k=temperature_k,
                )
                log_nm_macro = pka_raw - pka
            result.macro.append(
                MacroPkaRecord(
                    charge_acid=charge_acid,
                    charge_base=charge_base,
                    g_acid=g_acid,
                    g_base=g_base,
                    proton_energy=float(proton_energy),
                    delta_g=delta_g,
                    pka=pka,
                    n_acid_microstates=len(acid_energies),
                    n_base_microstates=len(base_energies),
                    n_fwd=n_fwd_macro,
                    n_rev=n_rev_macro,
                    log10_n_over_m=log_nm_macro,
                    solvent=pair_solvent,
                    temperature_k=float(temperature_k),
                )
            )
            log(
                f"Macro-pKa charge {charge_acid:+d} <=> {charge_base:+d}: "
                f"pKa={pka:.4f}  DG={delta_g:.4f} kcal/mol "
                f"Q=Σ g exp(-G/RT) n_fwd={n_fwd_macro} n_rev={n_rev_macro} "
                f"log10(n/m)={log_nm_macro:.4f} "
                f"(n_AH={len(acid_energies)}, n_A-={len(base_energies)})"
            )
        else:
            log(
                f"Skipping macro-pKa for charge {charge_acid:+d} <=> {charge_base:+d}: "
                "missing solution-phase free energies on one or both ensembles",
                level=LogLevel.VERBOSE,
            )

        log(
            f"Micro-pKa charge {charge_acid:+d} <=> {charge_base:+d}: "
            f"{sum(1 for rec in result.micro if rec.charge_acid == charge_acid and rec.charge_base == charge_base)} "
            f"single-proton pair(s) "
            f"(skeleton groups={len(grouped)}, vector comparisons={n_compared})",
            level=LogLevel.VERBOSE,
        )

    result.micro.sort(
        key=lambda rec: (
            rec.charge_base,
            rec.delta_g,
            rec.pka,
            rec.acid_smiles,
            rec.base_smiles,
        )
    )
    if not result.solvent:
        result.solvent = next((rec.solvent for rec in result.macro if rec.solvent), "")
    return result


def filter_micro_pka_records(
    records: list[MicroPkaRecord],
    *,
    filter_type: str = "count",
    filter_value: Optional[float] = None,
) -> list[MicroPkaRecord]:
    """Filter micro-pKa reactions for visualization.

    ``count`` keeps the N lowest-ΔG_rxn pairs (ΔG = G(A-) + G(H+) - G(AH)).
    ``threshold`` keeps pairs whose ΔG_rxn is within ``filter_value`` kcal/mol
    of the lowest ΔG_rxn in that neighboring-charge ensemble.
    """
    if filter_type not in ("count", "threshold"):
        raise ValueError(f"Unknown pKa filter type: {filter_type}")
    if not records:
        return []

    grouped: dict[tuple[int, int], list[MicroPkaRecord]] = {}
    for rec in records:
        grouped.setdefault((rec.charge_acid, rec.charge_base), []).append(rec)

    kept: list[MicroPkaRecord] = []
    for _pair, group in grouped.items():
        ranked = sorted(
            group,
            key=lambda rec: (rec.delta_g, rec.pka, rec.acid_smiles, rec.base_smiles),
        )
        if filter_type == "count":
            n_keep = 10 if filter_value is None else int(filter_value)
            if n_keep <= 0:
                continue
            kept.extend(ranked[:n_keep])
            continue
        cutoff = float(filter_value) if filter_value is not None else 10.0
        min_delta = ranked[0].delta_g
        kept.extend(rec for rec in ranked if rec.delta_g - min_delta <= cutoff)
    kept.sort(
        key=lambda rec: (
            rec.charge_base,
            rec.delta_g,
            rec.pka,
            rec.acid_smiles,
            rec.base_smiles,
        )
    )
    return kept


def pka_results_to_dataframe(result: pkaResult) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rec in result.macro:
        rows.append(
            {
                "kind": "macro",
                "charge_acid": rec.charge_acid,
                "charge_base": rec.charge_base,
                "acid_smiles": "",
                "base_smiles": "",
                "acid_tautomer_id": "",
                "acid_protomer_id": "",
                "base_tautomer_id": "",
                "base_protomer_id": "",
                "site_atom_idx": "",
                "site_element": "",
                "g_acid_kcal_mol": rec.g_acid,
                "g_base_kcal_mol": rec.g_base,
                "g_proton_kcal_mol": rec.proton_energy,
                "delta_g_kcal_mol": rec.delta_g,
                "pka": rec.pka,
                "pka_stat": rec.pka,
                "n_fwd": rec.n_fwd,
                "n_rev": rec.n_rev,
                "log10_n_over_m": rec.log10_n_over_m,
                "pair_energy_kcal_mol": min(rec.g_acid, rec.g_base),
                "pair_rel_energy_kcal_mol": "",
                "acid_boltzmann_fraction": "",
                "base_boltzmann_fraction": "",
                "temperature_k": rec.temperature_k,
                "solvent": rec.solvent,
            }
        )
    for rec in result.micro:
        rows.append(
            {
                "kind": "micro",
                "charge_acid": rec.charge_acid,
                "charge_base": rec.charge_base,
                "acid_smiles": rec.acid_smiles,
                "base_smiles": rec.base_smiles,
                "acid_tautomer_id": rec.acid_tautomer_id,
                "acid_protomer_id": rec.acid_protomer_id,
                "base_tautomer_id": rec.base_tautomer_id,
                "base_protomer_id": rec.base_protomer_id,
                "site_atom_idx": rec.site_atom_idx,
                "site_element": rec.site_element,
                "g_acid_kcal_mol": rec.g_acid,
                "g_base_kcal_mol": rec.g_base,
                "g_proton_kcal_mol": rec.proton_energy,
                "delta_g_kcal_mol": rec.delta_g,
                "pka": rec.pka,
                "pka_stat": rec.pka_stat,
                "n_fwd": rec.n_fwd,
                "n_rev": rec.n_rev,
                "log10_n_over_m": rec.log10_n_over_m,
                "pair_energy_kcal_mol": rec.pair_energy,
                "pair_rel_energy_kcal_mol": rec.pair_rel_energy,
                "acid_boltzmann_fraction": rec.acid_fraction,
                "base_boltzmann_fraction": rec.base_fraction,
                "temperature_k": rec.temperature_k,
                "solvent": rec.solvent,
            }
        )
    if not rows:
        return pd.DataFrame(columns=PKA_CSV_COLUMNS)
    return pd.DataFrame(rows, columns=PKA_CSV_COLUMNS)


def format_pka_report(result: pkaResult) -> str:
    lines = [
        "=== pKa predictions "
        f"(T={result.temperature_k:g} K, G(H+)={result.proton_energy:g} kcal/mol"
        + (f", solvent={result.solvent}" if result.solvent else "")
        + ") ===",
        "",
        "Macroscopic pKa (Q = Σ g exp(-G/RT) ensemble free energies):",
    ]
    if not result.macro:
        lines.append("  (none)")
    for rec in result.macro:
        lines.append(
            f"  charge {rec.charge_acid:+d} <=> {rec.charge_base:+d}: "
            f"pKa={rec.pka:.4f}  "
            f"G(AH)={rec.g_acid:.4f}  G(A-)={rec.g_base:.4f}  "
            f"DG={rec.delta_g:.4f} kcal/mol  "
            f"n_fwd={rec.n_fwd} n_rev={rec.n_rev} "
            f"log10(n/m)={rec.log10_n_over_m:.4f}  "
            f"n={rec.n_acid_microstates}/{rec.n_base_microstates}"
        )
    lines.extend(["", "Microscopic pKa (single-proton tautomer pairs):"])
    if not result.micro:
        lines.append("  (none)")
    for rec in result.micro:
        site = (
            f"{rec.site_element}[{rec.site_atom_idx}]"
            if rec.site_atom_idx is not None
            else "n/a"
        )
        lines.append(
            f"  charge {rec.charge_acid:+d} <=> {rec.charge_base:+d}  "
            f"pKa={rec.pka:.4f}  pKa_stat={rec.pka_stat:.4f}  site={site}  "
            f"n_fwd={rec.n_fwd} n_rev={rec.n_rev} log10(n/m)={rec.log10_n_over_m:.4f}  "
            f"AH={rec.acid_smiles} (taut {rec.acid_tautomer_id}, prot {rec.acid_protomer_id})  "
            f"A-={rec.base_smiles} (taut {rec.base_tautomer_id}, prot {rec.base_protomer_id})"
        )
    return "\n".join(lines)


def resolve_pka_csv_path(output_csv: str | None, output_pka_csv: str | None) -> str:
    if output_pka_csv:
        return output_pka_csv
    from pathlib import Path

    base = Path(output_csv) if output_csv else Path("results.csv")
    return str(base.with_name(f"{base.stem}_pka.csv"))


def resolve_pka_plot_path(
    output_plots: str | None,
    output_pka_plots: str | None,
    output_csv: str | None,
) -> str:
    if output_pka_plots:
        return output_pka_plots
    from pathlib import Path

    if output_plots:
        from .visualization import resolve_plot_save_path

        return str(resolve_plot_save_path(output_plots, "pka"))
    base = Path(output_csv) if output_csv else Path("results.csv")
    return str(base.with_name(f"{base.stem}_pka.png"))
