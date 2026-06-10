from .protomer import Tautomer, Species
from rdkit.Chem import AllChem, Atom, ValenceType
from rdkit.Chem.MolStandardize.rdMolStandardize import TautomerEnumerator

import warnings

class ChargeEngine:
    """
    Engine for searching for specific acid/base substructures.
    Calculations are performed in here to allow the SMARTS substructures to be cached.
    """
    def __init__(self):
        self.tautomer_enumerator = TautomerEnumerator()
        self.SMARTS_DICT = {
                "strong_basic": 
                    {"groups": ["[#7+0]", "[#6-]"], "sites": [0, 0]}, # TODO: exclude NH acids.
                "weak_basic": 
                    {"groups": ["[#6,#15,#16](=[O+0])[!OX2H+0]"], "sites": [1]}, 
                "strong_acidic": # Acid-type groups (e.g., -ate acids, NH+, acidic carbons/nitrogens [(CH/C-OH or NH/N-OH) connected to C#N, or to nitrate, or C=O group)]
                    {"groups": ["[#6,#15,#16](=O)[OX2H+0]", "[#7+;!H0+0]", "[#6,#7;!H0+0]C#N", "[#6,#7;!H0+0][N+](=O)[O-]", "[#6,#7;H!0+0][#6,#7;H0]=O", "N#[C;H1+0]"], "sites": [2, 0, 0, 0, 0, 1]},
                "weak_acidic": # -OH, -NH, -SH acids TODO: C=CCC=C (aromatic or non), N#[CH], N2O-CHn. See IUPAC dataset for examples.
                    {"groups": ["[#7,#8,#16;!H0+0]"], "sites": [0]} 
                }
        
        for acidity_type, values in self.SMARTS_DICT.items():
            self.SMARTS_DICT[acidity_type]["cached_mols"] = [AllChem.MolFromSmarts(x) for x in values['groups']]

    def search_ionization_centers(
        self,
        taut: Tautomer,
        search_type: str,
        *,
        site_search_mode: str = "default",
    ) -> list[int]:
        """Given a Tautomer, returns atom indices for the query acidity/basicity."""
        if site_search_mode not in ("default", "strong", "all", "none"):
            raise ValueError(
                "site_search_mode must be 'default', 'strong', 'all', or 'none'"
            )
        if site_search_mode == "none":
            return []

        if search_type == "acidic":
            strong_key, weak_key = "strong_acidic", "weak_acidic"
        elif search_type == "basic":
            strong_key, weak_key = "strong_basic", "weak_basic"
        elif search_type not in self.SMARTS_DICT:
            raise ValueError(
                f"Search type must be in: {self.SMARTS_DICT.keys()}, or 'acidic' or 'basic'"
            )
        else:
            smarts_collection = self.SMARTS_DICT[search_type]
            return taut.find_ionization_sites(
                smarts_collection["cached_mols"], smarts_collection["sites"]
            )

        strong_collection = self.SMARTS_DICT[strong_key]
        strong_sites = taut.find_ionization_sites(
            strong_collection["cached_mols"], strong_collection["sites"]
        )

        if site_search_mode == "strong":
            return strong_sites

        if site_search_mode == "default":
            if strong_sites:
                return strong_sites
            weak_collection = self.SMARTS_DICT[weak_key]
            return taut.find_ionization_sites(
                weak_collection["cached_mols"], weak_collection["sites"]
            )

        if site_search_mode == "all":
            weak_collection = self.SMARTS_DICT[weak_key]
            weak_sites = taut.find_ionization_sites(
                weak_collection["cached_mols"], weak_collection["sites"]
            )
            return list(dict.fromkeys(strong_sites + weak_sites))

    def search_for_tautomers(self, spec: Species) -> list[str]:
        """
        Search a species for tautomers, given a reference tautomer (which has a reference protomer).
        """
        ref_mol = spec.tautomers[0].protomers[0].mol
        AllChem.AssignStereochemistry(ref_mol, cleanIt=True, force=True)
        results = self.tautomer_enumerator.Enumerate(mol = ref_mol)

        # TODO: Find atoms that correspond to forbidden groups
        # see https://github.com/rdkit/rdkit/discussions/6822
        # Exclude also if it falls into an [aromatic - aromatic(OH/NH) - aromatic]

        atom_ids = []
        for smarts in ["[CX3](=[OX1])O "]:
            smarts_substructure = AllChem.MolFromSmarts(smarts)
            matches = ref_mol.GetSubstructMatches(smarts_substructure)
            for match in matches:
                atom_ids.extend(match[0:2]) # only take the C, =O groups

        atom_ids = list(set(atom_ids))
        candidate_smiles = [spec.tautomers[0].protomers[0].smiles]

        # Keep only tautomers that preserve assigned tetrahedral stereochemistry
        # from the reference structure.
        ref_chiral_centers = dict(
            AllChem.FindMolChiralCenters(
                ref_mol, includeUnassigned=False, useLegacyImplementation=False
            )
        )

        def check_atom_for_equivalence(atom_1: Atom, atom_2: Atom) -> bool:
            for m in ['GetFormalCharge', 'GetNumImplicitHs', 'GetAtomicNum', 'GetDegree','GetHybridization']:
                if getattr(atom_1, m)() != getattr(atom_2, m)():
                    return False
                if getattr(atom_1, 'GetValence')(ValenceType.EXPLICIT) != getattr(atom_2, 'GetValence')(ValenceType.EXPLICIT):
                    return False
            for idx, bond_1 in enumerate(atom_1.GetBonds()):
                bond_2 = atom_2.GetBonds()[idx]
                for n in ['GetBondType']:
                    if getattr(bond_1, n)() != getattr(bond_2, n)():
                        return False
            return True


        ref_mol_atoms = ref_mol.GetAtoms()

        # exclude duplicate tautomers
        for smiles in results.smiles:
            tautomerized_atoms = []
            analyte_mol = AllChem.MolFromSmiles(smiles)
            AllChem.AssignStereochemistry(analyte_mol, cleanIt=True, force=True)
            analyte_chiral_centers = dict(
                AllChem.FindMolChiralCenters(
                    analyte_mol, includeUnassigned=False, useLegacyImplementation=False
                )
            )

            # reject if  chiral center flips or disappears compared to reference
            if any(analyte_chiral_centers.get(idx) != label for idx, label in ref_chiral_centers.items()):
                continue

            for atom in analyte_mol.GetAtoms():
                idx = atom.GetIdx()
                if check_atom_for_equivalence(atom, ref_mol_atoms[idx]) == False:
                    tautomerized_atoms.append(idx)

            if not all([x in tautomerized_atoms for x in atom_ids]) or len(atom_ids) == 0:
                candidate_smiles.append(smiles)
                
        return candidate_smiles