from rdkit import Chem
from rdkit.Chem import Mol
from PIL import Image
from rdkit.Chem.rdchem import KekulizeException, AtomKekulizeException

def protonate_at_site(mol : Mol, site : int):
    '''
    Add a proton of a mol object at the provided index. 
    
    Args:
        mol: Mol object
        site: RDKit atom index of the site to be de/protonated.
    '''

    atom = mol.GetAtomWithIdx(site)
    # Use atom-local hydrogen count only (never include neighboring atoms),
    # otherwise basic nitrogens can be over-protonated.
    hcount = atom.GetTotalNumHs(includeNeighbors=False)
    atom.SetFormalCharge(atom.GetFormalCharge() + 1)
    atom.SetNumExplicitHs(max(0, int(hcount) + 1))
    atom.SetNoImplicit(True)
    atom.UpdatePropertyCache(False)


def deprotonate_at_site(mol : Mol, site : int):
    '''
    Remove a proton of a mol object at the provided index. 
    Args:
        mol: Mol object
        site: RDKit atom index of the site to be de/protonated.
    '''

    atom = mol.GetAtomWithIdx(site)
    # Use atom-local hydrogen count only (never include neighboring atoms).
    hcount = atom.GetTotalNumHs(includeNeighbors=False)
    atom.SetFormalCharge(atom.GetFormalCharge() - 1)
    atom.SetNumExplicitHs(max(0, int(hcount) - 1))
    atom.SetNoImplicit(True)
    atom.UpdatePropertyCache(False)

def extract_matches_from_smarts_collection(query_mol: Mol, groups: list[Mol], sites: list[int]) -> list[int]:
    """
    Given any mol and a list of groups and group of acidity centers corresponding to those substructs,
    returns the matching atom indices of the query mol matching those substructures.
    Args:
        query_mol:  mol object to find the matching indices
        groups: list of substructures (mol)
        sites: list of acidity center indices (of the substructures) where the H atom is attached to
    Returns:
        matching_sites: list of atom indices that match the acidic or basic site
    """
    matching_sites = []

    for idx, substruct in enumerate(groups):
        matches = query_mol.GetSubstructMatches(substruct)
        for match in matches:
            site = sites[idx]
            atom_match = match[site]
            matching_sites.append(atom_match)

    return matching_sites    

def combine_images(imgs: list, buffer: int = 6, mode: str = "vertical") -> Image.Image:
    """Stack images vertically or horizontally into one RGB canvas.
    Adapted, and modified, from Greg Landrum's blog: 
    https://greglandrum.github.io/rdkit-blog/posts/2023-05-26-drawing-options-explained.html
    """

    if not imgs:
        raise ValueError("combine_images requires at least one image.")

    height = 0
    width = 0
    assert mode in ("vertical", "horizontal")

    gap = max(0, buffer)
    for img in imgs:
        if img.mode != "RGB":
            img = img.convert("RGB")
        if mode == "vertical":
            width = max(width, img.width)
            height += img.height
        elif mode == "horizontal":
            height = max(height, img.height)
            width += img.width

    if len(imgs) > 1:
        if mode == "vertical":
            height += gap * (len(imgs) - 1)
        elif mode == "horizontal":
            width += gap * (len(imgs) - 1)

    res = Image.new("RGB", (width, height), "white")
    offset = 0
    for img in imgs:
        if img.mode != "RGB":
            img = img.convert("RGB")
        if mode == "vertical":
            res.paste(img, (0, offset))
            offset += img.height + gap
        elif mode == "horizontal":
            res.paste(img, (offset, 0))
            offset += img.width + gap

    return res


def show_images(imgs: list, buffer: int = 6, mode = "vertical", save_path=None):
    """ 
    Given a list of images, display or save one combined image.
    """
    if not imgs:
        return

    res = combine_images(imgs, buffer=buffer, mode=mode)
    if save_path is not None:
        from pathlib import Path

        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        res.save(out)
    else:
        res.show()

def canonicalize_atom_order(
    mol: Mol | None,
    *,
    return_index_map: bool = False,
) -> Mol | None | tuple[Mol | None, dict[int, int]]:
    """
    Return a copy of ``mol`` with RDKit canonical atom ordering.

    Uses a canonical SMILES round-trip so equivalent SMILES drawings share the
    same atom index map before 3D embedding or QM workflows run.
    """
    if mol is None:
        return (None, {}) if return_index_map else None

    if return_index_map:
        mapped = Chem.Mol(mol)
        for atom_idx, atom in enumerate(mapped.GetAtoms()):
            atom.SetAtomMapNum(atom_idx + 1)
        smiles = Chem.MolToSmiles(mapped, canonical=True)
        canon = Chem.MolFromSmiles(smiles)
        if canon is None:
            return None, {}
        old_to_new: dict[int, int] = {}
        for atom in canon.GetAtoms():
            old_idx = atom.GetAtomMapNum() - 1
            if old_idx >= 0:
                old_to_new[old_idx] = atom.GetIdx()
                atom.SetAtomMapNum(0)
        return canon, old_to_new

    return Chem.MolFromSmiles(Chem.MolToSmiles(mol, canonical=True))


def canon_smiles(smiles: str) -> str:
    """
    Canonicalize a SMILES string.
    """
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    try:
        Chem.SanitizeMol(mol)
    except:
        try:
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES)
        except (KekulizeException, AtomKekulizeException):
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES ^ Chem.SANITIZE_KEKULIZE)
    if mol:
        return Chem.MolToSmiles(mol)
    return None