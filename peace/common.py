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
    atom.SetFormalCharge(atom.GetFormalCharge() + 1)

    hcount = atom.GetTotalNumHs(includeNeighbors=True)
    newcharge = hcount + 1
    atom.SetNumExplicitHs(newcharge)


def deprotonate_at_site(mol : Mol, site : int):
    '''
    Remove a proton of a mol object at the provided index. 
    Args:
        mol: Mol object
        site: RDKit atom index of the site to be de/protonated.
    '''

    atom = mol.GetAtomWithIdx(site)    
    atom.SetFormalCharge(atom.GetFormalCharge() - 1)

    hcount = atom.GetTotalNumHs(includeNeighbors=True)
    newcharge = hcount - 1
    atom.SetNumExplicitHs(newcharge)

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

def show_images(imgs: list, buffer: int = 5, mode = "vertical"):
    """ 
    Given a list of images, return 1 image.
    Adapted from Greg Landrum's blog: 
    https://greglandrum.github.io/rdkit-blog/posts/2023-05-26-drawing-options-explained.html
    """
    height = 0
    width = 0  
    assert mode in ("vertical", "horizontal")

    for img in imgs:
        if mode == "vertical":
            width = max(width, img.width)
            height += img.height

        elif mode == "horizontal":
            height = max(height, img.height)
            width += img.width
    
    if mode == "vertical":
        width += buffer*(len(imgs)-1)
    elif mode == "horizontal":        
        height += buffer*(len(imgs)-1)
    
    res = Image.new("RGBA",(width,height))
    x = 0
    for img in imgs:
        if mode == "vertical":
            res.paste(img,(0,x))
            x += img.height + buffer
        elif mode == "horizontal":        
            res.paste(img,(x,0))
            x += img.width + buffer
    
    res.show()

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