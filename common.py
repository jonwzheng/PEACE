from rdkit.Chem import Mol

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