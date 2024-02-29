from Bio.PDB.PDBIO import Select
from Bio.PDB import PDBParser, PDBIO



class PDBCutSelect(Select):
    """Select everything for PDB output (for use as a base class).

    Default selection (everything) during writing - can be used as base class
    to implement selective output. This selects which entities will be written out.
    """

    def __init__(self, cut_head=0, cut_tail=0) -> None:
        self.cut_head = cut_head
        self.cut_tail = cut_tail
        
    def __repr__(self):
        """Represent the output as a string for debugging."""
        return f"<Select {self.cut_head} to -{self.cut_tail} >"

    def accept_residue(self, residue):
        """Overload this to reject residues for output."""
        idx = residue.get_id()[1]
        n_residue = len(residue.get_parent())
        if idx < self.cut_head or idx > n_residue - self.cut_tail:
            return 0
        return 1


def cut_pdb(path_in, path_out, cut_head, cut_tail):
    """Cut the pdb file to cut_head and cut_tail."""
    parser = PDBParser()
    structure = parser.get_structure("pdb", path_in)
    io = PDBIO()
    io.set_structure(structure)
    io.save(path_out, select=PDBCutSelect(cut_head, cut_tail))