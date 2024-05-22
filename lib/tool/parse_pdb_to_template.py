import traceback
from Bio.PDB import PDBParser, Selection
from Bio.Seq import Seq
import numpy as np
import torch
from Bio import pairwise2
import lib.tool.alphafold.common.residue_constants as residue_constants

from typing import Dict, Union


def Path_to_PDB(
    pdb_path: str,
    model_num: int = 0,
) -> object:
    parser = PDBParser()
    structure = parser.get_structure("PHA-L", pdb_path)[model_num]

    return structure


def sequence_to_aatype(sequence: str) -> torch.LongTensor:
    alphabet = np.array(list("ARNDCQEGHILKMFPSTWYVX-"), dtype="|S1").view(np.uint8)
    aatype = np.array(list(sequence), dtype="|S1").view(np.uint8)
    for i in range(alphabet.shape[0]):
        aatype[aatype == alphabet[i]] = i

    aatype[aatype > 20] = 20

    return torch.LongTensor(aatype)


def PDB_to_atom37(
    sequence: str,
    structure: object,
    name: str,
) -> Dict[str, Union[torch.FloatTensor, torch.LongTensor, torch.BoolTensor]]:
    """
    Args:
        sequence: sequence string
        structure: PDB structure
    Returns:
        Dictionary containing:
            * 'all_atom_positions': atom37 representation of all atom coordinates.
            * 'all_atom_mask': atom37 representation of mask on all atom coordinates.
    """

    # Load PDB file

    def align(sequence1, sequence2, trim=False):
        """
        Args:
            sequence1: fasta
            sequence2: pdb_sequence
        Requirements:
            len(s1) >= len(s2), is `trim` is False.
        """
        sequence1 = sequence1.upper()
        sequence2 = sequence2.upper()
        if trim and len(sequence1) < len(sequence2):
            sequence2 = sequence2[: len(sequence1)]
        native_sequence = Seq(sequence1)
        pdb_sequence = Seq(sequence2)
        alignments = pairwise2.align.globalms(
            native_sequence, pdb_sequence, 5, -4, -3, -0.1
        )
        align = None
        for align in alignments:
            align = align
            break
        assert align is not None
        seqA = str(align.seqA)
        seqB = str(align.seqB)
        if "-" in seqA:
            print("- in seqA, removing accordingly")
            seqB = "".join([b for a, b in zip(seqA, seqB) if a != "-"])
        assert len(seqB) == len(sequence1)
        return seqB

    atom37_pos_from_pdb = []
    sequence_from_pdb = []

    for res in Selection.unfold_entities(structure, "R"):
        res_coords = torch.full((37, 3), float("-inf"), dtype=torch.float32)
        resname = res.get_resname()
        resname = resname if resname != "MSE" else "MET"
        if not resname in residue_constants.restype_name_to_atom14_names.keys():
            continue
        atom14_indices = residue_constants.restype_name_to_atom14_names[resname]
        sequence_from_pdb.append(residue_constants.restype_3to1[resname])
        for atom in res.get_atoms():
            atom_name = atom.get_name()
            if atom_name not in atom14_indices:
                continue
            res_coords[residue_constants.atom_order[atom_name]] = torch.from_numpy(
                atom.get_coord()
            )

        atom37_pos_from_pdb.append(res_coords)

    sequence_from_pdb = "".join(sequence_from_pdb)

    try:
        sequence_aligned = align(sequence, sequence_from_pdb, trim=True)
    except AssertionError as e:
        print("Alignment error.")
        print(traceback.print_exc())
        return None
    # Get coordinates for the aligned sequence

    atom37_pos = torch.full([len(sequence_aligned), 37, 3], float("-inf"))
    j = 0
    for i, a in enumerate(sequence_aligned):
        if a != "-":
            atom37_pos[i] = atom37_pos_from_pdb[j]
            j += 1
    atom37_mask = ~(atom37_pos.isinf().sum(-1).bool())
    atom37_pos[~atom37_mask] = 0.0

    atom37_pos = torch.FloatTensor(atom37_pos)
    atom37_mask = torch.BoolTensor(atom37_mask)

    return {
        "all_atom_positions": atom37_pos,
        "all_atom_mask": atom37_mask,
    }


def parse_pdb_to_template(sequence, pdb_path, template_name="cusotmize_A"):
    structure = Path_to_PDB(pdb_path)

    sequence = sequence.replace("U", "C")
    sequence = sequence.replace("B", "D")
    sequence = sequence.replace("Z", "E")

    aatype = sequence_to_aatype(sequence)

    """
        'aatype' : [NUM_RES]
    """
    protein = {"aatype": aatype}

    """
        'all_atom_mask': [NUM_RES, None],
        'all_atom_positions': [NUM_RES, None, None],
        
    """
    atom_37 = PDB_to_atom37(sequence, structure, "fuck")
    protein.update(atom_37)

    protein["all_atom_mask"][:, 5:] = 0.0
    protein["all_atom_positions"][:, 5:, :] = 0.0

    import torch.nn.functional as F

    template_dict = {
        "template_aatype": F.one_hot(protein["aatype"], num_classes=22)
        .unsqueeze(0)
        .numpy(),
        "template_sequence": np.array([sequence], dtype=object),
        "template_all_atom_masks": protein["all_atom_mask"].unsqueeze(0).numpy(),
        "template_all_atom_positions": protein["all_atom_positions"]
        .unsqueeze(0)
        .numpy(),
        "template_domain_names": np.array([template_name], dtype=object),
        "template_sum_probs": np.array([[1.0]], dtype=np.float32),
    }
    return template_dict
