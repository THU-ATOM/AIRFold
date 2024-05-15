"""
New Processing functions (self contained)

Logic of processing files:
1. Read the file into lines
2. Transform lines into List of Seqs
3. Process Seqs
4. Transform Seqs into lines
5. Write lines into file
"""
import os
from pathlib import Path
import string
from copy import deepcopy
from dataclasses import dataclass
import tempfile
from typing import List, Tuple
from loguru import logger

import numpy as np
import matplotlib.pyplot as plt

from lib.constant import TMP_ROOT
from lib.tool.align import align_sequences


LINE_CHANGE = "\n"
CLUSTER_MARK = "#"
COMMENT_MARK = ">"
INPUT_MARK = "input_"
GAP = "-"

HAS_COMMENT_FILES = (".fasta", ".a3m", ".hh3a3m", ".hhba3m", ".hmsa3m", ".jaca3m")
NO_COMMENT_FILES = (".aln", ".hh3aln", ".hhbaln", ".hmsaln", ".jacaln")

ascii_lowercase_table = str.maketrans(dict.fromkeys(string.ascii_lowercase))


@dataclass
class Seq:
    comment: str = COMMENT_MARK
    sequence: str = ""


def add_comment(seq: Seq, comment: str):
    """
    Add comment to a Seq
    """
    if seq.comment == "":
        seq.comment = f"{COMMENT_MARK}no_name"
    seq.comment += f" {comment}"
    return seq


def load_file(path: str) -> List[str]:
    with open(path, "r") as fd:
        lines = fd.readlines()
    return lines


def loads_raw(raw_str):
    lines = raw_str.split(LINE_CHANGE)
    return lines


def lines2seq(lines: List[str], has_comment=True) -> List[Seq]:
    """
    Transform lines into List of Seqs
    """
    seqs = []
    comment = ""
    sequence = ""

    # scan lines one by one
    for i, line in enumerate(lines):
        line_str = line.strip()
        if has_comment:
            # if line is comment, set comment and reset sequence
            if line.startswith(COMMENT_MARK):
                # if sequence and comment are not empty, add seq to seqs
                if sequence != "" and comment != "":
                    seqs.append(Seq(comment=comment, sequence=sequence))
                comment = line_str
                sequence = ""
            else:
                # deal with multiple line sequence
                sequence += line_str
        elif len(line_str) > 0:
            seqs.append(
                Seq(
                    comment=f"{COMMENT_MARK}{INPUT_MARK}_{i}", sequence=line_str
                )
            )

    if sequence != "" and comment != "":
        seqs.append(Seq(comment=comment, sequence=sequence))

    return seqs


def seq2lines(seqs: List[Seq], has_comment=True) -> List[str]:
    """
    Transform Seqs into lines
    """
    lines = []
    for seq in seqs:
        if has_comment:
            lines.append(seq.comment + LINE_CHANGE)
        lines.append(seq.sequence + LINE_CHANGE)
    return lines


def save_file(path, lines):
    with open(path, "w") as fd:
        fd.writelines(lines)


def join_lines(lines: List[str]) -> str:
    return "".join(lines)


def assert_length_eq(
    sequences: List[str],
    primary_sequence: str = None,
    ignore_lower: bool = False,
    silent: bool = False,
):
    """
    Check if the length of sequences are the same
    """
    if primary_sequence is None:
        primary_sequence = sequences[0]

    if ignore_lower:
        sequences = [s.translate(ascii_lowercase_table) for s in sequences]
        primary_sequence = primary_sequence.translate(ascii_lowercase_table)

    is_len_eq = np.array(
        [len(primary_sequence) == len(seq) for seq in sequences]
    )

    is_eq = is_len_eq.all()
    logger.info(f"Length of sequences are equal: {is_len_eq}")
    err_msg = (
        f"Sequences should be the same length, "
        f"but got #{np.argmin(is_len_eq)} ({len(sequences[np.argmin(is_len_eq)])})"
        f"compared with primary sequence ({len(primary_sequence)})."
    )
    if silent:
        return is_eq, err_msg
    else:
        assert is_eq, err_msg


def calc_id_cov(sequences, primary_sequence: str = None):
    """
    Calculate the identity coverage of two sequences

    Args
    ----------
    sequences: List[str], the list of sequences, the first one is the primary
        sequence. All sequences should be the same length without considering
        lower case letters.

    Returns
    ----------
    result: a dict with keys:
        identity: the identity of the sequences to the primary sequence
        coverage: the coverage of the sequences to the primary sequence
        non_gaps: the non-gaps matrix in the sequences
        id_matrix: the identity matrix of the sequences
    """
    if primary_sequence is None:
        primary_sequence = sequences[0]

    sequences = [s.translate(ascii_lowercase_table) for s in sequences]
    assert_length_eq(sequences, primary_sequence)

    seq_array = np.array([list(seq) for seq in sequences])
    id_matrix = seq_array == np.array(list(primary_sequence))
    non_gaps = seq_array != GAP
    identity = id_matrix.mean(axis=-1)
    coverage = non_gaps.mean(axis=-1)
    result = {
        "identity": identity,
        "coverage": coverage,
        "non_gaps": non_gaps,
        "id_matrix": id_matrix,
    }
    return result


class ProteinFile:
    def __init__(self, seqs: List[Seq], name: str = None):
        self.seqs = seqs
        self._original_seqs = deepcopy(seqs)

        self.name = name

        # (identity, coverage) to the primary sequence
        self.id_cov = None
        self._is_id_cov_calculated = False

    def __len__(self):
        return len(self.seqs)

    def __repr__(self) -> str:
        res = self.to_raw(top_n=10)
        if len(self.seqs) > 10:
            res += f"\n... ({len(self.seqs)} total)"
        return res

    @classmethod
    def from_file(cls, path: str, name: str = None):
        if str(path).endswith(HAS_COMMENT_FILES):
            has_comment = True
        elif str(path).endswith(NO_COMMENT_FILES):
            has_comment = False
        else:
            raise ValueError(f"Unsupported file type: {path}")
        lines = load_file(path)
        seqs = lines2seq(lines, has_comment=has_comment)
        if name is None:
            name = Path(path).stem
        return cls(seqs)

    @classmethod
    def from_raw(cls, raw_str: str, has_comment=True):
        lines = loads_raw(raw_str)
        seqs = lines2seq(lines, has_comment=has_comment)
        return cls(seqs)

    @classmethod
    def merge(
        cls,
        pfiles: List["ProteinFile"],
        names: List[str] = None,
        deduplicate=True,
    ):
        seqs = []
        if names is not None:
            assert len(names) == len(pfiles), (
                f"Number of names ({len(names)}) should be equal to "
                f"number of protein files ({len(pfiles)})"
            )
        else:
            names = [None for _ in pfiles]
        for pfile, name in zip(pfiles, names):
            if name is not None:
                seqs.extend(
                    [add_comment(seq, f"src={name}") for seq in pfile.seqs]
                )
            else:
                seqs.extend(pfile.seqs)
        obj = cls(seqs)

        if deduplicate:
            obj.deduplicate()

        return obj

    def save(self, path: str):
        if str(path).endswith(HAS_COMMENT_FILES):
            has_comment = True
        elif str(path).endswith(NO_COMMENT_FILES):
            has_comment = False
        else:
            raise ValueError(f"Unsupported file type: {path}")
        lines = seq2lines(self.seqs, has_comment=has_comment)
        save_file(path, lines)

    def to_raw(self, has_comment=True, top_n=None):
        if top_n is not None:
            seqs = self.seqs[:top_n]
        else:
            seqs = self.seqs
        lines = seq2lines(seqs, has_comment=has_comment)
        return join_lines(lines)

    @property
    def primary_seq(self):
        """
        Return the primary seq
        """
        return self.seqs[0]

    @property
    def primary_sequence(self):
        """
        Return the primary sequence
        """
        return self.seqs[0].sequence

    @property
    def sequences(self):
        return [seq.sequence for seq in self.seqs]

    @property
    def profile(self):
        return {
            "name": self.name,
            "N": len(self.sequences),
            "L": len(self.primary_sequence)
        }

    def reset(self):
        self.seqs = deepcopy(self._original_seqs)
        return self

    def is_length_eq(self, ignore_lower=True, silent=True):
        """Check if all sequences are the same length

        Args
        ----------
        ignore_lower: bool, if True, ignore lower case letters in the sequences
        silent: bool, if True, return (is_len_eq, err_msg)

        Returns
        ----------
        is_len_eq: bool, if True, all sequences are the same length
        err_msg: str, the error message if is_len_eq is False

        Usage
        ----------
        Directly assert:
        >>> pfile.is_length_eq(silent=False)

        Return the result and assert by yourself:
        >>> is_len_eq, err_msg = pfile.is_length_eq(silent=True)
        >>> assert is_len_eq, err_msg
        """
        return assert_length_eq(
            self.sequences,
            self.primary_sequence,
            ignore_lower=ignore_lower,
            silent=silent,
        )

    def rm_lower(self):
        """
        Remove lower case letters
        """
        for seq in self.seqs:
            seq.sequence = seq.sequence.translate(ascii_lowercase_table)
        return self

    def to_fasta(self):
        return self.rm_lower()

    def rm_gaps_and_upper(self):
        """
        Remove gaps
        """
        for seq in self.seqs:
            seq.sequence = seq.sequence.replace("-", "").upper()
        self._is_id_cov_calculated = False
        return self

    def dealign(self):
        return self.rm_gaps_and_upper()

    def base_on_primary(self):
        """
        Modify all sequences based on primary sequence.

        Note:
            The primary sequence is the first sequence in the list. This
            sequence is used as the reference to modify all other sequences.
            First, the gap position of the primary sequence is recorded. Then,
            all other sequences are modified based on the primary sequence.
        """
        self.is_length_eq(silent=False)

        is_primary_gap = [c == GAP for c in self.primary_sequence]
        for seq in self.seqs:
            seq.sequence = "".join(
                [
                    (c.lower() if c != GAP else "") if is_primary_gap[i] else c
                    for i, c in enumerate(seq.sequence)
                ]
            )
        self._is_id_cov_calculated = False
        return self

    def to_a3m(self):
        return self.base_on_primary()

    def align(self, tool="kalign"):
        """
        Align sequences using kalign, clustalo, or probcons
        """
        tmp_in_fd, tmp_in_path = tempfile.mkstemp(dir=TMP_ROOT, suffix=".fasta")
        tmp_out_fd, tmp_out_path = tempfile.mkstemp(
            dir=TMP_ROOT, suffix=".fasta"
        )
        path_in = Path(tmp_in_path)
        path_out = Path(tmp_out_path)
        self.save(path_in)
        align_sequences(path_in, path_out, tool=tool)
        self.seqs = lines2seq(load_file(path_out), has_comment=True)
        os.close(tmp_in_fd)
        os.close(tmp_out_fd)
        path_in.unlink()
        path_out.unlink()
        self._is_id_cov_calculated = False
        return self

    def calc_id_cov(self):
        """
        Calculate identity and coverage
        """
        if not self._is_id_cov_calculated:
            self.id_cov = calc_id_cov(self.sequences)
            self._is_id_cov_calculated = True
        return self.id_cov

    def _map_seqs(self, indices: List[int]):
        """
        Adjusting seqs based on indices.
        """
        self.seqs = [self.seqs[i] for i in indices]
        if self._is_id_cov_calculated:
            self.id_cov = {
                key: self.id_cov[key][indices] for key in self.id_cov
            }
        return self

    def filter(self, min_id=0.0, min_cov=0.0):
        """
        Filter sequences based on identity and coverage
        """
        self.calc_id_cov()

        satisfy_id = (self.id_cov["identity"] >= min_id) & (
            self.id_cov["coverage"] >= min_cov
        )
        indices = np.where(satisfy_id)[0]
        self._map_seqs(indices=indices)
        return self

    def sort(self, key="identity", descending=True):
        """
        Sort sequences based on identity or coverage
        """
        self.calc_id_cov()

        if key == "identity":
            indices = self.id_cov["identity"].argsort()
        elif key == "coverage":
            indices = self.id_cov["coverage"].argsort()
        else:
            raise ValueError(f"Unsupported key: {key}")
        if descending:
            indices = indices[::-1]
        self._map_seqs(indices)
        return self

    def deduplicate(self):
        """
        Remove duplicate sequences
        """
        seen = set()
        indices = []
        for i, seq in enumerate(self.seqs):
            if seq.sequence not in seen:
                seen.add(seq.sequence)
                indices.append(i)

        if len(seen) != len(self.seqs):
            self._is_id_cov_calculated = False

        self._map_seqs(indices)

        return self

    def shuffle(self, fix_primary=True):
        """
        Shuffle sequences
        """
        if fix_primary:
            indices = [0] + list(np.random.permutation(len(self.seqs) - 1) + 1)
        else:
            indices = np.random.permutation(len(self.seqs))
        self._map_seqs(indices)
        return self

    def plot(
        self,
        save_path: str = None,
        dpi: int = 100,
        figsize: Tuple[int, int] = (8, 5),
        title: str = "Sequence coverage",
    ):
        self.calc_id_cov()

        non_gaps = self.id_cov["non_gaps"].astype(np.float)
        non_gaps[non_gaps == 0] = np.nan
        lines = non_gaps * self.id_cov["identity"][:, None]
        lines = lines[::-1]

        plt.figure(figsize=figsize, dpi=dpi)
        plt.title(title)

        plt.imshow(
            lines,
            interpolation="nearest",
            aspect="auto",
            cmap="rainbow_r",
            vmin=0,
            vmax=1,
            origin="lower",
            extent=(0, lines.shape[1], 0, lines.shape[0]),
        )

        # plot coverage line
        plt.plot((np.isnan(lines) == False).sum(0), color="black")

        # set x and y axis
        plt.xlim(0, lines.shape[1])
        plt.ylim(0, lines.shape[0])
        plt.colorbar(label="Sequence identity to query")
        plt.xlabel("Positions")
        plt.ylabel("Sequences")

        if save_path is not None:
            plt.savefig(save_path, dpi=dpi)

        return plt
