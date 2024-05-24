import json
import string
import pickle as pkl
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple, Optional, Union

import jsonlines
from loguru import logger
import numpy as np
from rich import print
from Bio import pairwise2
from Bio.Seq import Seq

ascii_lowercase_table = str.maketrans(dict.fromkeys(string.ascii_lowercase))
amino_acids = [
    "L",
    "A",
    "G",
    "V",
    "S",
    "E",
    "R",
    "T",
    "I",
    "D",
    "P",
    "K",
    "Q",
    "N",
    "F",
    "Y",
    "M",
    "H",
    "W",
    "C",
    "X",
    "B",
    "U",
    "Z",
    "O",
    "-",
]
LINE_CHANGE = "\n"


def read_lines(path, rm_empty_lines=False, strip=False):
    with open(path, "r") as f:
        lines = f.readlines()
    if strip:
        lines = [line.strip() for line in lines]
    if rm_empty_lines:
        lines = [line for line in lines if len(line.strip()) > 0]
    return lines


def write_lines(path, lines):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        lines = [
            line if line.endswith(LINE_CHANGE) else f"{line}{LINE_CHANGE}"
            for line in lines
        ]
        f.writelines(lines)


def read_jsonlines(path):
    with jsonlines.open(path) as reader:
        samples = list(reader)
    return samples


def write_jsonlines(path, samples):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(path, "w") as writer:
        writer.write_all(samples)


def read_json(path):
    with open(path) as f:
        data = json.load(f)
    return data


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def read_text_file(path):
    with open(path, "r") as fd:
        data = fd.read()
    return data


def write_text_file(plaintext, path):
    with open(path, "w") as fd:
        fd.write(plaintext)


def read_pickle(path):
    with open(path, "rb") as f:
        data = pkl.load(f)
    return data


def write_pickle(obj, path):
    with open(path, "wb") as fd:
        pkl.dump(obj=obj, file=fd)


# Processing functions


def process_file(path_in, path_out=None, lines_funcs=[]):
    lines = read_lines(path_in)
    if not isinstance(lines_funcs, Iterable):
        lines_funcs = [lines_funcs]
    for lines_func in lines_funcs:
        lines = lines_func(lines)
    if path_out is not None:
        write_lines(path_out, lines)
    return lines


def is_comment_line(line):
    return len(line) == 0 or line[0] == "#" or line[0] == ">"


def build_list_from_dir(directory, accept_files=[".a3m", ".fasta", ".aln", ".pdb"]):
    names = []
    for path in sorted(Path(directory).expanduser().glob("*")):
        if path.suffix in accept_files:
            names.append(path.stem)
    names = list(set(names))
    samples = [{"name": name} for name in sorted(names)]
    return samples


def sample2fasta(sample, key="sequence"):
    lines = []
    lines.append(f">{sample['name']}")
    lines.append(f"{sample[key]}")
    return lines


def sample2aln(sample, key="sequence"):
    lines = []
    lines.append(f"{sample[key]}")
    return lines


INPUT_MARK = "input_"


def aln2seq(lines):
    assert len(lines) > 0, "empty lines."
    lines = [f">{INPUT_MARK}0\n", lines[0]]
    return lines


def aln2fasta(lines):
    new_lines = []
    for i, line in enumerate(lines):
        new_lines.append(f">{INPUT_MARK}{i}")
        new_lines.append(line.strip())
    return new_lines


def a3m_rm_lines(lines, rm_comment_key=INPUT_MARK):
    """filter out sequence with `>{rm_comment_key}` as start

    The target sequence is reserved.
    """
    new_lines = []
    # the target sequence
    new_lines.extend(lines[:2])
    # searching sequences
    for i, line in enumerate(lines):
        if line.startswith(">") and not line.startswith(f">{rm_comment_key}"):
            new_lines.extend(lines[i : i + 2])
    return new_lines


def a3m2lines(lines, rm_lower=False):
    lines = [
        line.strip().translate(ascii_lowercase_table) if rm_lower else line.strip()
        for line in lines
        if not is_comment_line(line)
    ]
    return lines


def a3m2fasta(lines):
    lines = [
        line.strip()
        if is_comment_line(line)
        else line.strip().translate(ascii_lowercase_table)
        for line in lines
    ]
    return lines


def a3m2aln(lines):
    return a3m2lines(lines, rm_lower=True)


def dealign_a3m(lines):
    lines = [
        line.strip()
        if is_comment_line(line.strip())
        else line.strip().translate(str.maketrans(dict.fromkeys(["-"]))).upper()
        for line in lines
    ]
    return lines


def append_linechange(lines):
    lines = [f"{line}{LINE_CHANGE}" for line in lines]
    return lines


def append_number(lines):
    lines = [
        f"{line} LINE={i}" if is_comment_line(line.strip()) else line.strip()
        for i, line in enumerate(lines)
    ]
    return lines


def is_protein(seq):
    return all(c in amino_acids for c in seq)


def get_target(path):
    with open(path, "r") as f:
        for line in f:
            if len(line) != 0 and line[0] != "#" and line[0] != ">":
                return line.strip()
    return None


# ------
def add_gaps_seg(msas, start, end, length):
    for idx in range(len(msas)):
        msas[idx] = "-" * start + msas[idx] + "-" * (length - end - 1)
    return msas


def fasta2list(path, rm_empty_lines=False):
    with open(path, "r") as f:
        lines = f.readlines()
    lines = [l for l in lines if not l.startswith(">")]
    if rm_empty_lines:
        lines = [line for line in lines if len(line.strip()) > 0]
    lines = [line.strip() for line in lines]
    return lines[:50000]


def fasta2aln(fastapath, alnpath):
    lines = fasta2list(fastapath)
    list2aln(alnpath, lines)


def merge_msas(msa_lists):
    # print(f"len {len(msa_lists)}")
    result = set([])
    for index in range(len(msa_lists)):
        result = result.union(msa_lists[index][1:])
    result = [msa_lists[0][0]] + list(result)
    # print(result)
    return result


def list2fasta(path, lines):
    # from a list to fasta file
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        # with open(output_dir, 'w') as f:
        f.writelines([">" + "\n" + line + "\n" for line in lines])


def list2aln(path, lines):
    # from a list object to an aln file
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        # with open(output_dir, 'w') as f:
        f.write("\n".join(lines))


def seq2a3m(msa):
    target = msa[0]
    newseq = [target]
    for seq in msa[1:]:
        temp, seq = align_pair(target, seq)
        # print("aligning...")
        # print(temp)
        # print(seq)
        seq_lower = ""
        for i, char in enumerate(temp):
            if char == "-":
                seq_lower += seq[i].lower()
            else:
                seq_lower += seq[i]
        newseq.append(seq_lower)
    return newseq


# ===================== print


def print_msa(msa):
    n_digit = len(str(len(msa) - 1))
    target = msa[0]
    for i, seq in enumerate(msa):
        if i == 0:
            render_text = (
                f"[bold yellow]{seq}[/bold yellow] " f"[bold cyan]Identity[/bold cyan]"
            )
        else:
            # target, seq = align_pair(msa[0], seq)
            render_text = ""
            for c_t, c_g in zip(target, seq):
                if c_t != c_g and c_g != "-":
                    render_text += f"[green]{c_g}[/green]"
                else:
                    render_text += c_g

            # base_text = (f"[bold yellow]{target}[/bold yellow] [bold cyan]Identity[/bold cyan]")
            render_text += f" [cyan]{identity(target, seq)*100:.2f}%[/cyan]"

            # print(f"{i:>{n_digit}} {base_text}")
        print(f"{i:>{n_digit}} {render_text}")
    print("----------")
    print(f"[bold white]N: {len(msa)}, L: {len(target)}")


# ===================== ffindex


@dataclass
class FFindexEntry:
    """

    Parameters
    ----------
    name:
        ffindex id
    offset:
        start point of the block in ffdata
    length:
        length of the block in ffdata
    N:
        number of sequences in the MSA
    L:
        length of the first sequence in the MSA
    pair: optional
        two sequence id to be target and homology
    weight: optional
        reciprocal of total sampled pairs of the source MSA, used for
        computing loss
    seq: optional
        a list of sampled sequence id

    """

    name: str = ""
    offset: int = -1
    length: int = -1
    N: int = -1
    L: int = -1

    # optional properties
    pair: Optional[Tuple[int, int]] = (-1, -1)
    weight: Optional[float] = 1.0

    seq: Optional[Tuple[int]] = ()

    @property
    def P(self):
        return self.N * (self.N - 1)


def ffindex2ffentry(line, ex_props=[]):  # Process a single line in jsonl file
    offset, length = line["part"]
    entry = FFindexEntry(  # Get the entry
        name=line["name"],
        offset=offset,
        length=length,
        N=line["N"],
        L=line["L"],
    )
    for prop in ex_props:
        assert prop in line, f"{prop} does not exist. Please check input files."
        if prop == "iw":
            entry.weight = 1 / prop["iw"]
        else:
            setattr(entry, prop, line[prop])
    return entry


def fffile2ffentries(ffindex_path, ex_props=[]):
    """Read ffindex file into a list of FFindexEntry"""
    return [
        ffindex2ffentry(sample, ex_props=ex_props)
        for sample in read_jsonlines(ffindex_path)
    ]


def index2pair(n_item, pair_index, bidirection=True):
    if not bidirection:
        assert (
            pair_index < n_item * (n_item - 1) - 1
        ), "pos should smaller than total pairs"
        i_pos = n_item * (n_item - 1) // 2 - pair_index - 1
        i_start = int(np.ceil(np.sqrt((i_pos + 1) * 2 + 0.25) - 0.5))
        i_end = i_pos - i_start * (i_start - 1) // 2
        start, end = n_item - i_start - 1, n_item - i_end - 1
        return start, end
    else:
        reverse = (pair_index % 2) == 1
        start, end = index2pair(n_item, pair_index // 2, bidirection=False)
        if reverse:
            return end, start
        else:
            return start, end


def align_pair(seq1: str, seq2: str):
    seq1 = Seq(seq1)
    seq2 = Seq(seq2)
    alignments = pairwise2.align.globalxx(seq1, seq2)
    #    for aln in alignments:
    #        print(aln.seqA, aln.seqB)
    return alignments[0].seqA, alignments[0].seqB


def identity(seq1: str, seq2: str) -> float:
    """return the identity of two aligned sequences"""
    seq1, seq2 = pair_a3m(seq1, seq2)
    # print("after pairing...")
    # print(seq1)
    # print(seq2)
    assert len(seq1) == len(seq2)
    identity = (np.array(list(seq1)) == np.array(list(seq2))).mean()
    return identity


# =====================


def merge_aln(base_path, extend_path, output_path, check=True):
    base_msa = read_lines(base_path, rm_empty_lines=True)
    extend_msa = read_lines(extend_path, rm_empty_lines=True)

    # check target sequence
    assert (
        base_msa[0] == extend_msa[0]
    ), f"target sequences mismatch: \n\t{base_path}\n\t{extend_path}"

    # check columns
    L = len(base_msa[0].strip())
    for i, seq in enumerate(base_msa):
        l_seq = len(seq.strip())
        assert l_seq == L or l_seq == 0, f"columns not equal: {base_path}, line {i}"
    for i, seq in enumerate(extend_msa):
        l_seq = len(seq.strip())
        assert l_seq == L or l_seq == 0, f"columns not equal: {extend_path}, line {i}"

    merge_msa = base_msa + extend_msa[1:]
    write_lines(output_path, merge_msa)
    return {"N": len(merge_msa), "L": len(merge_msa[0].strip())}


def merge_fasta(base_path, extend_path, output_path, check=True):
    base_msa = read_lines(base_path, rm_empty_lines=True)
    extend_msa = read_lines(extend_path, rm_empty_lines=True)
    merge_msa = base_msa + extend_msa[2:]
    write_lines(output_path, merge_msa)
    return {"N": len(merge_msa), "L": len(merge_msa[0].strip())}


def build_sample_list_aln(sample_dir, output_path, old_list=None):
    sample_dir = Path(sample_dir).expanduser()
    new_samples = []
    if old_list is not None:
        samples = read_jsonlines(old_list)
        names = [sample["name"].split(".")[0] for sample in samples]
    else:
        names = [sample.stem for sample in sample_dir.glob("*.aln")]

    for name in names:
        path = sample_dir / f"{name}.aln"
        if path.exists():
            lines = read_lines(path, rm_empty_lines=True)
            sample = {"name": name, "N": len(lines), "L": len(lines[0].strip())}
            new_samples.append(sample)
        else:
            print(f"missing: {path}")

    write_jsonlines(output_path, new_samples)
    return new_samples


def deduplicate_msa_a3m(a3m_paths, output_path):
    lines = []
    for p in a3m_paths:
        with open(p) as fd:
            _lines = fd.read().split("\n")
        lines.extend(_lines)
    a3m_msa = [seq.strip() for seq in lines if seq.strip() != ""]
    desc = []
    seqs = []
    for seq in a3m_msa:
        if seq.startswith(">"):
            desc.append(seq)
        else:
            seqs.append(seq)
    seq_set = set()
    collects = []
    for seq, des in zip(seqs, desc):
        if seq in seq_set:
            continue
        else:
            seq_set.add(seq)
            collects.append(f"{des}\n")
            collects.append(f"{seq}\n")
    with open(output_path, "w") as f:
        f.writelines(collects)
    logger.info(
        f" msa deduplicate with input_lines :{len(a3m_msa)}, output_lines:{len(collects)} "
    )


# =====================


class Sequence:
    def __init__(self, seq):
        self.seq = seq
        self.idx = 0

    def next_char(self):
        if self.idx >= len(self.seq):
            return None
        else:
            char = self.seq[self.idx]
            self.idx += 1
            return char

    def is_next_lower(self):
        if self.idx >= len(self.seq):
            return False
        else:
            return self.seq[self.idx].islower()

    def skip_lower(self):
        while self.is_next_lower():
            self.idx += 1


def pair_a3m(seq1: str, seq2: str) -> Tuple[str, str]:
    """Make two sequences from one MSA into a pair of (target, homology).

    Notes
    ----------
    (target, homology):
        - target is full length without gaps
        - homology is aligned to the target, possible to have gaps
    The result can be seen as two sequences in [FASTA]_ format, where seq1
    becomes the first sequence.

    Parameters
    ----------
    seq1 : str
        sequence from a msa, to be the target of the output pair.
    seq2 : str
        sequence from the same mas as seq1, to be the homology of the output pair.

    Returns
    -------
    Tuple[str, str]
        target sequence, homology sequence

    References
    ----------
    .. [FASTA] https://yanglab.nankai.edu.cn/trRosetta/msa_format.html#fasta

    """
    new_seq1, new_seq2 = "", ""
    seq1 = Sequence(seq1)
    seq2 = Sequence(seq2)
    while True:
        char1 = seq1.next_char()
        if char1 is None:
            break
        elif char1.islower():
            new_seq1 += char1.upper()
            if seq2.is_next_lower():
                new_seq2 += seq2.next_char().upper()
            else:
                new_seq2 += "-"
        else:
            seq2.skip_lower()
            char2 = seq2.next_char()
            if char1.isupper():
                new_seq1 += char1
                new_seq2 += char2
    return new_seq1, new_seq2


# =====================
# MSA analysis
# =====================


def load_a3m_ordered_by_qid(a3m_path):

    delete_lowercase = lambda line: "".join([t for t in list(line) if not t.islower()])
    calculate_qid = lambda x, y: sum([i == j for i, j in zip(x, y)]) / len(x)
    descriptions = []
    sequences = []
    with open(a3m_path) as fd:
        for line in fd:
            line = line.strip()
            if line.startswith(">"):
                descriptions.append(line)
            else:
                sequences.append(line)
    ori_seq = sequences[0]
    ds_qid = [
        (calculate_qid(ori_seq, delete_lowercase(s)), d, s)
        for d, s in zip(descriptions, sequences)
    ]
    ds_qid = list(set(ds_qid))
    ordered_ds_qid = sorted(ds_qid, key=lambda x: x[0], reverse=True)
    return ordered_ds_qid


def save_object_as_pickle(obj: Any, path: Union[str, Path]):
    with open(path, "wb") as fd:
        pkl.dump(obj=obj, file=fd, protocol=4)
    return path
