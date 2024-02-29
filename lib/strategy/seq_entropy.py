import argparse
from collections import Counter
import dataclasses
import numpy as np
from multiprocessing import Pool
from functools import partial
import multiprocessing
import os
from pathlib import Path
from typing import List
from loguru import logger

cpu_num = multiprocessing.cpu_count()


@dataclasses.dataclass(frozen=True)
class A3Mentry:
    description: str
    sequence: str
    fasta_sequence: str


def a3m_sequence_to_fasta(sequence):
    res = sequence.strip()
    res = "".join([ch for ch in res if not ch.islower()])
    return res


def read_a3m_file(file_path: str):
    with open(file_path) as f:
        lines = f.readlines()
    a3m_entries = []
    sequence = None
    desc = None
    for l in lines:
        if l.startswith(">"):
            if sequence:
                a3m_entries.append(
                    A3Mentry(
                        desc,
                        sequence,
                        a3m_sequence_to_fasta(sequence),
                    )
                )
            desc = l.strip()
            sequence = ""
        else:
            sequence += l.strip()
    if desc:
        a3m_entries.append(A3Mentry(desc, sequence, a3m_sequence_to_fasta(sequence)))
    return a3m_entries


def getEntropyPerSequence(a3m: A3Mentry, stats: List[int], total_num: int):
    ent = 0
    for i, ch in enumerate(a3m.fasta_sequence):
        try:
            ent += np.log(total_num) - np.log(stats[i][ch])
        except:
            raise
            # print(len(stats),total_num,i,ch)
    return ent


def dropLeastEnt(a3ms, stats, reduce_ratio=0.1):
    num = max(int(len(a3ms) * reduce_ratio), 5)
    with Pool(int(cpu_num * 0.6)) as p:
        ents = np.array(
            p.map(
                partial(getEntropyPerSequence, stats=stats, total_num=len(a3ms)),
                a3ms,
            )
        )
    _sort = np.argsort(ents)

    rm_idx_set = {v for v in _sort[:num] if v != 0}
    _seqs = [s for i, s in enumerate(a3ms) if i not in rm_idx_set]
    _rm_seq = [s for i, s in enumerate(a3ms) if i in rm_idx_set]

    _rm_stats = [
        Counter(item)
        for item in list(zip(*[entry.fasta_sequence for entry in _rm_seq]))
    ]

    _stats = [a - b for a, b in zip(stats, _rm_stats)]
    return _seqs, _stats


def process(sfn, tfn, reduce_ratio, least_seqs):

    a3m_entries = read_a3m_file(sfn)
    logger.info(f"{len(a3m_entries)} sequences in {sfn}")
    fasta_sequences = [entry.fasta_sequence for entry in a3m_entries]
    length = None
    for seq in fasta_sequences:
        if not length:
            length = len(seq)
        else:
            if length != len(seq):
                raise ValueError("Sequences are not of the same length")
    stats = [Counter(item) for item in list(zip(*fasta_sequences))]
    _a3ms, _stats = a3m_entries, stats
    while len(_a3ms) > least_seqs:
        _a3ms, _stats = dropLeastEnt(_a3ms, _stats, reduce_ratio)
        logger.info(f"{len(_a3ms)} sequences left")

    logger.info(f"{len(_a3ms)} sequences left")
    wstr = "".join([f"{a3m.description}\n{a3m.sequence}\n" for a3m in _a3ms])

    with open(tfn, "w") as fd:
        fd.write(wstr)


def _run(fasta_dir, strategy_dir, reduce_ratio, least_seqs):
    # strategy_dir = base_dir
    os.makedirs(Path(strategy_dir).parent, exist_ok=True)
    sfn = fasta_dir
    tfn = strategy_dir
    if not os.path.exists(tfn):
        process(sfn, tfn, reduce_ratio, least_seqs)
    else:
        logger.info("File already exists, skipping")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # a3m_dir,strategy_dir,seq_id,cov_id,sample,rm_tmp_files=False
    parser.add_argument("-i", "--input_a3m_path", required=True, type=str)
    parser.add_argument("-o", "--output_a3m_path", required=True, type=str)
    parser.add_argument("-r", "--reduce_ratio", required=True, type=float)
    parser.add_argument("-l", "--least_seqs", required=True, type=int)
    # parser.add_argument("--cid", default=0.8)
    # parser.add_argument("--rm", default=False)
    args = parser.parse_args()
    _run(args.input_a3m_path, args.output_a3m_path, args.reduce_ratio, args.least_seqs)
