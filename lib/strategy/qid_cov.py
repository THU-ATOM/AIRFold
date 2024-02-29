import numpy as np
import os
import argparse
from pathlib import Path


def qid_cov_filter(msas, q_t, c_t):
    query_seq = np.asarray([list(msas[0])])
    msa_ = np.asarray([list(seq) for seq in msas])
    # n * L
    cov_ = msa_ != "_"
    # n * L
    qid_ = msa_ == query_seq
    accepted = []
    l_ = len(msas[0])
    for n in range(len(msas)):
        q = qid_[n].sum() / cov_[n].sum()
        c = cov_[n].sum() / l_
        if q > q_t and c > c_t:
            accepted.append(msas[n])

    return accepted


def _run(fasta_dir, strategy_dir, q_t, c_t):
    # strategy_dir = base_dir
    os.makedirs(Path(strategy_dir).parent, exist_ok=True)
    with open(fasta_dir) as f:
        msas = f.readlines()

        tmp = []
        for l_ in msas:
            l_ = l_.strip()
            if not l_.startswith(">"):
                # change a3m to fasta format
                _l = "".join([l for l in l_ if not l.islower()])
                tmp.append(_l)
    msas = tmp
    msas = qid_cov_filter(msas, q_t, c_t)
    with open(strategy_dir, "w") as f:
        f.writelines([">" + "\n" + line + "\n" for line in msas])
    # entropy_filter(msas,mask_per=mask_p,output_dir=strategy_dir,name=sample["name"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # a3m_dir,strategy_dir,seq_id,cov_id,sample,rm_tmp_files=False
    parser.add_argument("-i", "--input_a3m_path", required=True, type=str)
    parser.add_argument("-o", "--output_a3m_path", required=True, type=str)
    # parser.add_argument("-n", "--name", required=True, type=str)
    parser.add_argument("--q_t", required=True, type=float)
    parser.add_argument("--c_t", required=True, type=float)

    args = parser.parse_args()
    _run(args.input_a3m_path, args.output_a3m_path, args.q_t, args.c_t)
