# filter the msa according to the entropy.
import argparse
import math
import os
from pathlib import Path

# strategy {entropy_filer:{mask_per:0.3},q_c_filer:{q_t:0.0,c_t:0.0},}
# {strategy:{msa_filter_strategy: "entropy"}}
def entropy_filter(msas, mask_per=0.3, output_dir=""):
    seq_len = len(msas[0])

    def _get_entropy(seq_l):
        # this get the single shot entropy
        entropy = {}
        frequency = {}
        # print(seq_l[0])
        for i in range(len(seq_l[0])):
            frequency[i] = {}
        for l_ in seq_l:
            for i in range(len(l_)):
                if l_[i] != "-":
                    # if gap we skip
                    if l_[i] not in frequency[i]:
                        frequency[i][l_[i]] = 1
                    else:
                        frequency[i][l_[i]] += 1
        for i in range(len(seq_l[0])):
            if len(frequency) == 0 or len(frequency) == 1:
                entropy[i] = float("inf")
            else:
                sum_ = sum(frequency[i].values())
                count = 0
                for acid in frequency[i].keys():
                    count += -(frequency[i][acid] / sum_) * math.log(
                        frequency[i][acid] / sum_
                    )
                entropy[i] = count
        return entropy

    entrop_ = _get_entropy(msas)
    entrop_ = dict(sorted(entrop_.items(), key=lambda item: item[1]))
    for l_ in msas:
        for i in range(int(seq_len * mask_per)):
            # mask and change to gap
            index = list(entrop_.keys())[i]
            l_ = l_[:index] + "-" + l_[index + 1 :]
            """
            Another solution is changing it into some commonly 
            """
    with open(output_dir, "w") as f:
        f.writelines([">" + "\n" + line + "\n" for line in msas])

    return {"msa": msas}


def _run(fasta_dir, strategy_dir, mask_p):
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
    entropy_filter(msas, mask_per=mask_p, output_dir=strategy_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # a3m_dir,strategy_dir,seq_id,cov_id,sample,rm_tmp_files=False
    parser.add_argument("-i", "--input_a3m_path", required=True, type=str)
    parser.add_argument("-o", "--output_a3m_path", required=True, type=str)
    parser.add_argument("-m", "--mask_p", required=True, type=float)
    # parser.add_argument("--cid", default=0.8)
    # parser.add_argument("--rm", default=False)
    args = parser.parse_args()
    _run(args.input_a3m_path, args.output_a3m_path, args.mask_p)


"""
Debugging Code Only

def read_jsonlines(path):
    with jsonlines.open(path) as reader:
        samples = list(reader)
    return samples
fasta_base_dir = "/sharefs/thuhc-data/CAMEO/data/2022-03-28/hhblits/raw_search_fasta/"
output_dir = "/sharefs/thuhc-data/CAMEO/data/2022-03-28/hhblits/"

def make_fasta_files(js_files):
    samples = read_jsonlines(js_files)
    for sample in samples:
        with open(fasta_base_dir+sample["name"]+".fasta") as f:
            msas = f.readlines()
            tmp = []
            for l_ in msas:
                l_ = l_.strip()
                if len(l_) > 2:
                    tmp.append(l_)
            msas = tmp
            print(sample["name"])
            entropy_filter(msas,mask_per=0.5,base_dir=output_dir,name=sample["name"])


    
make_fasta_files("/sharefs/thuhc-data/CAMEO/data/2022-03-28/data.jsonl")
"""
