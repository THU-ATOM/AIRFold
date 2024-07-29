import argparse
import torch
import numpy as np
import torch
import random
import re
from collections import OrderedDict
# import RoseTTAFold2
from lib.tool.rosettafold2.network import predict
from lib.utils.systool import get_available_gpus


def get_sequence(path):
    with open(path, "r") as f:
        for line in f:
            if len(line) != 0 and line[0] != "#" and line[0] != ">":
                return line.strip()
    return None

def get_unique_sequences(seq_list):
    unique_seqs = list(OrderedDict.fromkeys(seq_list))
    return unique_seqs

def run_tf(fasta_path, a3m_file, out_base, model_params, run_config, seed):
    sequence = get_sequence(fasta_path)
    # process sym
    sym = "X" #@param ["X","C", "D", "T", "I", "O"]
    order = 1 #@param ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"] {type:"raw"}
    if sym in ["X","C"]:
        copies = order
    elif sym in ["D"]:
        copies = order * 2
    else:
        copies = {"T":12,"O":24,"I":60}[sym]
        order = ""
    symm = sym + str(order)

    # process the sequence
    sequence = re.sub("[^A-Z:]", "", sequence.replace("/",":").upper())
    sequence = re.sub(":+",":",sequence)
    sequence = re.sub("^[:]+","",sequence)
    sequence = re.sub("[:]+$","",sequence)
    sequences = sequence.replace(":","/").split("/")
    if run_config["collapse_identical"]:
        u_sequences = get_unique_sequences(sequences)
    else:
        u_sequences = sequences
    sequences = sum([u_sequences] * copies,[])
    lengths = [len(s) for s in sequences]
    # TODO
    subcrop = 1000 if sum(lengths) > 1400 else -1
    sequence = "/".join(sequences)
    
    # config preciction
    msa_concat_mode = run_config["msa_concat_mode"] #@param ["diag", "repeat", "default"]
    # RoseTTAFold2 settings
    num_recycles = run_config["num_recycles"] #@param [0, 1, 3, 6, 12, 24] {type:"raw"}
    # stochastic settings
    use_mlm = run_config["use_mlm"] #@param {type:"boolean"}
    use_dropout = run_config["use_dropout"] #@param {type:"boolean"}
    max_msa = run_config["max_msa"] #@param [16, 32, 64, 128, 256, 512] {type:"raw"}
    max_extra_msa = max_msa * 8
    
    # TODO: set models
    # random_seed = run_config["random_seed"] #@param {type:"integer"}
    # num_models = run_config["num_models"] #@param ["1", "2", "4", "8", "16", "32"] {type:"raw"}
    
    print(".... compile RoseTTAFold2")
    # model_params = "RF2_apr23.pt"

    device_ids = get_available_gpus(1)
    device = torch.device(f"cuda:{device_ids[0]}") if torch.cuda.is_available() else torch.device("cpu")

    pred = predict.Predictor(model_params, device)


    # for seed in range(random_seed, random_seed + num_models):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    # out_prefix=f"{out_base}/rf2_seed{seed}"
    out_prefix = out_base
    # npz = f"{jobname}/rf2_seed{seed}_00.npz
    # npz = f"{out_prefix}_00.npz"
    pred.predict(inputs=a3m_file,
                out_prefix=out_prefix,
                symm=symm,
                ffdb=None,
                n_recycles=num_recycles,
                msa_mask=0.15 if use_mlm else 0.0,
                msa_concat_mode=msa_concat_mode,
                nseqs=max_msa,
                nseqs_full=max_extra_msa,
                subcrop=subcrop,
                is_training=use_dropout)
        # plddt = np.load(npz)["lddt"].mean()
        # if best_plddt is None or plddt > best_plddt:
        #     best_plddt = plddt
        #     best_seed = seed


def main(args):
    rose_config = {"random_seed": args.random_seed,
                   "num_models": args.num_models,
                   "msa_concat_mode": args.msa_concat_mode,
                   "num_recycles": args.num_recycles, 
                   "max_msa": args.max_msa,
                   "collapse_identical": False if args.collapse_identical == 0 else True,
                    "use_mlm": False if args.use_mlm == 0 else True, 
                    "use_dropout": False if args.use_dropout == 0 else True
                }
    
    run_tf(args.fasta_path, args.a3m_path, args.rose_dir, args.rf2_pt, run_config=rose_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta_path", type=str, required=True)
    parser.add_argument("--a3m_path", type=str, required=True)
    parser.add_argument("--rose_dir", type=str, required=True)
    parser.add_argument("--rf2_pt", type=str, required=True)
    
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--num_models", type=int, default=1)
    
    
    parser.add_argument("--msa_concat_mode", type=str, default="diag", choices=["diag", "repeat", "default"])
    
    parser.add_argument("--num_recycles", type=int, default=6)
    parser.add_argument("--max_msa", type=int, default=256)
    
    parser.add_argument("--collapse_identical", type=int, default=0, choices=[0, 1])
    parser.add_argument("--use_mlm", type=int, default=0, choices=[0, 1])
    parser.add_argument("--use_dropout", type=int, default=0, choices=[0, 1])

    args = parser.parse_args()
    main(args)
