import argparse
import os
import random
import time

import dataclasses
import multiprocessing
import pickle
import torch
import torch.nn as nn
from loguru import logger
from pathlib import Path
from gpustat import new_query

from lib.strategy.plmsim.plmsearch_util.model import plmsearch
from lib.strategy.plmsim import embedding_generate, similarity_calculate


REFRESH_SECONDS = 30

cpu_num = multiprocessing.cpu_count()

esm_model_path = "/data/protein/datasets_2024/plmsearch_data/model/esm/esm1b_t33_650M_UR50S.pt"
sim_model_path = "/data/protein/datasets_2024/plmsearch_data/model/plmsearch.sav"

@dataclasses.dataclass(frozen=True)
class A3Mentry:
    description: str
    sequence: str
    fasta_sequence: str


def a3m_sequence_to_fasta(sequence):
    res = sequence.strip()
    # two strategy: 
    # 1. remove lower letter and "_"; 
    # 2. remove "_" and change lower letter to upper
    
    # res = "".join([ch for ch in res if not (ch.islower() or ch == "-")])
    res = "".join([ch for ch in res if not ch == "-"])
    return res.upper()


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


def get_available_gpus(
    num: int = -1,
    min_memory: int = 20000,
    random_select: bool = True,
    wait_time: float = float("inf"),
):
    """Get available GPUs.

    Parameters
    ----------
    num : int, optional
        Number of GPUs to get. The default is -1.
    min_memory : int, optional
        Minimum memory available in GB. The default is 20000.
    random_select : bool, optional
        Random select a GPU. The default is True.
    wait_time : float, optional
        Wait time in seconds. The default is inf.
    """

    start = time.time()
    while time.time() - start < wait_time:
        gpu_list = new_query().gpus
        if random_select:
            random.shuffle(gpu_list)
        sorted_gpu_list = sorted(
            gpu_list,
            key=lambda card: (
                card.entry["utilization.gpu"],
                card.entry["memory.used"],
            ),
        )
        available_gpus = [
            gpu.entry["index"]
            for gpu in sorted_gpu_list
            if gpu.entry["memory.total"] - gpu.entry["memory.used"]
            >= min_memory
        ]
        if num > 0:
            available_gpus = available_gpus[:num]
        if len(available_gpus) > 0:
            return available_gpus
        else:
            logger.info(
                f"No GPU available, having waited {time.time() - start} seconds"
            )
            return False
    raise Exception("No GPU available")

def process(args):
    os.makedirs(Path(args.output_a3m_file).parent, exist_ok=True)
    path_prefix = os.path.splitext(args.output_a3m_file)
    
    # from a3m to fasta file
    a3m_entries = read_a3m_file(args.input_a3m_file)
    logger.info(f"{len(a3m_entries)} sequences in {args.input_a3m_file}")
    wstr = "".join([f"{a3m.description}\n{a3m.fasta_sequence}\n" for a3m in a3m_entries])
    target_fasta_file = path_prefix + "_target.fasta"
    with open(target_fasta_file, "w") as fd:
        fd.write(wstr)
    
    device_ids = get_available_gpus(1)
    if torch.cuda.is_available()==False:
        print("GPU selected but none of them is available.")
        device = "cpu"
    else:
        print("We have", torch.cuda.device_count(), "GPUs in total!, we will use as you selected")
        device = f"cuda:{device_ids[0]}"
    
    # generate embedding for query fasta and target fasta
    query_embedding_path = path_prefix + "_query.pkl"
    target_embedding_path = path_prefix + "_tar.pkl"
    embedding_generate.main(esm_model_path, args.input_fasta_path, query_embedding_path)
    embedding_generate.main(esm_model_path, target_fasta_file, target_embedding_path)
    with open(query_embedding_path, 'rb') as handle:
        query_embedding_dict = pickle.load(handle)
    with open(target_embedding_path, 'rb') as handle:
        target_embedding_dict = pickle.load(handle)
    
    model = plmsearch(embed_dim = 1280)
    model.load_pretrained(sim_model_path)
    model.eval()
    model_methods = model
    if (device != "cpu"):
        model = nn.DataParallel(model, device_ids=device_ids)
        model_methods = model.module
    model.to(device)
    
    search_result = similarity_calculate.main(query_embedding_dict, target_embedding_dict, device, model_methods, args.least_seqs)
    
    search_result_path = path_prefix + "_similarity.csv"
    select_desc = []
    with open(search_result_path, 'w') as f:
            for protein in search_result:
                for pair in search_result[protein]:
                    select_desc.append(pair[0])
                    f.write(f"{protein}\t{pair[0]}\t{pair[1]}\n")
    
    select_wstr = ""
    for a3m in a3m_entries:
        if a3m.description in select_desc:
            select_wstr += f"{a3m.description}\n{a3m.fasta_sequence}\n"
    
    with open(args.output_a3m_file, "w") as f:
        f.write(select_wstr)
    
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--input_fasta_path", required=True, type=str)
    parser.add_argument("-a", "--input_a3m_path", required=True, type=str)
    parser.add_argument("-o", "--output_a3m_path", required=True, type=str)
    parser.add_argument("-l", "--least_seqs", required=True, type=int)
    
    argv = parser.parse_args()
    process(argv)