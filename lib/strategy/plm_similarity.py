import argparse
from collections import Counter
import dataclasses
import numpy as np
from multiprocessing import Pool
from functools import partial
import multiprocessing
import os
import pickle
import torch
from pathlib import Path
from typing import List
from loguru import logger
import torch.nn as nn
from tqdm import trange
from lib.strategy.plmsim.plmsearch_util.model import plmsearch
from lib.strategy.plmsim import embedding_generate, main_similarity


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
    
    if torch.cuda.is_available()==False:
        print("GPU selected but none of them is available.")
        device = "cpu"
    else:
        print("We have", torch.cuda.device_count(), "GPUs in total!, we will use as you selected")
        device = f"cuda:{args.device_id[0]}"
    
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
        model = nn.DataParallel(model, device_ids = args.device_id)
        model_methods = model.module
    model.to(device)
    
    search_result = main_similarity.main(query_embedding_dict, target_embedding_dict, device, model_methods, args.least_seqs)
    
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
    parser.add_argument("-d", "--device_id", default=[0], nargs="*", help="gpu device list, if only cpu then set it None or empty")
    
    argv = parser.parse_args()
    process(argv)