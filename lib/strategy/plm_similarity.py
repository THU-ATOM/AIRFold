import argparse
import os

import dataclasses
import pickle
import torch
import torch.nn as nn
from loguru import logger
from pathlib import Path

from lib.strategy.plmsim.plmsearch_util.model import plmsearch
from lib.strategy.plmsim import embedding_generate, similarity_calculate
from lib.utils.systool import get_available_gpus


esm_model_path = "/data/protein/datasets_2024/plmsearch_data/model/esm/esm1b_t33_650M_UR50S.pt"
sim_model_path = "/data/protein/datasets_2024/plmsearch_data/model/plmsearch.sav"

@dataclasses.dataclass(frozen=True)
class A3Mentry:
    id: int
    src: str
    description: str
    sequence: str
    fasta_sequence: str


def a3m_sequence_to_fasta(sequence):
    res = sequence.strip()
    # two strategy: 
    # 1. remove lower letter and "_"; 
    # 2. remove "_" and change lower letter to upper
    
    # res = "".join([ch for ch in res if not (ch.islower() or ch == "-")])
    res = "".join([ch for ch in res if not (ch == "-" or ch in ["j", "J"])])
    return res.upper()


def read_a3m_file(file_path: str):
    with open(file_path) as f:
        lines = f.readlines()
    a3m_entries = []
    seq_id = 0
    seq_src = None
    sequence = None
    desc = None
    for l in lines:
        if l.startswith(">"):
            if sequence:
                a3m_entries.append(
                    A3Mentry(
                        seq_id,
                        seq_src,
                        desc,
                        sequence,
                        a3m_sequence_to_fasta(sequence),
                    )
                )
            desc = l.strip()
            seq_src = desc.split()[-1]
            sequence = ""
        else:
            seq_id += 1
            sequence += l.strip()
    if desc:
        a3m_entries.append(A3Mentry(seq_id, seq_src, desc, sequence, a3m_sequence_to_fasta(sequence)))
    return a3m_entries


def process(args):
    os.makedirs(Path(args.output_a3m_path).parent, exist_ok=True)
    path_prefix = os.path.splitext(args.output_a3m_path)[0]
    print("------------path_prefix: ", path_prefix)
    
    # from a3m to fasta file
    a3m_entries = read_a3m_file(args.input_a3m_path)
    logger.info(f"{len(a3m_entries)} sequences in {args.input_a3m_path}")
    wstr = "".join([f">{a3m.id}|{a3m.src}\n{a3m.fasta_sequence}\n" for a3m in a3m_entries])
    target_fasta_file = path_prefix + "_target.fasta"
    # dp_target_fasta_file = path_prefix + "_target_dp.fasta"
    with open(target_fasta_file, "w") as fd:
        fd.write(wstr)
    # dtool.deduplicate_msa_a3m_plus([target_fasta_file], dp_target_fasta_file)
    
    device_ids = get_available_gpus(1)
    gpu_devices = "".join([f"{i}" for i in device_ids])
    logger.info(f"The gpu device used for PLMSearch: {gpu_devices}")
    if torch.cuda.is_available()==False:
        print("GPU selected but none of them is available.")
        device = "cpu"
    else:
        print("We have", torch.cuda.device_count(), "GPUs in total!, we will use as you selected")
        device = f"cuda:{device_ids[0]}"
    
    # generate embedding for query fasta and target fasta
    
    query_embedding_path = path_prefix + "_query.pkl"
    target_embedding_path = path_prefix + "_tar.pkl"
    # if not os.path.exists(query_embedding_path):
    embedding_generate.main(esm_model_path, args.input_fasta_path, query_embedding_path, device, device_id=device_ids[0])
    # if not os.path.exists(target_embedding_path):
    embedding_generate.main(esm_model_path, target_fasta_file, target_embedding_path, device, device_id=device_ids[0])
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
    
    search_result_path = path_prefix + "_similarity.csv"
    # if not os.path.exists(search_result_path):
    similarity_calculate.main(query_embedding_dict, target_embedding_dict, device, model_methods, args.least_seqs, search_result_path)
    select_ids = []
    with open(search_result_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            select_id_src = line.split("\t")[1]
            select_id = int(select_id_src.split("|")[0])
            print("Select seq_id: %d" % select_id)
            select_ids.append(select_id)
            
    
    select_wstr = ""
    for a3m in a3m_entries:
        # if a3m.description in select_desc:
        if a3m.id in select_ids:
            select_wstr += f"{a3m.description}\n{a3m.sequence}\n"
    
    with open(args.output_a3m_path, "w") as f:
        f.write(select_wstr)
    
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--input_fasta_path", required=True, type=str)
    parser.add_argument("-a", "--input_a3m_path", required=True, type=str)
    parser.add_argument("-o", "--output_a3m_path", required=True, type=str)
    parser.add_argument("-l", "--least_seqs", required=True, type=int)
    
    argv = parser.parse_args()
    
    if not os.path.exists(argv.output_a3m_path):
        process(argv)
    else:
        logger.info(f"{argv.output_a3m_path} already exists, skip!")