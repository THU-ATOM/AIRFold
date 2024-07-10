import json
import requests
from loguru import logger
import os


def load_fasta(file_path, dir_name, data_suffix):
    # data_suffix: 2024-04-09
    seq_name = ""
    sequence = ""
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith(">"):
                seq_name = data_suffix + "_" + dir_name
            else:
                sequence = line.strip()
                
    return seq_name, sequence
                

def MQEWorker(request_dicts):
    
    mqe_url = f"http://10.0.0.12:8081/mqe"
    try:
        logger.info(f"------- Requests of mqe task: {request_dicts}")
        requests.post(mqe_url , json={'requests': request_dicts})
    except Exception as e:
        logger.error(str(e))

def main():
    # new weeks: 2024.05.04  2024.05.11  2024.05.18  2024.05.25
    cameo_dir = "/data/protein/datasets_2024/experiment/modeling/2024.05.18/"
    data_suffix = "2024-06-05"
    # case_suffix = "base_deepmsa_mmseqs"
    case_suffix = "bdm"
    
    json_files = ["./tmp/temp_6000_64_1_seqentropy_nodq.json",
                 "./tmp/temp_6000_64_1_seqentropy.json",
                 "./tmp/temp_6000_64_1_seqentropy_mmseqs.json",
                 "./tmp/temp_6000_64_1_plmsim_mmseqs.json"]

    dir_names = os.listdir(cameo_dir)
    for dir_name in  dir_names:
        request_dicts = []
        for json_file in json_files:
            with open(json_file, 'r') as jf:
                request_dict = json.load(jf)
            seq_file = cameo_dir + dir_name + "/" + "target.fasta"
            seq_name, sequence = load_fasta(seq_file, dir_name, data_suffix)
            request_dict["sequence"] = sequence
            request_dict["name"] = seq_name + "_" + case_suffix
            request_dict["target"] = seq_name
            
            request_dicts.append(request_dict)
        
        MQEWorker(request_dicts)
            

if __name__ == "__main__":
    logger.info("------- Model Quality  Evaluation -------")
    
    main()
    