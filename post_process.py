import os
import subprocess
from pathlib import Path
from typing import Union
import json
import json
from loguru import logger
import pickle

LDDT_EXECUTE = Path("__file__").parent / "lib" / "tool" / "lddt-linux" / "lddt"


def compute_lddt(
    pred_pdb: Union[str, Path], target_pdb: Union[str, Path], CA=False
):
    score = None
    if not Path(target_pdb).exists():
        raise ValueError(f"target_pdb: {target_pdb} not exist!")
    if not Path(pred_pdb).exists():
        raise ValueError(f"pred_pdb: {pred_pdb} not exist!")

    if Path(pred_pdb).exists() and Path(target_pdb).exists():
        report = subprocess.check_output(
            f"{LDDT_EXECUTE} {'-c' if CA else ''} {pred_pdb} {target_pdb}",
            shell=True,
        ).decode("utf-8")
        prefix = "Global LDDT score: "
        for line in report.split("\n"):
            if line.startswith(prefix):
                score = float(line[len(prefix) :])
    return score, report


def get_lddts(
    pred_pdb: Union[str, Path], target_pdb: Union[str, Path], CA=False
):
    score, report = compute_lddt(
        pred_pdb=pred_pdb, target_pdb=target_pdb, CA=CA
    )
    lines = report.strip().split("\n")
    prefix = "Local LDDT Scores:"
    start_idx = None
    for idx, item in enumerate(lines):
        if item.startswith(prefix):
            start_idx = idx + 2
            break
    lddts = {}
    for line in lines[start_idx:]:
        item = line.strip().split()
        if len(item) == 5:
            offset = 1
        elif len(item) == 6:
            offset = 0
        else:
            raise ValueError("format error")
        if len(item) > 0:
            try:
                lddts[int(item[2 - offset])] = (
                    float(item[4 - offset]) if item[4 - offset] != "-" else -1.0
                )
            except Exception:
                print(line)
                raise ValueError("error !")

    return score, lddts


def load_fasta(fasta_path):
    sequence = ""
    with open(fasta_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            if not line.startswith(">"):
                sequence = line.strip()
    return sequence
                

def a3m_count(a3m_file):
    # a3m_dir: /data/protein/CAMEO/data/2024-04-18/search/intergrated_a3m
    count = 0
    if os.path.exists(a3m_file):
        with open(a3m_file, "r") as af:
            lines = af.readlines()
            for line in lines:
                if line.startswith(">"):
                    count += 1
    else:
        logger.error(f"a3m_file {a3m_file} doesn't exist!")
        return 0
    return count


def process():
    output_dir = "/data/protein/datasets_2024/prediction/"
    target_dir = "/data/protein/datasets_2024/modeling/"
    weeks = ['2024.02.17', '2024.02.24', '2024.03.02', '2024.03.09', '2024.03.16', '2024.03.23', '2024.03.30', '2024.04.06']
    
    data_suffix = "2024-04-18"
    case_suffix = "base"
    
    results_dir = "/data/protein/CAMEO/data/" + data_suffix + "/"
    
    results = []
    for week in  weeks:
        week_dir = target_dir + week + "/"
        ow_dirs = os.listdir(week_dir)
        for target in ow_dirs:
            result = {}
            
            result["date"] = week
            result["target"] = target
            
            # seq, seq_name, seq_len
            result["seq_name"] = data_suffix + "_" + target + "_" + case_suffix
            logger.info(f"------- Processing the seq : {result['seq_name']}")
            
            result["fasta_file"] = week_dir + target + "/" + "target.fasta"
            result["target_pdb"] = week_dir + target + "/" + "target.pdb"
            
            result["sequence"] = load_fasta(result["fasta_file"])
            result["seq_len"] = len(result["sequence"])
            
            # a3m_num
            a3m_dir = results_dir + "search/intergrated_a3m/"
            a3m_file = a3m_dir + result["seq_name"] + ".a3m"
            result["msa_depth"] = a3m_count(a3m_file)
            
            # plddt and lddt of five model
            structure_dir = results_dir + "structure/seq_e_re_0.1_le_5000/" + result["seq_name"] + "/"
            plddt_file = structure_dir + "plddt_results.json"
            result["plddt"] = []
            result["lddt"] = []
            with open(plddt_file, "r") as pf:
                plddt_dict = json.load(pf)
                for key, val in plddt_dict.items():
                    result["plddt"].append(val)
                    result["predicted_pdb"] = structure_dir + key + "_relaxed.pdb"
                    global_lddt, _ = get_lddts(result["predicted_pdb"], result["target_pdb"], CA=False)
                    result["lddt"].append(global_lddt)
            
            results.append(result)
    
    output_file = output_dir + case_suffix + "_results.pkl"
    with open(output_file, "wb") as pf:
        pickle.dump(results, pf)
            

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-i", "--input_path", type=str, required=True)
    # args = parser.parse_args()
    
    logger.info("------- Start to post process -------")
    
    process()
    