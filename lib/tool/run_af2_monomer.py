import argparse
import collections
import contextlib
import copy
import dataclasses
import json
import os
import tempfile
from typing import Mapping, MutableMapping, Sequence
import numpy as np
from copy import deepcopy
from absl import logging
from loguru import logger


from lib.tool.alphafold.common import protein
from lib.tool.alphafold.common import residue_constants
from lib.tool.alphafold.data import feature_processing
from lib.tool.alphafold.data import msa_pairing
from lib.tool.alphafold.data import parsers
from lib.tool.alphafold.data import pipeline
from lib.tool.alphafold.data.tools import jackhmmer


import lib.utils.datatool as dtool
from lib.utils.systool import get_available_gpus
from lib.constant import AF_PARAMS_ROOT
from lib.tool.run_af2_stage import monomer_msa2feature, predict_structure, run_relaxation

logging.set_verbosity(logging.INFO)


def main(args):

    print("------- running stage: monomer structure prediction")
    # set visible gpu device
    gpu_devices = "".join([f"{i}" for i in get_available_gpus(1)])
    logger.info(f"The gpu device used for monomer structure prediction: {gpu_devices}")
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_devices

    out_preffix = str(os.path.join(args.root_path, args.model_name))
    out_unrelaxed_pdb = out_preffix + "_unrelaxed.pdb"
    out_relaxed_pdb = out_preffix + "_relaxed.pdb"
    pkl_output = out_preffix + "_output_raw.pkl"

    if not os.path.exists(out_relaxed_pdb):
        template_feat = args.template_feat
        template_feat = dtool.read_pickle(template_feat)
        argument_dict1 = {
            "sequence": args.sequence,
            "target_name": args.target_name,
            "msa_paths": [args.a3m_path],
            "template_feature": template_feat,
            "model_name": args.model_name,
            "random_seed": args.random_seed,
            "seqcov": args.seqcov,
            "seqqid": args.seqqid,
            "max_recycles": args.max_recycles,
            "max_msa_clusters": args.max_msa_clusters,
            "max_extra_msa": args.max_extra_msa,
            "num_ensemble": args.num_ensemble
        }
        argument_dict1 = deepcopy(argument_dict1)
        processed_feature, _ = monomer_msa2feature(**argument_dict1)
        
        argument_dict2 = {
            "target_name": args.target_name,
            "processed_feature": processed_feature,
            "model_name": args.model_name,
            "data_dir": str(AF_PARAMS_ROOT),
            "random_seed": args.random_seed,
            "return_representations": True,
            "seqcov": args.seqcov,
            "seqqid": args.seqqid,
            "max_recycles": args.max_recycles,
            "max_msa_clusters": args.max_msa_clusters,
            "max_extra_msa": args.max_extra_msa,
            "num_ensemble": args.num_ensemble,
            "run_multimer_system": False   # different from multimer
        }
        argument_dict2 = deepcopy(argument_dict2)
        try:
            prediction_results, unrelaxed_pdb_str, _ = predict_structure(**argument_dict2)
            dtool.save_object_as_pickle(prediction_results, pkl_output)
            dtool.write_text_file(plaintext=unrelaxed_pdb_str, path=out_unrelaxed_pdb)

            argument_dict3 = {"unrelaxed_pdb_str": unrelaxed_pdb_str}
            relaxed_pdb_str, _ = run_relaxation(**argument_dict3)
            dtool.write_text_file(relaxed_pdb_str, path=out_relaxed_pdb)

        except TimeoutError as exc:
            logger.exception(exc)
            return False
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=str, required=True)
    parser.add_argument("--target_name", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--a3m_path", type=str, required=True)
    parser.add_argument("--template_feat", type=str, required=True)

    parser.add_argument("--random_seed", type=int, default=0)

    # alphafold params
    parser.add_argument("--seqcov", type=int, default=0)
    parser.add_argument("--seqqid", type=int, default=0)
    parser.add_argument("--max_recycles", type=int, default=64)
    parser.add_argument("--max_msa_clusters", type=int, default=508)
    parser.add_argument("--max_extra_msa", type=int, default=5120)
    parser.add_argument("--num_ensemble", type=int, default=1)

    args = parser.parse_args()
    main(args)
