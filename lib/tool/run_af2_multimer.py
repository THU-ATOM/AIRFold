# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functions for building the features for the AlphaFold multimer model."""
import argparse
import collections
import dataclasses
import os

from typing import Mapping, MutableMapping, Sequence
import numpy as np
from copy import deepcopy
from absl import logging
from loguru import logger

from lib.tool.alphafold.common import protein
from lib.tool.alphafold.common import residue_constants
from lib.tool.alphafold.data import feature_processing
from lib.tool.alphafold.data import from_msa_to_feature
from lib.tool.alphafold.data import msa_pairing
from lib.tool.alphafold.data import parsers
from lib.tool.alphafold.data import pipeline


import lib.utils.datatool as dtool
from lib.utils.systool import get_available_gpus
from lib.constant import AF_PARAMS_ROOT
from lib.tool.run_af2_stage import monomer_msa2feature, predict_structure, run_relaxation

logging.set_verbosity(logging.INFO)

# Internal import (7716).


@dataclasses.dataclass(frozen=True)
class _FastaChain:
    sequence: str
    description: str


def _make_chain_id_map(
    *,
    sequences: Sequence[str],
    descriptions: Sequence[str],
) -> Mapping[str, _FastaChain]:
    """Makes a mapping from PDB-format chain ID to sequence and description."""
    if len(sequences) != len(descriptions):
        raise ValueError(
            "sequences and descriptions must have equal length. "
            f"Got {len(sequences)} != {len(descriptions)}."
        )
    if len(sequences) > protein.PDB_MAX_CHAINS:
        raise ValueError(
            "Cannot process more chains than the PDB format supports. "
            f"Got {len(sequences)} chains."
        )
    chain_id_map = {}
    for chain_id, sequence, description in zip(
        protein.PDB_CHAIN_IDS, sequences, descriptions
    ):
        chain_id_map[chain_id] = _FastaChain(sequence=sequence, description=description)
    return chain_id_map


def convert_monomer_features(
    monomer_features: pipeline.FeatureDict, chain_id: str
) -> pipeline.FeatureDict:
    """Reshapes and modifies monomer features for multimer models."""
    converted = {}
    converted["auth_chain_id"] = np.asarray(chain_id, dtype=np.object_)
    unnecessary_leading_dim_feats = {
        "sequence",
        "domain_name",
        "num_alignments",
        "seq_length",
    }
    for feature_name, feature in monomer_features.items():
        if feature_name in unnecessary_leading_dim_feats:
            # asarray ensures it's a np.ndarray.
            feature = np.asarray(feature[0], dtype=feature.dtype)
        elif feature_name == "aatype":
            # The multimer model performs the one-hot operation itself.
            feature = np.argmax(feature, axis=-1).astype(np.int32)
        elif feature_name == "template_aatype":
            feature = np.argmax(feature, axis=-1).astype(np.int32)
            new_order_list = residue_constants.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE
            feature = np.take(new_order_list, feature.astype(np.int32), axis=0)
        elif feature_name == "template_all_atom_masks":
            feature_name = "template_all_atom_mask"
        converted[feature_name] = feature
    return converted


def int_id_to_str_id(num: int) -> str:
    """Encodes a number as a string, using reverse spreadsheet style naming.

    Args:
      num: A positive integer.

    Returns:
      A string that encodes the positive integer using reverse spreadsheet style,
      naming e.g. 1 = A, 2 = B, ..., 27 = AA, 28 = BA, 29 = CA, ... This is the
      usual way to encode chain IDs in mmCIF files.
    """
    if num <= 0:
        raise ValueError(f"Only positive integers allowed, got {num}.")

    num = num - 1  # 1-based indexing.
    output = []
    while num >= 0:
        output.append(chr(num % 26 + ord("A")))
        num = num // 26 - 1
    return "".join(output)


def add_assembly_features(
    all_chain_features: MutableMapping[str, pipeline.FeatureDict],
) -> MutableMapping[str, pipeline.FeatureDict]:
    """Add features to distinguish between chains.

    Args:
      all_chain_features: A dictionary which maps chain_id to a dictionary of
        features for each chain.

    Returns:
      all_chain_features: A dictionary which maps strings of the form
        `<seq_id>_<sym_id>` to the corresponding chain features. E.g. two
        chains from a homodimer would have keys A_1 and A_2. Two chains from a
        heterodimer would have keys A_1 and B_1.
    """
    # Group the chains by sequence
    seq_to_entity_id = {}
    grouped_chains = collections.defaultdict(list)
    for chain_id, chain_features in all_chain_features.items():
        seq = str(chain_features["sequence"])
        if seq not in seq_to_entity_id:
            seq_to_entity_id[seq] = len(seq_to_entity_id) + 1
        grouped_chains[seq_to_entity_id[seq]].append(chain_features)

    new_all_chain_features = {}
    chain_id = 1
    for entity_id, group_chain_features in grouped_chains.items():
        for sym_id, chain_features in enumerate(group_chain_features, start=1):
            new_all_chain_features[
                f"{int_id_to_str_id(entity_id)}_{sym_id}"
            ] = chain_features
            seq_length = chain_features["seq_length"]
            chain_features["asym_id"] = chain_id * np.ones(seq_length)
            chain_features["sym_id"] = sym_id * np.ones(seq_length)
            chain_features["entity_id"] = entity_id * np.ones(seq_length)
            chain_id += 1

    return new_all_chain_features


def pad_msa(np_example, min_num_seq):
    np_example = dict(np_example)
    num_seq = np_example["msa"].shape[0]
    if num_seq < min_num_seq:
        for feat in ("msa", "deletion_matrix", "bert_mask", "msa_mask"):
            np_example[feat] = np.pad(
                np_example[feat], ((0, min_num_seq - num_seq), (0, 0))
            )
        np_example["cluster_bias_mask"] = np.pad(
            np_example["cluster_bias_mask"], ((0, min_num_seq - num_seq),)
        )
    return np_example


def gen_msa_feature(msa_paths, format="a3m", max_seqs=50000):
    msa_collections = []
    for msa_path in msa_paths:
        _msa = from_msa_to_feature.load_msa_from_path(msa_path, format)
        _name = os.path.basename(msa_path)
        msa_collections.append((_name, _msa))

    msas = [
        (_name, parsers.parse_a3m(_msa[format]))
        for _name, _msa in msa_collections
    ]

    msa_features = from_msa_to_feature.make_msa_features(msas=[_m.truncate(max_seqs=max_seqs) for _n, _m in msas])
    for _name, _msa in msas:
        logging.info(f"{_name} MSA size: {len(_msa)} sequences.")

    valid_feats = msa_pairing.MSA_FEATURES + ("msa_species_identifiers",)
    feats = {
        f"{k}_all_seq": v for k, v in msa_features.items() if k in valid_feats
    }
    return feats


def main(args):

    print("------- running stage: multimer structure prediction")
    # set visible gpu device
    gpu_devices = "".join([f"{i}" for i in get_available_gpus(1)])
    logger.info(f"The gpu device used for monomer structure prediction: {gpu_devices}")
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_devices

    multimer_model_pj =  {
            "model_1":"model_1_multimer_v2",
            "model_2":"model_2_multimer_v2",
            "model_3":"model_3_multimer_v2",
            "model_4":"model_4_multimer_v2",
            "model_5":"model_5_multimer_v2"
        }

    out_preffix = str(os.path.join(args.root_path, args.model_name))
    out_unrelaxed_pdb = out_preffix + "_unrelaxed.pdb"
    out_relaxed_pdb = out_preffix + "_relaxed.pdb"
    pkl_output = out_preffix + "_output_raw.pkl"

    chain_sequences = args.chain_sequences
    chain_targets = args.targets
    a3m_paths = args.a3m_paths
    uniprot_a3m_paths = args.uniprot_a3m_paths
    template_feats = args.template_feats

    chain_id_map = _make_chain_id_map(
        sequences=chain_sequences, descriptions=chain_targets
    )

    chain_ids = list(chain_id_map.kyes())
    all_chain_features = {}

    is_homomer_or_monomer = len(set(chain_sequences)) == 1

    if not os.path.exists(out_relaxed_pdb):
        for i in len(chain_sequences):

            template_feat = template_feats[i]
            template_feat = dtool.read_pickle(template_feat)
            argument_dict1 = {
                "sequence": chain_sequences[i],
                "target_name": chain_targets[i],
                "msa_paths": [a3m_paths[i]],
                "template_feature": template_feat,
                "model_name": multimer_model_pj[args.model_name],
                "random_seed": args.random_seed,
                "seqcov": args.seqcov,
                "seqqid": args.seqqid,
                "max_recycles": args.max_recycles,
                "max_msa_clusters": args.max_msa_clusters,
                "max_extra_msa": args.max_extra_msa,
                "num_ensemble": args.num_ensemble
            }
            argument_dict1 = deepcopy(argument_dict1)
            chain_features, _ = monomer_msa2feature(**argument_dict1)

            if not is_homomer_or_monomer:
                uniprot_msa_features = gen_msa_feature(uniprot_a3m_paths[i], format="a3m")
                # for briefly, the msa and deletion_matrix_int will be updated (by uniprot val)
                chain_features.update(uniprot_msa_features)
            chain_features = convert_monomer_features(chain_features, chain_id=chain_ids[i])
            all_chain_features[chain_ids[i]] = chain_features
        
        # get all features
        all_chain_features = add_assembly_features(all_chain_features)
        np_example = feature_processing.pair_and_merge(
            all_chain_features=all_chain_features
        )

        # Pad MSA to avoid zero-sized extra_msa.
        np_example = pad_msa(np_example, 512)
        # return np_example
        
        argument_dict2 = {
            "target_name": chain_targets[i],
            "processed_feature": np_example,
            "model_name": multimer_model_pj[args.model_name],
            "data_dir": str(AF_PARAMS_ROOT),
            "random_seed": args.random_seed,
            "return_representations": True,
            "seqcov": args.seqcov,
            "seqqid": args.seqqid,
            "max_recycles": args.max_recycles,
            "max_msa_clusters": args.max_msa_clusters,
            "max_extra_msa": args.max_extra_msa,
            "num_ensemble": args.num_ensemble,
            "run_multimer_system": True   # different from monomer
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
    parser.add_argument("--chain_sequences", type=str, nargs='*', required=True)
    parser.add_argument("--targets", type=str, nargs='*', required=True)
    parser.add_argument("--a3m_paths", type=str, nargs='*', required=True)
    parser.add_argument("--uniprot_a3m_paths", type=str, nargs='*', required=True)
    parser.add_argument("--template_feats", type=str, nargs='*', required=True)

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

