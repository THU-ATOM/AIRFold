import os, re
from typing import Tuple
from loguru import logger
import numpy as np
import argparse

from sklearn.cluster import SpectralClustering
from collections import defaultdict

from lib.tool.align import align_pdbs
# from lib.tool.colabfold.alphafold.common import protein
from typing import List, Tuple
import lib.tool.tool_utils as utils
import pickle as pkl
import lib.tool.mmcif_get_chain as mmcif_get_chain


def copy_chain_from_mmcif(cif_path, chain_id, name, tgt_cif_path):
    st = mmcif_get_chain.MMCIFParser(mmcif_file=cif_path)
    chain = st.get_chain(chain_id=chain_id)
    mmcif_get_chain.MMCIFParser.write_structure(chain, tgt_cif_path, name=name)


def get_tm_score_matrix_plddt(pdb_paths: List[str]) -> Tuple[np.ndarray, list]:

    results = align_pdbs(*pdb_paths)
    logger.info(
        f"tm_score matrix [shape: {results['tm_score'].shape}] compute complete"
    )

    return results["tm_score"]


def template_selection(
    tm_score_matrix: np.ndarray,
    names: List[str],
    sum_probs: List[float],
    num_cluster: int = 4,
) -> Tuple[list, np.ndarray]:
    groups = defaultdict(list)
    sc = SpectralClustering(
        num_cluster,
        affinity="precomputed",
        n_init=1000,
        assign_labels="discretize",
    )
    labels = sc.fit_predict(tm_score_matrix)
    for l, n, p in zip(labels, names, sum_probs):
        groups[l].append((n, p))
    rets = []
    for l, name2scores in groups.items():
        rets.append(max(name2scores, key=lambda x: x[-1]))
    return rets, labels


def _get_pdb_id_and_chain(name: str) -> Tuple[str, str]:
    """Returns PDB id and chain id for an HHSearch Hit."""
    # PDB ID: 4 letters. Chain ID: 1+ alphanumeric letters or "." if unknown.
    id_match = re.match(r"[a-zA-Z\d]{4}_[a-zA-Z0-9.]+", name)
    if not id_match:
        raise ValueError(f"hit.name did not start with PDBID_chain: {name}")
    pdb_id, chain_id = id_match.group(0).split("_")
    return pdb_id.lower(), chain_id


def main(
    template_feature_path,
    selected_template_feature_path,
    base_template_dir,
    cluster_num=4,
):
    with open(template_feature_path, "rb") as fd:
        tplt_feature = pkl.load(fd)
    if len(tplt_feature["template_domain_names"].tolist()) <= cluster_num:
        with open(selected_template_feature_path, "wb") as fd:
            pkl.dump(tplt_feature, fd)
        return selected_template_feature_path
    with utils.tmpdir_manager() as workdir:
        template_names = []
        name2idx = {}
        name2path = {}
        name2sumprobs_tuple = [
            (n.decode("utf-8") if not isinstance(n, str) else n, s[0])
            for n, s in zip(
                tplt_feature["template_domain_names"],
                tplt_feature["template_sum_probs"],
            )
        ]
        name2sumprobs = dict(name2sumprobs_tuple)
        print(name2sumprobs_tuple)
        print(name2sumprobs)
        customized_idx = []

        logger.info(f"all valid templates are:{name2sumprobs_tuple}")
        for idx, (name, sum_probs) in enumerate(name2sumprobs_tuple):
            pdb_id, chain_id = _get_pdb_id_and_chain(name)
            template_cif_path = os.path.join(base_template_dir, f"{pdb_id}.cif")
            if not os.path.exists(template_cif_path):
                customized_idx.append(idx)
                logger.warning(f"{template_cif_path} not found")
                continue
            template_names.append(name)
            name2idx[name] = idx
            name2path[name] = os.path.join(workdir, f"{name}.cif")
            copy_chain_from_mmcif(
                cif_path=template_cif_path,
                tgt_cif_path=name2path[name],
                chain_id=chain_id,
                name=name,
            )
        logger.info(f"to cluster templates are:{template_names}")

        score_matrix = get_tm_score_matrix_plddt(
            pdb_paths=[name2path[name] for name in template_names]
        )
        rets, labels = template_selection(
            tm_score_matrix=score_matrix,
            names=template_names,
            sum_probs=[name2sumprobs[name] for name in template_names],
            num_cluster=cluster_num,
        )
        selected_names = [r[0] for r in rets]
        logger.info(f"selected templates are:{selected_names}")
    selected_feature = {
        k: tplt_feature[k][customized_idx + [name2idx[n] for n in selected_names]]
        for k in tplt_feature
    }
    with open(selected_template_feature_path, "wb") as fd:
        pkl.dump(selected_feature, fd)
    return selected_template_feature_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--src_feat", type=str, required=True)
    parser.add_argument("-t", "--tgt_feat", type=str, required=True)
    parser.add_argument("-m", "--max_pool_size", type=int, default=20)
    parser.add_argument("-n", "--remain_num", type=int, default=4)
    parser.add_argument(
        "-d",
        "--base_template_dir",
        type=str,
        default="/data/protein/datasets_2022/pdb_mmcif/mmcif_files",
    )

    args = parser.parse_args()
    main(
        template_feature_path=args.src_feat,
        selected_template_feature_path=args.tgt_feat,
        base_template_dir=args.base_template_dir,
        cluster_num=args.remain_num,
    )
