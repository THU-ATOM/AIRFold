import os
import glob
import shutil
from typing import Tuple
from loguru import logger
from pathlib import Path
import numpy as np
import argparse

from sklearn.cluster import SpectralClustering
from collections import defaultdict

from lib.tool.align import align_pdbs
from lib.tool.alphafold.common import protein
from typing import List, Tuple


def copy_to_dir(src_pattern: str, target_dir: str) -> None:
    paths = glob.glob(src_pattern)
    for p in paths:
        f = "_".join(p.split("/")[-3:])
        filename = Path(target_dir) / f
        shutil.copy(src=p, dst=filename)


def get_tm_score_matrix_plddt(
    pdb_paths: List[str],
    threshold: float = 0.1,
    cut_head: int = 0,
    cut_tail: int = 0,
) -> Tuple[np.ndarray, list]:

    plddts = []
    pdbfiles = []
    for pdb in pdb_paths:
        with open(pdb) as fd:
            prot = protein.from_pdb_string(fd.read())
            plddt = np.mean(prot.b_factors[:, 0])
            logger.info(f"{plddt:.2f} {pdb}")
            if plddt > threshold:
                plddts.append(plddt)
                pdbfiles.append(pdb)

    results = align_pdbs(*pdbfiles, cut_head=cut_head, cut_tail=cut_tail)
    logger.info(
        f"tm_score matrix [shape: {results['tm_score'].shape}] compute complete"
    )

    return results["tm_score"], plddts, pdbfiles


def model_selection(
    tm_score_matrix: np.ndarray,
    names: List[str],
    plddts: List[float],
    num_cluster: int = 5,
) -> Tuple[list, np.ndarray]:
    groups = defaultdict(list)
    sc = SpectralClustering(
        num_cluster,
        affinity="precomputed",
        n_init=1000,
        assign_labels="discretize",
    )
    labels = sc.fit_predict(tm_score_matrix)
    for l, n, p in zip(labels, names, plddts):
        groups[l].append((n, p))
    group_info = "\n".join([str(groups[l]) for l in groups])
    logger.info(f"cluster groups:\n {group_info}")
    rets = []
    for l, name2plddts in groups.items():
        rets.append(max(name2plddts, key=lambda x: x[-1]))
        items = sorted(name2plddts, key=lambda x: x[-1], reverse=True)
        logger.info(f"{items[0][1]:.4f} {items[0][0]}" )
        for n, p in items[1:]:
            logger.info(f"  - {p:.4f} {n}")
    return rets, labels


def gen_submission(
    submit_dir: str,
    target: str,
    author_code: str = "1673-5955-6191",
) -> str:
    paths = glob.glob(f"{submit_dir}/*")
    sorted_paths = sorted(paths, reverse=True)
    contents = (
        f"PFRMAT TS\n"
        f"TARGET {target}\n"
        f"AUTHOR {author_code}\n"
        f"METHOD Description of methods used\n"
    )

    for i, res in enumerate(sorted_paths):
        coordinates = "".join(
            filter(
                lambda x: x.startswith("ATOM"),
                open(res, "r").readlines(),
            )
        )
        coordinates = f"MODEL  {i+1}\nPARENT N/A\n{coordinates}TER\nEND\n"
        contents = f"{contents}{coordinates}"
    return contents


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--src_pattern", type=str, required=True)
    parser.add_argument("-t", "--tgt_dir", type=str, required=True)
    parser.add_argument("-p", "--plddt_threshold", type=float, default=80.0)
    parser.add_argument("-n", "--num_cluster", type=int, default=5)
    parser.add_argument(
        "-ch",
        "--cut_head",
        type=int,
        default=0,
        help="cut head residues",
    )
    parser.add_argument(
        "-ct",
        "--cut_tail",
        type=int,
        default=0,
        help="cut tail residues",
    )
    parser.add_argument(
        "-a", "--author", type=str, default="air", choices=["air", "helixon"]
    )
    args = parser.parse_args()

    if args.author == "air":
        author_code = "1673-5955-6191"
    elif args.author == "helixon":
        author_code = "1684-3203-7374"
    else:
        raise ValueError("no such author")
    pdbs_dir = args.tgt_dir + "_pdbs"
    tgt_dir = args.tgt_dir + "_submit"
    if Path(pdbs_dir).exists():
        logger.warning(f"{pdbs_dir} already exists, removing..")
        shutil.rmtree(pdbs_dir)
    if Path(tgt_dir).exists():
        logger.warning(f"{tgt_dir} already exists, removing..")
        shutil.rmtree(tgt_dir)
    os.makedirs(pdbs_dir)
    os.makedirs(tgt_dir)

    copy_to_dir(args.src_pattern, pdbs_dir)
    pdbfiles = glob.glob(str(Path(pdbs_dir) / "*.pdb"))
    score_matrix, plddts, pdb_files = get_tm_score_matrix_plddt(
        pdb_paths=pdbfiles,
        threshold=args.plddt_threshold,
        cut_head=args.cut_head,
        cut_tail=args.cut_tail,
    )
    rets, labels = model_selection(
        score_matrix, pdb_files, plddts, num_cluster=args.num_cluster
    )

    name = os.path.basename(args.tgt_dir)
    table = []
    for pdb_path, plddt in rets:
        file_name = f"{name}_{plddt:.2f}_{os.path.basename(pdb_path)}"
        shutil.copy(pdb_path, os.path.join(tgt_dir, file_name))
        table.append((plddt, file_name))

    sources = []
    plddts = []
    for plddt, file_name in sorted(table, reverse=True):
        logger.info(f"{plddt:.2f} {file_name}")
        plddts.append(f"{plddt:.2f}")
        sources.append("H" if "ruihan" in file_name else "A")
    logger.info("\t".join(sources))
    logger.info("\t".join(plddts))

    submit_results = gen_submission(
        submit_dir=tgt_dir,
        target=name,
        author_code=author_code,
    )

    merged_file_path = Path(tgt_dir).parent / f"{name}_submit.pdb"
    with open(merged_file_path, "w") as fd:
        fd.write(submit_results)
