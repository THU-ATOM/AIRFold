import os
import tempfile
from pathlib import Path
from itertools import combinations
from multiprocessing import Pool, cpu_count

import numpy as np
from loguru import logger

import lib.utils.datatool as dtool
from lib.constant import (
    KALIGN_EXECUTE,
    CLUSTALO_EXECUTE,
    PROBCONS_EXECUTE,
    TMALIGN_EXECUTE,
    TMP_ROOT,
)
from lib.utils.execute import execute
from lib.utils.pdbtool import cut_pdb


def align_pdbs(*pdbs, threads=-1, cut_head=0, cut_tail=0):
    """Align pdbs in parallel.

    Args
    -------
        pdbs: list of pdb files
        threads: number of threads to use, -1 for all available

    Returns
    -------
        dict of resulting matrices.

    Note
    -------
    when aligning protein i and protein j, first get rotation matrix from
    rotation[i][j], and then rotate protein i with the 3*4 matrix by
    for(i=0; i<L; i++)
    {
        X[i] = t[0] + u[0][0]*x[i] + u[0][1]*y[i] + u[0][2]*z[i];
        Y[i] = t[1] + u[1][0]*x[i] + u[1][1]*y[i] + u[1][2]*z[i];
        Z[i] = t[2] + u[2][0]*x[i] + u[2][1]*y[i] + u[2][2]*z[i];
    }
    """
    n_pdb = len(pdbs)
    assert n_pdb >= 2, "Must provide 2 pdbs at least"
    for pdb in pdbs:
        assert Path(pdb).exists(), f"{pdb} not exists"

    tm_scores = np.eye(n_pdb, dtype=np.float64)
    rmsds = np.zeros((n_pdb, n_pdb), dtype=np.float64)
    rotations = np.concatenate(
        [
            np.zeros((n_pdb, n_pdb, 3, 1)),
            np.array([[np.eye(3)] * n_pdb] * n_pdb),
        ],
        axis=-1,
    )

    comb = list(combinations(range(n_pdb), 2))
    n_comb = len(comb)
    comb_pdbs = [(pdbs[i], pdbs[j], cut_head, cut_tail) for i, j in comb]

    n_cpu = cpu_count()
    if threads == -1:
        threads = min(n_cpu, n_comb)
    else:
        threads = min(threads, n_cpu, n_comb)
    logger.info(f"Using {threads} threads to align {n_comb} combinations")
    with Pool(threads) as pool:
        results = list(pool.starmap(align_one_to_one, comb_pdbs))
    for (i, j), res in zip(comb, results):
        tm_scores[i, j] = res["tm_score"][0]
        tm_scores[j, i] = res["tm_score"][1]
        rmsds[i, j] = rmsds[j, i] = res["rmsd"]
        rotations[i, j, :] = res["rotation"]

    return {"tm_score": tm_scores, "rmsd": rmsds, "rotation": rotations}


def align_one_to_one(pdb_i, pdb_j, cut_head=0, cut_tail=0):
    pdb_i = Path(pdb_i)
    pdb_j = Path(pdb_j)
    try:
        tmp_out_fd, tmp_out_path = tempfile.mkstemp(dir=TMP_ROOT, suffix=".txt")
        path_out = Path(tmp_out_path)
        tmp_rotation_fd, tmp_rotation_path = tempfile.mkstemp(
            dir=TMP_ROOT, suffix=".txt"
        )
        path_rotation = Path(tmp_rotation_path)
        if cut_head > 0 or cut_tail > 0:
            tmp_pdb_i_fd, tmp_pdb_i_path = tempfile.mkstemp(
                dir=TMP_ROOT, suffix=".pdb"
            )
            cut_pdb(pdb_i, tmp_pdb_i_path, cut_head, cut_tail)
            tmp_pdb_j_fd, tmp_pdb_j_path = tempfile.mkstemp(
                dir=TMP_ROOT, suffix=".pdb"
            )
            cut_pdb(pdb_j, tmp_pdb_j_path, cut_head, cut_tail)
            pdb_i = Path(tmp_pdb_i_path)
            pdb_j = Path(tmp_pdb_j_path)
        execute(
            f"{TMALIGN_EXECUTE} {pdb_i} {pdb_j} -m {path_rotation}",
            log_path=path_out,
        )
        result = parse_scores(path_out)
        matrix = parse_matrix(path_rotation)
        os.close(tmp_out_fd)
        os.close(tmp_rotation_fd)
        if cut_head > 0 or cut_tail > 0:
            os.close(tmp_pdb_i_fd)
            os.close(tmp_pdb_j_fd)
        path_out.unlink()
        path_rotation.unlink()
    except:
        logger.exception("failed")
        raise

    return {
        "align_length": result["align_length"],
        "rmsd": result["rmsd"],
        "identity": result["identity"],
        "tm_score": result["tm_score"],
        "rotation": matrix,
    }


def parse_scores(result_path):
    """
    Note:
    tm_score: [<score normlized by chain 1>, <score normlized by chain 2>]
    """
    PREFIX_ALIGN = "Aligned length="
    PREFIX_TM = "TM-score="
    lines = dtool.read_lines(result_path)
    tm_score = []
    for line in lines:
        if line.startswith(PREFIX_ALIGN):
            t_align_length, t_rmsd, t_identity = line.split(",")
            align_length = int(t_align_length.split()[-1])
            rmsd = float(t_rmsd.split()[-1])
            identity = float(t_identity.split()[-1])
        elif line.startswith(PREFIX_TM):
            tm_score.append(line.split()[1])
    return {
        "align_length": align_length,
        "rmsd": rmsd,
        "identity": identity,
        "tm_score": tm_score,
    }


def parse_matrix(rotation_matrix_path):
    START_LINE = 2
    END_LINE = 5
    lines = dtool.read_lines(rotation_matrix_path)
    matrix = np.array(
        [
            [float(item) for item in line.split()[1:]]
            for line in lines[START_LINE:END_LINE]
        ]
    )
    return matrix


def align_sequences(
    in_fasta: Path, out_fasta: Path, tool="kalign", max_threads: int = 32
):
    """
    Args
    --------
        in_fasta: fasta file to align
        out_fasta: fasta file to write
        tool: tool used to align, kalign, probocons, or clustalo
        max_threads: max number of threads to use, only for clustalo
    """
    available_tool = ["kalign", "probcons", "clustalo"]
    assert tool in available_tool, f"{tool} not available"

    try:
        if tool == "kalign":
            execute(
                f"{KALIGN_EXECUTE} -i {in_fasta} -o {out_fasta} -format fasta"
            )
        elif tool == "probcons":
            execute(f"{PROBCONS_EXECUTE} {in_fasta} > {out_fasta}")
        elif tool == "clustalo":
            threads = min(cpu_count(), max_threads)
            execute(
                f"{CLUSTALO_EXECUTE} -i {in_fasta} -o {out_fasta} "
                f"--auto --threads {threads} --force"
            )
        else:
            raise ValueError(f"Unknown tool {tool}")
    except:
        logger.exception("failed")
        raise
