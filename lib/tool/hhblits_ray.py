"""Search MSA using HHblits.

Output files
----------

- raw_search: hhblits output
- clean_search: only reserve target sequence + retrieved sequences. If the
    input sequence of hhblit is a 


Format
----------
- a3m: homology sequences have insertions (lower characters)
- aln: only contains aligned sequences, no extra comments
- fasta: based on aln, but contains a comment line start with `>` before each sequence

"""
import argparse
import subprocess
import numpy as np
import sys
import time
import traceback
from typing import Any, Dict, List, Union
from functools import partial
from itertools import chain
from pathlib import Path

import jsonlines
import ray
from ray.util.queue import Queue
from loguru import logger

from lib.utils.ray_tools import ProgressBar
from lib.utils.execute import execute
import lib.utils.datatool as dtool
from lib.pathtree import SearchPathTree


N_GPU_PER_THREAD = 0
N_CPU_PER_THREAD = 2


def hhblits_wrapper(
    iteration: int = 3,
    diff: Union[int, str] = "inf",
    e_value: float = 0.001,
    realign_max: int = 100000,
    maxfilt: int = 100000,
    min_prefilter_hits: int = 1000,
    maxseq: int = 1000000,
    datasets: Union[str, List[str]] = None,
    input_path: Union[str, Path] = None,
    output_a3m_path: Union[str, Path] = None,
    verbose: int = 2,
    cpu: int = 8,
) -> str:
    """hhblits wrapper.

    Parameters
    ----------
    iteration : int, optional
        number of iterations, by default 3
    diff : Union[int, str], optional
        filter MSAs by selecting most diverse set of sequences, keeping
        at least this many seqs in each MSA block of length 50
        Zero and non-numerical values turn off the filtering,
        by default 1000
    e_value : float, optional
        E-value threshold for inclusion in the alignment, by default 0.001.
    realign_max : int, optional
        realign max. <int> hits.
    maxfilt : int, optional
        max number of hits allowed to pass 2nd prefilter, by default 100000
    min_prefilter_hits : int, optional
        min number of hits to pass prefilter, by default 1000
    maxseq : int, optional
        max number of input rows, by default 1000000
    datasets : Union[str, List[str]], optional
        datasets to search, by default None
    input_path : Union[str, Path], optional
        input fasta file, by default None
    output_a3m_path : Union[str, Path], optional
        output a3m file, by default None
    verbose : int, optional
        verbose level, by default 2
    cpu : int, optional
        number of cpu, by default 8

    Returns
    -------
    str
        hhblits command to be
    """
    if type(datasets) is str:
        datasets = [datasets]

    command = (
        f"hhblits"
        f" -n {iteration}"
        f" -diff {diff}"
        f" -e {e_value}"
        f" -realign_max {realign_max}"
        f" -maxfilt {maxfilt}"
        f" -min_prefilter_hits {min_prefilter_hits}"
        f" -maxseq {maxseq}"
        f" -M first"
        f" {' '.join(['-d ' + dataset for dataset in datasets])}"
        f" -i {input_path}"
        f" -oa3m {output_a3m_path}"
        f" -cpu {cpu}"
        f" -v {verbose}"
    )
    return command


fast_hhblits_wrapper = partial(hhblits_wrapper, diff=1000)


def read_jobs(args):
    if args.list is None:
        samples = dtool.build_list_from_dir(args.input_dir)
    else:
        samples = dtool.read_jsonlines(args.list)
    samples = [sample for sample in samples if not is_completed(sample, args)]
    return samples


def is_completed(sample, args):
    ptree = SearchPathTree(args.output_dir, sample)
    return ptree.hhblist_bfd_uniclust_a3m.exists()
    # return ptree.hhblist_bfd_uniclust_aln.exists()


@ray.remote(num_cpus=N_CPU_PER_THREAD, num_gpus=N_GPU_PER_THREAD)
def process_jobs(jobs_queue, args, actor):
    results = []
    while not jobs_queue.empty():
        job = jobs_queue.get()
        try:
            result = execute_one_job(job, args)
            if result is not None:
                results.append(result)
        except:
            logger.info(f"failed: {job}")
            traceback.print_exception(*sys.exc_info())
        actor.update.remote(1)
    return results


def execute_one_job(sample, args):
    ptree = SearchPathTree(args.output_dir, sample)
    input_dir = Path(args.input_dir).expanduser()

    name = sample["name"].split(".")[0]
    input_path = input_dir / f"{name}.aln"

    if is_completed(sample, args):
        logger.info(f"skip {name}")
    else:
        if input_path.exists():
            # prepare input (aln -> fasta)
            fasta_input_path = ptree.in_fasta
            lines_funcs = (
                [dtool.aln2seq] if args.in_seq_only else [dtool.aln2fasta]
            )
            dtool.process_file(input_path, fasta_input_path, lines_funcs)

            # search (fasta -> a3m)
            # AlphaFold2 Supplementary 1.2.2 Genetic search
            # HHBlits:
            #   -n 3 -e 0.001 -realign_max 100000 -maxfilt 100000
            #   -min_prefilter_hits 1000 -maxseq 1000000
            ptree.hhblist_bfd_uniclust_a3m.parent.mkdir(
                parents=True, exist_ok=True
            )
            logger.info(f"searching {name}...")
            start = time.time()
            try:
                command = hhblits_wrapper(
                    iteration=args.iteration,
                    e_value=args.e_value,
                    realign_max=args.realign_max,
                    maxfilt=args.maxfilt,
                    min_prefilter_hits=args.min_prefilter_hits,
                    maxseq=args.maxseq,
                    diff=args.diff_default,
                    datasets=args.data,
                    input_path=fasta_input_path,
                    output_a3m_path=ptree.hhblist_bfd_uniclust_a3m,
                    verbose=2 if args.verbose else 0,
                    cpu=args.cpu,
                )
                execute(command, timeout=args.timeout)
            except subprocess.TimeoutExpired:
                command = hhblits_wrapper(
                    iteration=args.iteration,
                    e_value=args.e_value,
                    realign_max=args.realign_max,
                    maxfilt=args.maxfilt,
                    min_prefilter_hits=args.min_prefilter_hits,
                    maxseq=args.maxseq,
                    diff=args.diff_fast,
                    datasets=args.data,
                    input_path=fasta_input_path,
                    output_a3m_path=ptree.hhblist_bfd_uniclust_a3m,
                    verbose=2 if args.verbose else 0,
                    cpu=args.cpu,
                )
                execute(command)
            time_cost = "%.2f s" % (time.time() - start)

            # process searching results (a3m -> aln)
            dtool.process_file(
                ptree.hhblist_bfd_uniclust_a3m,
                ptree.hhblist_bfd_uniclust_aln,
                lines_funcs=[dtool.a3m2aln],
            )
            # process searching results (aln -> fasta)
            lines = dtool.process_file(
                ptree.hhblist_bfd_uniclust_a3m,
                ptree.hhblist_bfd_uniclust_fa,
                lines_funcs=[dtool.a3m2fasta],
            )

            logger.info(
                f"searching {name} completed. {ptree.hhblist_bfd_uniclust_a3m.exists()}. "
                f"{time_cost} N: {len(lines)} L: {len(lines[0])}"
            )
        else:
            logger.info(f"{input_path} does not exist!")

    if ptree.hhblist_bfd_uniclust_a3m.exists():
        result = {
            "name": name,
            "N": len(lines),
            "L": len(lines[0]),
            "Time": time_cost,
        }
        return result
    else:
        logger.info(f"{name} has no result.")
        return None


def hhblits_search(
    data: Union[Path, List[Path]],
    path_list: Path,
    input_dir: Path,
    output_dir: Path,
    in_seq_only: bool = False,
    verbose: bool = False,
    iteration: int = 3,
    e_value: float = 0.001,
    realign_max: int = 100000,
    maxfilt: int = 100000,
    min_prefilter_hits: int = 1000,
    maxseq: int = 1000000,
    diff_default: int = 1000,
    diff_fast: int = 100,
    timeout: float = 3600,
    thread: int = 5,
    cpu: int = 8,
    gpu: int = 0,
) -> List[Dict[str, Any]]:
    """Search MSA using HHblits.

    Parameters
    ----------
    data : Path
        _description_
    path_list : Path
        _description_
    input_dir : Path
        _description_
    output_dir : Path
        _description_
    in_seq_only : bool, optional
        _description_, by default False
    verbose : bool, optional
        _description_, by default False
    thread : int, optional
        _description_, by default 1
    cpu : int, optional
        _description_, by default 1
    gpu : int, optional
        _description_, by default 0
    timeout : float, optional
        _description_, by default 3600
    Returns
    -------
    List[Dict[str, Any]]
        _description_
    """
    args = argparse.Namespace(
        data=data,
        list=path_list,
        input_dir=input_dir,
        output_dir=output_dir,
        in_seq_only=in_seq_only,
        verbose=verbose,
        iteration=iteration,
        e_value=e_value,
        realign_max=realign_max,
        maxfilt=maxfilt,
        min_prefilter_hits=min_prefilter_hits,
        maxseq=maxseq,
        diff_default=diff_default,
        diff_fast=diff_fast,
        timeout=timeout,
        thread=thread,
        cpu=cpu,
        gpu=gpu,
    )
    hhblits_search_run(args)


def hhblits_search_run(args):
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    jobs = read_jobs(args)
    if len(jobs) > 0:
        logger.info(f"-----------------")
        logger.info(f"Total jobs: {len(jobs)}")

        N_CPU_PER_THREAD = args.cpu
        N_GPU_PER_THREAD = args.gpu
        n_thread = min(len(jobs), args.thread)
        total_cpus = N_CPU_PER_THREAD * n_thread
        total_gpus = N_GPU_PER_THREAD * n_thread
        ray.init(num_cpus=total_cpus, num_gpus=total_gpus)
        logger.info(f"Number of threads: {n_thread}.")
        logger.info(f"CPUs per thread: {N_CPU_PER_THREAD}.")
        logger.info(f"Total CPUs: {total_cpus}.")
        if N_GPU_PER_THREAD > 0:
            logger.info(f"GPUs per thread: {N_GPU_PER_THREAD}.")
            logger.info(f"Total GPUs: {total_gpus}.")
        logger.info(f"-----------------")

        jobs_queue = Queue()
        for job in jobs:
            jobs_queue.put(job)
        pb = ProgressBar(len(jobs_queue))
        actor = pb.actor
        job_list = []
        for _ in range(n_thread):
            job_list.append(process_jobs.remote(jobs_queue, args, actor))

        pb.print_until_done()
        job_results = list(chain(*ray.get(job_list)))
        if len(job_results) > 0:
            mean_time = np.round(
                np.mean([float(j["Time"][:-2]) for j in job_results]), 2
            )
            mean_number = np.round(
                np.mean([float(j["N"]) for j in job_results]), 2
            )
            mean_length = np.round(
                np.mean([float(j["L"]) for j in job_results]), 2
            )
            job_results = [
                {
                    "mean_N": mean_number,
                    "mean_L": mean_length,
                    "mean_Time": mean_time,
                }
            ] + job_results
        ray.get(actor.get_counter.remote())

        result_path = Path(args.output_dir) / "result.jsonl"
        with jsonlines.open(result_path, "w") as writer:
            writer.write_all(job_results)
    else:
        logger.info(f"No job to run.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--data", type=str, nargs="+", required=True)
    parser.add_argument("-l", "--list", type=str, required=True)
    parser.add_argument("-i", "--input_dir", type=str, required=True)
    parser.add_argument("-o", "--output_dir", type=str, required=True)

    parser.add_argument(
        "--in_seq_only",
        action="store_true",
        help="search only bases on the target",
    )
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument(
        "--iteration", type=int, default=3, help="number of iteration"
    )
    parser.add_argument(
        "--e_value", type=float, default=0.001, help="e-value threshold"
    )
    parser.add_argument(
        "--realign_max", type=int, default=100000, help="max number of realign"
    )
    parser.add_argument(
        "--maxfilt", type=int, default=100000, help="max number of filter"
    )
    parser.add_argument(
        "--min_prefilter_hits",
        type=int,
        default=1000,
        help="min prefilter hits",
    )
    parser.add_argument(
        "--maxseq", type=int, default=1000000, help="max number of sequences"
    )

    parser.add_argument(
        "--diff_default", type=str, default="inf", help="default diff value"
    )
    parser.add_argument(
        "--diff_fast", type=str, default="1000", help="fast diff value"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=7200,
        help=(
            "maximum seconds to search, "
            "change diff from default to fast when timeout"
        ),
    )
    parser.add_argument(
        "--thread", type=int, default=5, help="number of threads"
    )
    parser.add_argument(
        "--cpu", type=int, default=8, help="number of cpu per thread"
    )
    parser.add_argument(
        "--gpu", type=float, default=0, help="number of gpu per thread"
    )

    args = parser.parse_args()
    hhblits_search_run(args)
