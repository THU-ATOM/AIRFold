"""Library to run Jackhmmer from Python."""

from concurrent import futures
import glob
import os
import subprocess
from typing import Any, Callable, Mapping, Optional, Sequence
from urllib import request

from absl import logging
import argparse
from itertools import chain
import numpy as np
import jsonlines

from loguru import logger


import ray
from ray.util.queue import Queue
import traceback
import subprocess

import time
import sys

# from lib.tool import utils
from lib.tool.hhblits import N_CPU_PER_THREAD, N_GPU_PER_THREAD
from lib.utils.ray_tools import ProgressBar
from lib.utils.execute import execute
import lib.utils.datatool as dtool
from pathlib import Path

from lib.pathtree import SearchPathTree

# TODO fix parser and utils

from lib.tool.parsers import parse_a3m, parse_fasta
from lib.tool import tool_utils


class Jackhmmer:
    """Python wrapper of the Jackhmmer binary."""

    def __init__(
        self,
        *,
        binary_path: str,
        database_path: str,
        sto_path: str,
        n_cpu: int = 8,
        n_iter: int = 1,
        e_value: float = 0.0001,
        z_value: int = 135301051,
        get_tblout: bool = False,
        filter_f1: float = 0.0005,
        filter_f2: float = 0.00005,
        filter_f3: float = 0.0000005,
        incdom_e: Optional[float] = None,
        dom_e: Optional[float] = None,
        num_streamed_chunks: Optional[int] = None,
        streaming_callback: Optional[Callable[[int], None]] = None,
    ):
        """Initializes the Python Jackhmmer wrapper.

        Args:
          binary_path: The path to the jackhmmer executable.
          database_path: The path to the jackhmmer database (FASTA format).
          n_cpu: The number of CPUs to give Jackhmmer.
          n_iter: The number of Jackhmmer iterations.
          e_value: The E-value, see Jackhmmer docs for more details.
          z_value: The Z-value, see Jackhmmer docs for more details.
          get_tblout: Whether to save tblout string.
          filter_f1: MSV and biased composition pre-filter, set to >1.0 to turn off.
          filter_f2: Viterbi pre-filter, set to >1.0 to turn off.
          filter_f3: Forward pre-filter, set to >1.0 to turn off.
          incdom_e: Domain e-value criteria for inclusion of domains in MSA/next
            round.
          dom_e: Domain e-value criteria for inclusion in tblout.
          num_streamed_chunks: Number of database chunks to stream over.
          streaming_callback: Callback function run after each chunk iteration with
            the iteration number as argument.
        """
        self.binary_path = binary_path
        self.database_path = database_path
        self.num_streamed_chunks = num_streamed_chunks

        if (
            not os.path.exists(self.database_path)
            and num_streamed_chunks is None
        ):
            logging.error("Could not find Jackhmmer database %s", database_path)
            raise ValueError(
                f"Could not find Jackhmmer database {database_path}"
            )

        self.n_cpu = n_cpu
        self.n_iter = n_iter
        self.e_value = e_value
        self.z_value = z_value
        self.filter_f1 = filter_f1
        self.filter_f2 = filter_f2
        self.filter_f3 = filter_f3
        self.incdom_e = incdom_e
        self.dom_e = dom_e
        self.get_tblout = get_tblout
        self.streaming_callback = streaming_callback
        self.sto_path = sto_path

    def get_command(self, input_fasta_path: str):
        # get the
        logger.info(self.sto_path)
        readable_output = (
            Path(self.sto_path).parent
            / "readable_output"
            / f"{Path(input_fasta_path).stem}.json"
        )
        (Path(self.sto_path).parent / "readable_output").mkdir(
            parents=True, exist_ok=True
        )
        # readable_output = os.path.join(str(Path(self.sto_path).parent), '/jackhmmer')
        #
        cmd_flags = [
            # Don't pollute stdout with Jackhmmer output.
            "-o",
            str(readable_output),
            "-A",
            str(self.sto_path),
            "--noali",
            "--F1",
            str(self.filter_f1),
            "--F2",
            str(self.filter_f2),
            "--F3",
            str(self.filter_f3),
            "--incE",
            str(self.e_value),
            # Report only sequences with E-values <= x in per-sequence output.
            "-E",
            str(self.e_value),
            # '--cpu', str(self.n_cpu),
            "-N",
            str(self.n_iter),
        ]
        if self.get_tblout:
            tblout_path = os.path.join(
                str(Path(self.sto_path).parent), "tblout.txt"
            )
            cmd_flags.extend(["--tblout", tblout_path])

        if self.z_value:
            cmd_flags.extend(["-Z", str(self.z_value)])

        if self.dom_e is not None:
            cmd_flags.extend(["--domE", str(self.dom_e)])

        if self.incdom_e is not None:
            cmd_flags.extend(["--incdomE", str(self.incdom_e)])

        cmd = (
            [str(self.binary_path)]
            + cmd_flags
            + [str(input_fasta_path), str(self.database_path)]
        )

        # note jackhmmer

        return " ".join(cmd)

    def _query_chunk(
        self, input_fasta_path: str, database_path: str
    ) -> Mapping[str, Any]:
        """Queries the database chunk using Jackhmmer."""

        with tool_utils.tmpdir_manager(base_dir="/tmp") as query_tmp_dir:
            sto_path = os.path.join(query_tmp_dir, "output.sto")
            # The F1/F2/F3 are the expected proportion to pass each of the filtering
            # stages (which get progressively more expensive), reducing these
            # speeds up the lib at the expensive of sensitivity.  They are
            # currently set very low to make querying Mgnify run in a reasonable
            # amount of time.
            cmd_flags = [
                # Don't pollute stdout with Jackhmmer output.
                "-o",
                "/dev/null",
                "-A",
                sto_path,
                "--noali",
                "--F1",
                str(self.filter_f1),
                "--F2",
                str(self.filter_f2),
                "--F3",
                str(self.filter_f3),
                "--incE",
                str(self.e_value),
                # Report only sequences with E-values <= x in per-sequence output.
                "-E",
                str(self.e_value),
                "--cpu",
                str(self.n_cpu),
                "-N",
                str(self.n_iter),
            ]
            if self.get_tblout:
                tblout_path = os.path.join(query_tmp_dir, "tblout.txt")
                cmd_flags.extend(["--tblout", tblout_path])

            if self.z_value:
                cmd_flags.extend(["-Z", str(self.z_value)])

            if self.dom_e is not None:
                cmd_flags.extend(["--domE", str(self.dom_e)])

            if self.incdom_e is not None:
                cmd_flags.extend(["--incdomE", str(self.incdom_e)])

            cmd = (
                [self.binary_path]
                + cmd_flags
                + [input_fasta_path, database_path]
            )

            logging.info('Launching subprocess "%s"', " ".join(cmd))
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            with tool_utils.timing(
                f"Jackhmmer ({os.path.basename(database_path)}) query"
            ):
                _, stderr = process.communicate()
                retcode = process.wait()

            if retcode:
                raise RuntimeError(
                    "Jackhmmer failed\nstderr:\n%s\n" % stderr.decode("utf-8")
                )

            # Get e-values for each target name
            tbl = ""
            if self.get_tblout:
                with open(tblout_path) as f:
                    tbl = f.read()

            with open(sto_path) as f:
                sto = f.read()

        raw_output = dict(
            sto=sto,
            tbl=tbl,
            stderr=stderr,
            n_iter=self.n_iter,
            e_value=self.e_value,
        )

        return raw_output

    def query(self, input_fasta_path: str) -> Sequence[Mapping[str, Any]]:
        """Queries the database using Jackhmmer."""
        if self.num_streamed_chunks is None:
            return [self._query_chunk(input_fasta_path, self.database_path)]

        db_basename = os.path.basename(self.database_path)
        db_remote_chunk = lambda db_idx: f"{self.database_path}.{db_idx}"
        db_local_chunk = lambda db_idx: f"/tmp/ramdisk/{db_basename}.{db_idx}"

        # Remove existing files to prevent OOM
        for f in glob.glob(db_local_chunk("[0-9]*")):
            try:
                os.remove(f)
            except OSError:
                logger.info(f"OSError while deleting {f}")

        # Download the (i+1)-th chunk while Jackhmmer is running on the i-th chunk
        with futures.ThreadPoolExecutor(max_workers=2) as executor:
            chunked_output = []
            for i in range(1, self.num_streamed_chunks + 1):
                # Copy the chunk locally
                if i == 1:
                    future = executor.submit(
                        request.urlretrieve,
                        db_remote_chunk(i),
                        db_local_chunk(i),
                    )
                if i < self.num_streamed_chunks:
                    next_future = executor.submit(
                        request.urlretrieve,
                        db_remote_chunk(i + 1),
                        db_local_chunk(i + 1),
                    )

                # Run Jackhmmer with the chunk
                future.result()
                chunked_output.append(
                    self._query_chunk(input_fasta_path, db_local_chunk(i))
                )

                # Remove the local copy of the chunk
                os.remove(db_local_chunk(i))
                future = next_future
                if self.streaming_callback:
                    self.streaming_callback(i)
        return chunked_output


def read_jobs(args):
    if args.list is None:
        samples = dtool.build_list_from_dir(args.input_fasta)
    else:
        samples = dtool.read_jsonlines(args.list)
    samples = [sample for sample in samples if not is_completed(sample, args)]
    return samples


def is_completed(sample, args):
    ptree = SearchPathTree(args.output_dir, sample)
    if "uniref90" in args.database_path:
        return ptree.jackhammer_uniref90_a3m.exists()
    elif "mgnify" in args.database_path:
        return ptree.jackhammer_mgnify_a3m.exists()
    else:
        logger.info("Unknown database")
        return None
        # database_path = args.database_path
    # return ptree.hhblist_bfd_uniclust_a3m.exists()


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
    name = sample["name"].split(".")[0]
    input_dir = Path(args.input_dir).expanduser()

    input_path = input_dir / f"{name}.aln"

    if is_completed(sample, args):
        logger.info(f"skip {name}")
    else:
        if input_path.exists():
            fasta_input_path = ptree.in_fasta
            lines_funcs = (
                [dtool.aln2seq] if args.in_seq_only else [dtool.aln2fasta]
            )
            dtool.process_file(input_path, fasta_input_path, lines_funcs)
            if "uniref90" in args.database_path:
                sto_path = ptree.jackhammer_uniref90_sto
                # Path(args.output_dir).expanduser() / f"{name}.sto"
                ptree.jackhammer_uniref90_a3m.parent.mkdir(
                    parents=True, exist_ok=True
                )
                ptree.jackhammer_uniref90_fa.parent.mkdir(
                    parents=True, exist_ok=True
                )
                tmp_path = ptree.jackhammer_uniref90_a3m
            elif "mgnify" in args.database_path:
                sto_path = ptree.jackhammer_mgnify_sto
                ptree.jackhammer_mgnify_a3m.parent.mkdir(
                    parents=True, exist_ok=True
                )
                ptree.jackhammer_mgnify_fa.parent.mkdir(
                    parents=True, exist_ok=True
                )
                tmp_path = ptree.jackhammer_mgnify_a3m
            else:
                logger.info(f"no such database {args.database_path}")
                return None
            logger.info(f"Jackhmmer searching {name}...")
            start = time.time()
            try:
                cmd = Jackhmmer(
                    binary_path=args.binary_path,
                    database_path=args.database_path,
                    sto_path=sto_path,
                    n_iter=args.n_iter,
                    e_value=args.e_value,
                    z_value=args.z_value,
                    get_tblout=args.get_tblout,
                    filter_f1=args.filter_f1,
                    filter_f2=args.filter_f2,
                    filter_f3=args.filter_f3,
                    #  incdom_e=args.incdom_e,
                    #  dom_e=args.dom_e,
                ).get_command(fasta_input_path)
                logger.info(f"generated {cmd}")

                execute(cmd, timeout=args.timeout)
            except subprocess.TimeoutExpired:
                logger.info(f"Timeout: {name}")
                return
            time_cost = "%.2f s" % (time.time() - start)

            # post-process (sto -> a3m)
            if "uniref90" in args.database_path:
                lines = parse_a3m(sto_path, ptree.jackhammer_uniref90_a3m)
                parse_fasta(sto_path, ptree.jackhammer_uniref90_fa)
            if "mgnify" in args.database_path:
                lines = parse_a3m(sto_path, ptree.jackhammer_mgnify_a3m)
                parse_fasta(sto_path, ptree.jackhammer_mgnify_fa)

            logger.info(
                f"searching {name} completed. {tmp_path.exists()}. "
                f"{time_cost} N: {len(lines.items())}"
            )
        else:
            logger.info(f"{input_path} does not exist!")

    if Path(sto_path).exists():
        result = {
            "name": name,
            "N": len(lines),
            "Time": time_cost,
        }
        return result
    else:
        logger.info(f"{name} has no result.")
        return None


def jackhammer_run(args):
    # use parameter args.output_dir
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    jobs = read_jobs(args)
    # for debugging
    logger.info(jobs)

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
            # mean_length = np.round(
            #     np.mean([float(j["L"]) for j in job_results]), 2
            # )
            job_results = [
                {
                    "mean_N": mean_number,
                    # "mean_L": mean_length,
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b",
        "--binary_path",
        default="jackhmmer",
        type=str,
    )
    parser.add_argument("-d", "--database_path", type=str, required=True)
    # TODO: modift the stopath
    # parser.add_argument("-s", "--sto_path", type=str, required=True)
    parser.add_argument("-l", "--list", type=str, required=True)
    parser.add_argument(
        "-o", "--output_dir", type=str, required=True, help="output directory"
    )
    # the output_dir is essentially the root directory of the search path tree.
    parser.add_argument("-i", "--input_dir", type=str, required=True)
    # the input dir has the fasta files.

    parser.add_argument("-n", "--n_iter", type=int, default=1)
    parser.add_argument("-e", "--e_value", type=float, default=1e-4)
    # update the z_value for different datasets
    parser.add_argument("-z", "--z_value", type=int, default=135301051)
    parser.add_argument("-g", "--get_tblout", type=bool, default=False)
    parser.add_argument("-f1", "--filter_f1", type=float, default=0.0005)
    parser.add_argument("-f2", "--filter_f2", type=float, default=0.00005)
    parser.add_argument("-f3", "--filter_f3", type=float, default=0.000005)
    parser.add_argument(
        "-t",
        "--timeout",
        type=float,
        default=float("inf"),
        help="maximum seconds for searching one MSA",
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

    parser.add_argument(
        "--in_seq_only",
        action="store_true",
        help="search only bases on the target",
    )

    args = parser.parse_args()
    jackhammer_run(args)
    # parser.add_argument("-z", "--z_value", type=float, default=0.05)
