"""
Input files & format
----------
1. args.list: sample list file (.jsonl)
```
{name: xxx1, ...}
{name: xxx2, ...}
...
```

2. args.seq_dir: folder with many files (<name>.fasta), each contains only one
sequence corresponding to one line in sample list file. 
```
> ID, ...meta infomation
ABCABC...
```

3. args.msa_dir: folder with many files (<name>.fasta), each contains mutliple
sequence alignments corresponding to one line in sample list file.
```
> ID, ...meta infomation
ABCABC...
> ID, ...meta infomation
ABCABC...
> ID, ...meta infomation
ABCABC...
....
```

4. args.pdb_dir (optional): folder with many files (<name>.pdb), each contains the
reference strucutre information, are used for computing lDDT.

Output files
-------

"""

from functools import partial
import os
import argparse
import pickle
import random
import sys
import time
import traceback
from itertools import chain
from pathlib import Path
from typing import Union, Dict, Any
from lib.utils.systool import get_available_gpus

import ray
from ray.util.queue import Queue
from lib.constant import COLABFOLD_PYTHON_PATH, AF_PARAMS_ROOT
from lib.base import BasePathTree

from loguru import logger

import lib.utils.datatool as dtool
from lib.tool import metrics
from lib.utils.execute import (
    cuda_visible_devices_wrapper,
    execute,
    rlaunch_wrapper,
    rlaunch_exists,
)
from lib.utils.timetool import time2str, with_time


RUNNER_SCRIPT_PATH = Path(__file__).resolve().parent / "alphafold_runner.py"


class AlphaFoldPathTree(BasePathTree):
    def __init__(self, root: Union[str, Path], request: Dict[str, Any]) -> None:
        super().__init__(root, request)
        self.root = self.root / self.id

    @property
    def time_cost(self):
        return self.root / "time_cost.txt"

    @property
    def template_feat(self):
        return self.root / "template_feat.pkl"

    @property
    def relaxed_pdbs(self):
        return list(sorted(self.root.glob("rank_*_relaxed.pdb")))

    @property
    def submit_pdbs(self):
        return list(sorted(self.root.glob("model_*_relaxed.pdb")))

    @property
    def unrelaxed_pdbs(self):
        return list(sorted(self.root.glob("*_unrelaxed.pdb")))

    @property
    def result(self):
        return self.root / "result.json"

    @property
    def lddt(self):
        return self.root / "lddt" / "lddt.json"

    @property
    def msa_pickle(self):
        return self.root / "msa.pickle"

    @property
    def input_a3m(self):
        return self.root / "input_msa.a3m"

    @property
    def msa_filtered_pickle(self):
        return self.root / "msa_filtered.pickle"

    @property
    def log(self):
        return self.root / "log.txt"

    @property
    def plddt_image(self):
        return self.root / "predicted_LDDT.png"

    @property
    def msa_coverage_image(self):
        return self.root / "msa_coverage.png"

    @property
    def model_files(self):
        files = []
        for item in self.relaxed_pdbs:
            key = "_".join(item.name.split("_")[:-1])
            itemfiles = {}
            model_key = "_".join(item.name.split("_")[2:4])
            itemfiles["relaxed_pdb"] = item
            itemfiles["unrelaxed_pdb"] = self.root / f"{key}_unrelaxed.pdb"
            itemfiles["plddt"] = self.root / model_key / "result.json"
            itemfiles["image"] = self.root / f"{key}.png"
            files.append(itemfiles)
        if len(files) > 1:
            files = sorted(files, key=lambda x: x["relaxed_pdb"])
        return files


def get_plddt(ptree: AlphaFoldPathTree):
    plddts = {}
    for files in ptree.model_files:
        key = "_".join(files["relaxed_pdb"].stem.split("_")[:-1])
        if files["plddt"].exists():
            plddts[key] = dtool.read_json(files["plddt"])["plddt"]
    return plddts


def t_print(message=None, thread=None):
    logger.info(f"[{thread}] {message}")


def read_jobs(args):
    samples = dtool.read_jsonlines(args.list)
    return samples


def is_alphafold_completed(sample, args):
    ptree = AlphaFoldPathTree(args.output_dir, sample)
    for pdb in ptree.relaxed_pdbs:
        if pdb.name.startswith("rank_1"):
            return True and ptree.time_cost.exists()
    return False


def is_summarize_completed(sample, args):
    if args.force_sum:
        return False
    ptree = AlphaFoldPathTree(args.output_dir, sample)
    return ptree.result.exists()


def execute_one_job(thread, sample, args=None):
    print = partial(t_print, thread=thread)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    seq_dir = Path(args.seq_dir).expanduser()
    msa_dir = Path(args.msa_dir).expanduser()
    ptree = AlphaFoldPathTree(args.output_dir, sample)

    """
    TODO: double check whether this part should be reorginized. If so
    """

    name = sample["name"].split(".")[0]
    seq_path = seq_dir / f"{name}.fasta"
    msa_path = msa_dir / f"{name}.fasta"

    ptree.root.mkdir(exist_ok=True, parents=True)

    # Run AlphaFold2
    alphafold_completed = is_alphafold_completed(sample, args)
    alphafold_time_cost = -1
    if alphafold_completed or args.sum_only:
        print(f"skip running alphafold: {name}")
    else:
        if msa_path.exists():
            time.sleep(random.random())
            memory = 50000 * (2 ** sample["_try"])
            print(f">>>>>>>>>> runing alphafold: {name}")
            command = (
                f"{COLABFOLD_PYTHON_PATH} {RUNNER_SCRIPT_PATH} "
                f"--params_loc {args.params_loc} "
                f"-seqn {name} "
                f"-i {seq_path} "
                f"-cm {msa_path} "
                f"-o {ptree.root} "
                f"-a_m True "
                f"-sc {args.seqcov} "
                f"-sq {args.seqqid} "
                f"-r {args.max_recycles} "
            )
            if rlaunch_exists():
                command = rlaunch_wrapper(
                    command,
                    cpu=6,
                    gpu=1,
                    memory=memory,
                    charged_group=args.group,
                )
            else:
                command = cuda_visible_devices_wrapper(
                    command, device_ids=get_available_gpus(num=1)
                )
            _, alphafold_time_cost = with_time(execute)(
                command, verbose=True, print_off=False, log_path=ptree.log
            )
            dtool.write_lines(ptree.time_cost, [str(alphafold_time_cost)])

            alphafold_completed = is_alphafold_completed(sample, args)
            print(
                f"<<<<<<<<<< Running {name} completed. {alphafold_completed}. "
                f"{time2str(alphafold_time_cost)}."
            )
        else:
            print(f"{msa_path} does not exist!")

    # Summarize results
    if alphafold_completed:
        summarize_completed = is_summarize_completed(sample, args)
        if summarize_completed and not args.force_sum:
            print(f"skip summarize: {name}")
            result = dtool.read_json(ptree.result)
        else:
            print(f">>>>>>>>>> summarizing: {name}")
            result = {
                "name": name,
                "time": alphafold_time_cost
                if alphafold_time_cost > 0
                else float(dtool.read_lines(ptree.time_cost)[0]),
            }
            result.update(describe_sample(ptree.root))

            # compute lddt
            if args.pdb_dir:
                gt_pdb_path = Path(args.pdb_dir).expanduser() / f"{name}.pdb"
                if gt_pdb_path:
                    lddts = compute_save_lddt(gt_pdb_path, ptree.root)
                    result["lddts"] = lddts
                else:
                    print(f"missing reference pdb files: {gt_pdb_path}")
            dtool.write_json(ptree.result, result)
            summarize_completed = is_summarize_completed(sample, args)
            print(f">>>>>>>>>> Summarizing {name} completed. {summarize_completed}. ")
        return result
    else:
        print(f"{name} is not ready to be summarized.")
        return None


def compute_save_lddt(gt_pdb_path, sample_output_dir):
    lddts = {}
    for relaxed_pdb in list(sample_output_dir.glob("*_relaxed.pdb")):
        lddt, report = metrics.compute_lddt(relaxed_pdb, gt_pdb_path)
        lddts[relaxed_pdb.stem] = lddt
        path_report = sample_output_dir / "lddt" / f"{relaxed_pdb.stem}.text"
        dtool.write_lines(path_report, [report])
    dtool.write_json(sample_output_dir / "lddt" / "lddt.json", lddts)
    return lddts


def describe_sample(sample_output_dir):
    meta = {}

    msa_pickle_path = sample_output_dir / "msa.pickle"
    if msa_pickle_path.exists():
        with open(msa_pickle_path, "rb") as f:
            data = pickle.load(f)
        N = len(data["msas"][0])
        L = len(data["msas"][0][0])
        meta.update({"L": L, "N": N})

    msa_filtered_pickle_path = sample_output_dir / "msa_filtered.pickle"
    if msa_filtered_pickle_path.exists():
        with open(msa_filtered_pickle_path, "rb") as f:
            data = pickle.load(f)
        filter_N = len(data["msas"][0])
        meta.update({"filter_N": filter_N})

    return meta


@ray.remote(num_cpus=0.1)
def process_jobs(thread, queue, args, actor=None):
    results = []
    print = partial(t_print, thread=thread)
    print("start.")
    while not queue["jobs"].empty():
        job = queue["jobs"].get()
        job["_try"] += 1
        result = None
        try:
            queue["working"].put({})
            result = execute_one_job(thread, job, args)
            print(f"job execution done 1 : {job}")
            if result is not None:
                results.append(result)
                queue["results"].put(result)
        except:
            print(f"failed: {job}")
            traceback.print_exception(*sys.exc_info())
        finally:
            if not queue["working"].empty():
                queue["working"].get()
            msg = ""
            if result is not None:
                msg += f"success: {job}."
                if actor is not None:
                    actor.update.remote(1)
            else:
                if job["_try"] < args.max_try:
                    msg += f"failed {job['_try']} times, but will try agiain: {job}."
                    queue["jobs"].put(job)
                else:
                    msg += f"failed {job['_try']} times: {job}."
                    if actor is not None:
                        actor.update.remote(1)
            msg += (
                f" {len(queue['jobs'])} jobs remain. {len(queue['working'])} running."
            )
            print(msg)

    print("exit.")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("-l", "--list", type=str, required=True)
    parser.add_argument("-s", "--seq_dir", type=str, required=True)
    parser.add_argument("-m", "--msa_dir", type=str, required=True)
    parser.add_argument("-p", "--pdb_dir", type=str)
    parser.add_argument("-o", "--output_dir", type=str, required=True)

    parser.add_argument("--params_loc", default=AF_PARAMS_ROOT)
    parser.add_argument(
        "--seqcov", type=int, default=0, help="minimum coverage with target"
    )
    parser.add_argument(
        "--seqqid",
        type=int,
        default=0,
        help="minimum sequence identity with target",
    )
    parser.add_argument(
        "--max_recycles",
        default=3,
        type=int,
        help="controls the maximum number of times the structure is fed back into the neural network for refinement. (default is 3)",
    )

    parser.add_argument(
        "-j", "--thread", type=int, default=-1, help="number of threads"
    )
    parser.add_argument(
        "--max_try",
        type=int,
        default=3,
        help="max retry times for failed jobs",
    )
    parser.add_argument(
        "--sum_only",
        action="store_true",
        help="only summarize results",
    )
    parser.add_argument(
        "--force_sum",
        action="store_true",
        help="force to re-summarize all results",
    )

    parser.add_argument("--group", default="health", help="rlaunch group")
    args = parser.parse_args()

    jobs = read_jobs(args)
    if len(jobs) > 0:
        logger.info(f"-----------------")
        logger.info(f"Total jobs: {len(jobs)}")
        n_thread = min(args.thread, len(jobs)) if args.thread > 0 else len(jobs)
        ray.init()
        logger.info(f"Number of threads: {n_thread}.")
        logger.info(f"-----------------")

        queue = {"jobs": Queue(), "results": Queue(), "working": Queue()}
        for job in jobs:
            job["_try"] = 0
            queue["jobs"].put(job)
        # pb = ProgressBar(len(queue["jobs"]))
        # actor = pb.actor
        actor = None
        job_list = []
        for i in range(n_thread):
            job_list.append(process_jobs.remote(i, queue, args, actor))

        try:
            # pb.print_until_done()
            _ = list(chain(*ray.get(job_list)))
            # ray.get(actor.get_counter.remote())
        except KeyboardInterrupt:
            logger.info("Keyboard Interrupt.")
        except:
            traceback.print_exception(*sys.exc_info())
        finally:
            path_results = Path(args.output_dir) / "result.jsonl"
            logger.info(f"Writing results... {path_results}")
            jobs_results = []
            while not queue["results"].empty():
                jobs_results.append(queue["results"].get())
            dtool.write_jsonlines(path_results, jobs_results)

            logger.info("Shuting down all workers.")
            ray.shutdown()
    else:
        logger.info(f"No job to run.")
