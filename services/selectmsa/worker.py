import os
import glob
import json
from celery import Celery

from pathlib import Path
from typing import Any, Dict, List
from loguru import logger

from lib.base import BaseCommandRunner
from lib.state import State
from lib.pathtree import get_pathtree
from lib.monitor import info_report
from lib.utils.execute import rlaunch_exists, rlaunch_wrapper
from lib.utils import misc
import lib.utils.datatool as dtool
from lib.strategy import *

SEQUENCE = "sequence"
TARGET = "target"

CELERY_RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", "rpc://")
CELERY_BROKER_URL = (
    os.environ.get("CELERY_BROKER_URL", "pyamqp://guest:guest@localhost:5672/"),
)

celery = Celery(
    __name__,
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
)

celery.conf.task_routes = {
    "worker.*": {"queue": "queue_selectmsa"},
}


@celery.task(name="selectmsa")
def selectmsaTask(requests: List[Dict[str, Any]]):
    MSASelectRunner(requests=requests)()


class MSASelectRunner(BaseCommandRunner):
    """
    TODO not sure the error code throw part
    """

    def __init__(
        self,
        requests: List[Dict[str, Any]],
        cpu: int = 4,
    ) -> None:
        super().__init__(requests)
        self.cpu = cpu
        self.success_code = State.SELECT_SUCCESS
        self.error_code = State.SELECT_ERROR

    @property
    def start_stage(self) -> int:
        return State.SELECT_START

    @staticmethod
    def mk_cmd(least_seqs, input_path, output_path, input_fasta_path, executed_file, method_):
        params = []
        params.append(f"--input_a3m_path {input_path} ")
        params.append(f"--output_a3m_path {output_path} ")
        
        if method_ == "seq_entropy":
            params.append(f"--least_seqs {least_seqs} ")
        if method_ == "plm_similarity":
            params.append(f"--least_seqs {least_seqs} ")
            params.append(f"--input_fasta_path {input_fasta_path} ")
            
        command = f"python {executed_file} " + "".join(params)
        if rlaunch_exists():
            command = rlaunch_wrapper(command, cpu=4, gpu=0, memory=5000)
        return command
    
    @staticmethod
    def merge_segmented_a3m_files(target_path, target_sequence, target="input_0"):
        target_len = len(target_sequence)
        target_path = str(target_path)
        segmented_paths = glob.glob(f"{os.path.splitext(target_path)[0]}@*.a3m")
        logger.info(f"segmented_paths:{segmented_paths}")
        collections = [f">{target}", target_sequence]
        if Path(target_path).exists():
            with open(target_path) as fd:
                for line in fd.readlines()[2:]:
                    collections.append(line.strip())
        for path in segmented_paths:
            _, _, start, _, end = os.path.basename(path).replace(".a3m", "").split("@")
            start, end = int(start), int(end)
            logger.info(f"padding a3m file: {path}, with start={start} end={end}")
            with open(path) as fd:
                buff = fd.read()
            paded = lambda line: "".join(
                ["-" * start, line, "-" * (target_len - end - 1)]
            )
            for line in buff.strip().split("\n")[2:]:
                collections.append(paded(line) if not line.startswith(">") else line)
        with open(target_path, "w") as fd:
            wstring = "\n".join(collections)
            fd.write(wstring)
        return target_path
    
    @staticmethod
    def merge_a3m_files(
        target_path,
        msa_paths: Dict[str, str],
        target_sequence,
        max_seq_per_file=100000,
        target="input_0",
    ):
        collections = [f">{target}", target_sequence]

        statistics = {}
        for name, path in msa_paths.items():
            with open(path) as fd:
                buff = fd.read()
            lines = buff.strip().split("\n")[2:][:max_seq_per_file]
            lines = [
                f"{line} src={name}" if line.startswith(">") else line for line in lines
            ]
            collections.extend(lines)
            statistics[name] = len(lines) // 2
        logger.info(f"{target} search statistics: {json.dumps(statistics)}")
        with open(target_path, "w") as fd:
            wstring = "\n".join(collections)
            fd.write(wstring)
        return True
    
    def build_command(self, request: Dict[str, Any]) -> list:
        ptree = get_pathtree(request=request)
        select_args = misc.safe_get(request, ["run_config", "msa_select"])
        search_args = misc.safe_get(request, ["run_config", "msa_search"])
        
        
        key_list = list(select_args.keys())
        # input fasta dir not compatible with a3m
        # input_path = self.input_path
        command_list = []
        self.output_paths = []
        self.msa_paths={}
        
        for idx in range(len(key_list)):
            """TODO current implementation could not be compatible with ABA-like strategy"""
            
            method_ = key_list[idx]
            logger.info(f"The Method for msa selection: {method_}")
            executed_file = (Path(__file__).resolve().parent / "lib" / "strategy" / f"{method_}.py")
            input_fasta_path = str(ptree.seq.fasta)
            output_prefix = ptree.strategy.strategy_list[idx]
            output_prefix.parent.mkdir(exist_ok=True, parents=True)
            self.output_prefix = str(output_prefix)
            
            ls_dict = misc.safe_get(select_args, [method_, "least_seqs"])
            for tag in ls_dict.keys():
                
                least_seqs = ls_dict[tag]
                if tag == "hj" and "hhblits" in search_args.keys() and "jackhmmer" in search_args.keys():
                    input_path = str(ptree.search.integrated_search_hj_a3m_dp)
                if tag == "bl" and "blast" in search_args.keys():
                    input_path = str(ptree.search.integrated_search_bl_a3m_dp)
                if tag == "dq" and "deepmsa" in search_args.keys():
                    input_path = str(ptree.search.integrated_search_dq_a3m_dp)
                if tag == "dm" and "deepmsa" in search_args.keys():
                    input_path = str(ptree.search.integrated_search_dm_a3m_dp)
                if tag == "mm" and "mmseqs" in search_args.keys():
                    input_path = str(ptree.search.integrated_search_mm_a3m_dp)
                    
                output_path = self.output_prefix + "_" + tag + ".a3m"
                logger.info(f"{tag} ---- The output file for msa selection procedure: {output_path}")
                self.msa_paths[tag] = output_path
                self.output_paths.append(output_path)
                command = self.mk_cmd(least_seqs, input_path, output_path, input_fasta_path, executed_file, method_)
                command_list.append(command)
        return "&& ".join(command_list)

    def run(self, dry=False):
        # Check if the integrated_search_a3m file exists or not!

        config = self.requests[0]["run_config"]["msa_select"]
        if "idle" in config:
            logger.info("No MSA selection, skip!")
        else:
            super().run(dry)
            segment_merge = lambda path: self.merge_segmented_a3m_files(
                target_path=path,
                target_sequence=self.requests[0][SEQUENCE],
                target=self.requests[0][TARGET],
            )
            merge_msa_paths = {}
            for key, val in self.msa_paths.items():
                merge_msa_paths[key] = segment_merge(val)
            integrated_selected_a3m = self.output_prefix + ".a3m"
            integrated_selected_a3m_dp = self.output_prefix + "_dp.a3m"

            self.merge_a3m_files(
                target_path=integrated_selected_a3m,
                msa_paths=merge_msa_paths,
                target_sequence=self.requests[0][SEQUENCE],
                target=self.requests[0][TARGET],
            )
            dtool.deduplicate_msa_a3m([integrated_selected_a3m], integrated_selected_a3m_dp)
            self.output_path = integrated_selected_a3m_dp
   

    def on_run_end(self):
        request = self.requests[0]
        if self.info_reportor is not None:
            if Path(self.output_path).exists():
                self.info_reportor.update_state(
                    hash_id=request[info_report.HASH_ID],
                    state=self.success_code,
                )
            else:
                self.info_reportor.update_state(
                    hash_id=request[info_report.HASH_ID],
                    state=self.error_code,
                )
