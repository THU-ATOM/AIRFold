import os
from celery import Celery

from pathlib import Path
from typing import Any, Dict, List

from lib.base import BaseCommandRunner
from lib.state import State
from lib.pathtree import get_pathtree
from lib.utils import pathtool
from lib.monitor import info_report
from lib.tool import deepmsa_img, jgi_comb
from lib.utils.execute import rlaunch_exists, rlaunch_wrapper


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
    "worker.*": {"queue": "queue_deepmsa"},
}

para_json = dict(
    # database parameter 
    dMSAhhblitsdb=os.path.join("/data/protein/datasets_2024", 'uniclust30_2017_04/uniclust30_2017_04'),
    dMSAjackhmmerdb=os.path.join("/data/protein/datasets_2024", 'uniref90/uniref90.fasta'),
    dMSAhmmsearchdb=os.path.join("/data/protein/datasets_2024", 'metaclust/metaclust.fasta'),
    qMSAhhblitsdb=os.path.join("/data/protein/datasets_2024", 'UniRef30_2022_02/UniRef30_2022_02'),
    qMSAjackhmmerdb=os.path.join("/data/protein/datasets_2024", 'uniref90/uniref90.fasta'),
    qMSAhhblits3db=os.path.join("/data/protein/alphafold", 'bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt'),
    qMSAhmmsearchdb=os.path.join("/data/protein/datasets_2022", 'mgnify/mgy_clusters.fa'),
    mMSAJGI=os.path.join("/data/protein/datasets_2024", 'JGIclust')
)



@celery.task(name="deepmsa")
def deepmsaTask(requests: List[Dict[str, Any]]):
    DeepqMSARunner(requests=requests)()
    DeepdMSARunner(requests=requests)()
    DeepmMSARunner(requests=requests)()


class DeepqMSARunner(BaseCommandRunner):
    def __init__(
        self, requests: List[Dict[str, Any]]
    ):
        super().__init__(requests)
        self.cpu = 8  # to do

    @property
    def start_stage(self) -> int:
        return State.DEEPqMSA_START

    def build_command(self, request: Dict[str, Any]) -> str:
        # /home/casp15/code/AIRFold/lib/tool/deepmsa2/bin/qMSA/scripts/qMSA.py
        executed_file = (
                Path(__file__).resolve().parent / "lib" / "tool" / "deepmsa2" / "bin" / "qMSA" / "scripts" / "qMSA2.py")
        
        # query fasta
        ptree = get_pathtree(request=request)
        
        command = f"python {executed_file} " \
                  f"{ptree.seq.fasta} " \
                  f"-hhblitsdb={para_json['qMSAhhblitsdb']} " \
                  f"-jackhmmerdb={para_json['qMSAjackhmmerdb']} " \
                  f"-hhblits3db={para_json['qMSAhhblits3db']} " \
                  f"-hmmsearchdb={para_json['qMSAhmmsearchdb']} " \
                  f"-outdir={ptree.search.deepqmsa_base} " \
                  f"-tmpdir={ptree.search.deepqmsa_base_tmp} " \
                  f"-ncpu={self.cpu} "
                  
                  
        if rlaunch_exists():
            command = rlaunch_wrapper(
                command,
                cpu=self.cpu,
                gpu=0,
                memory=5000,
            )
        return command
        

    def on_run_end(self):
        if self.info_reportor is not None:
            for request in self.requests:
                tree = get_pathtree(request=request)
                if tree.search.deepmsa_qa3m.exists():
                    self.info_reportor.update_state(
                        hash_id=request[info_report.HASH_ID],
                        state=State.DEEPqMSA_SUCCESS,
                    )
                else:
                    self.info_reportor.update_state(
                        hash_id=request[info_report.HASH_ID],
                        state=State.DEEPqMSA_ERROR,
                    )

class DeepdMSARunner(BaseCommandRunner):
    def __init__(
        self, requests: List[Dict[str, Any]]
    ):
        super().__init__(requests)
        self.cpu = 4  # to do

    @property
    def start_stage(self) -> int:
        return State.DEEPdMSA_START

    def build_command(self, request: Dict[str, Any]) -> str:
        # /home/casp15/code/AIRFold/lib/tool/deepmsa2/bin/dMSA/scripts/build_MSA.py
        executed_file = (
                Path(__file__).resolve().parent / "lib" / "tool" / "deepmsa2" / "bin" / "dMSA" / "scripts" / "build_MSA.py")
        
        # query fasta
        ptree = get_pathtree(request=request)

        command = f"python {executed_file} " \
                  f"{ptree.seq.fasta} " \
                  f"-hhblitsdb={para_json['dMSAhhblitsdb']} " \
                  f"-jackhmmerdb={para_json['dMSAjackhmmerdb']} " \
                  f"-hmmsearchdb={para_json['dMSAhmmsearchdb']} " \
                  f"-outdir={ptree.search.deepdmsa_base} " \
                  f"-tmpdir={ptree.search.deepdmsa_base_tmp} " \
                  f"-ncpu={self.cpu} "
                  
                  
        if rlaunch_exists():
            command = rlaunch_wrapper(
                command,
                cpu=self.cpu,
                gpu=0,
                memory=5000,
            )
        return command

    def on_run_end(self):
        if self.info_reportor is not None:
            for request in self.requests:
                tree = get_pathtree(request=request)
                if tree.search.deepmsa_da3m.exists():
                    self.info_reportor.update_state(
                        hash_id=request[info_report.HASH_ID],
                        state=State.DEEPdMSA_SUCCESS,
                    )
                else:
                    self.info_reportor.update_state(
                        hash_id=request[info_report.HASH_ID],
                        state=State.DEEPdMSA_ERROR,
                    )


class DeepmMSARunner(BaseCommandRunner):
    def __init__(
        self, requests: List[Dict[str, Any]]
    ):
        super().__init__(requests)
        self.cpu = 4  # to do

    @property
    def start_stage(self) -> int:
        return State.DEEPmMSA_START

    def build_command(self, request: Dict[str, Any]) -> str:
        ptree = get_pathtree(request=request)

        # get args of deepmsa_img
        jgi_path = para_json['mMSAJGI']
        hhlib_path = (Path(__file__).resolve().parent / "lib" / "tool" / "deepmsa2" / "bin" / "qMSA")
        dmsalib_path = (Path(__file__).resolve().parent / "lib" / "tool" / "deepmsa2" / "bin" / "dMSA")

        command_jgi = "".join(
            [
                f"python {pathtool.get_module_path(deepmsa_img)} ",
                f"--jgi {jgi_path} ",
                f"--hhlib {hhlib_path} ",
                f"--deepmmsa_base {ptree.search.deepmmsa_base} ",
                f"--deepmmsa_base_temp {ptree.search.deepmmsa_base_tmp} ",
                f"--dmsa_hhbaln {ptree.search.deepdmsa_hhbaln} "
            ]
        )
        
        if rlaunch_exists():
            command_jgi = rlaunch_wrapper(
                command_jgi,
                cpu=self.cpu,
                gpu=0,
                memory=5000,
            )
        
        print("2rd step/JGI combination is starting!\n")
    

        command_comb = "".join(
            [
                f"python {pathtool.get_module_path(jgi_comb)} ",
                f"--hhlib {hhlib_path} ",
                f"--dmsalib {dmsalib_path} ",
                f"--deepmmsa_base {ptree.search.deepmmsa_base} ",
                f"--deepmmsa_base_temp {ptree.search.deepmmsa_base_tmp} ",
                f"--seq {ptree.seq.fasta} ",
                f"--deepqmsa_hhbaln {ptree.search.deepqmsa_hhbaln} ",
                f"--deepqmsa_hhba3m {ptree.search.deepqmsa_hhba3m} ",
                f"--deepdmsa_hhbaln {ptree.search.deepdmsa_hhbaln} ",
                f"--deepdmsa_hhba3m {ptree.search.deepdmsa_hhba3m} "
            ]
        )
        
        if rlaunch_exists():
            command_comb = rlaunch_wrapper(
                command_comb,
                cpu=self.cpu,
                gpu=0,
                memory=5000,
            )
        return command_jgi + "&& " + command_comb

    def on_run_end(self):
        if self.info_reportor is not None:
            for request in self.requests:
                tree = get_pathtree(request=request)
                if tree.search.deepmsa_ma3m.exists():
                    self.info_reportor.update_state(
                        hash_id=request[info_report.HASH_ID],
                        state=State.DEEPmMSA_SUCCESS,
                    )
                else:
                    self.info_reportor.update_state(
                        hash_id=request[info_report.HASH_ID],
                        state=State.DEEPmMSA_ERROR,
                    )
