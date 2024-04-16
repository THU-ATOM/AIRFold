import os
from celery import Celery

from pathlib import Path
from typing import Any, Dict, List, Union

from lib.base import BaseCommandRunner
from lib.state import State
from lib.pathtree import get_pathtree
from lib.utils import misc, pathtool
from lib.monitor import info_report
from lib.tool import deepmsa_img
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

rootpath = "/home/casp15/code/MSA/DeepMSA2"
databasesrootpath = "/data/casp15/code/MSA/DeepMSA2"

para_json = dict(
    # main program parameter
    qMSApkg=os.path.join(rootpath, "bin/qMSA"),
    dMSApkg=os.path.join(rootpath, "bin/dMSA"),
    # database parameter 
    dMSAhhblitsdb=os.path.join("/data/protein/datasets_2022", 'uniclust30'),
    dMSAjackhmmerdb=os.path.join("/data/protein/datasets_2022", 'uniref90/uniref90.fasta'),
    dMSAhmmsearchdb=os.path.join("/data/protein/datasets_2024", 'metaclust/metaclust.fasta'),
    qMSAhhblitsdb=os.path.join("/data/protein/datasets_2024", 'UniRef30_2302'),
    qMSAjackhmmerdb=os.path.join("/data/protein/datasets_2022", 'uniref90/uniref90.fasta'),
    qMSAhhblits3db=os.path.join("/data/protein/datasets_2024", 'bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt'),
    qMSAhmmsearchdb=os.path.join("/data/protein/datasets_2022", 'mgnify/mgy_clusters.fasta'),
    mMSAJGI=os.path.join(databasesrootpath, 'JGIclust')
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

    @property
    def start_stage(self) -> int:
        return State.DEEPqMSA_START

    def build_command(self, request: Dict[str, Any]) -> str:
        # /home/casp15/code/AIRFold/lib/tool/deepmsa2/bin/qMSA/scripts/qMSA.py
        executed_file = (
                Path(__file__).resolve().parent / "lib" / "tool" / "deepmsa2" / "bin" / "qMSA" / "scripts" / "qMSA.py")
        
        # query fasta
        ptree = get_pathtree(request=request)
        input_fasta = ptree.seq.fasta

        command = f"python {executed_file} " \
                  f"-hhblitsdb={para_json['qMSAhhblitsdb']} " \
                  f"-jackhmmerdb={para_json['qMSAjackhmmerdb']} " \
                  f"-hhblits3db={para_json['qMSAhhblits3db']} " \
                  f"-hmmsearchdb={para_json['qMSAhmmsearchdb']} " \
                  f"-ncpu={self.cpu} " \
                  f"{input_fasta}"
                  
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

    @property
    def start_stage(self) -> int:
        return State.DEEPdMSA_START

    def build_command(self, request: Dict[str, Any]) -> str:
        # /home/casp15/code/AIRFold/lib/tool/deepmsa2/bin/dMSA/scripts/build_MSA.py
        executed_file = (
                Path(__file__).resolve().parent / "lib" / "tool" / "deepmsa2" / "bin" / "dMSA" / "scripts" / "build_MSA.py")
        
        # query fasta
        ptree = get_pathtree(request=request)
        input_fasta = ptree.seq.fasta

        command = f"python {executed_file} " \
                  f"-hhblitsdb={para_json['dMSAhhblitsdb']} " \
                  f"-jackhmmerdb={para_json['dMSAjackhmmerdb']} " \
                  f"-hmmsearchdb={para_json['dMSAhmmsearchdb']} " \
                  f"-ncpu={self.cpu} " \
                  f"{input_fasta}"
                  
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

    @property
    def start_stage(self) -> int:
        return State.DEEPmMSA_START

    def build_command(self, request: Dict[str, Any]) -> str:
        # query fasta
        ptree = get_pathtree(request=request)
        input_fasta = ptree.seq.fasta
        
        # search JGI
        if not ptree.search.deepmsa_qa3m.exists():
            content = 'sequence does not have MSA result yet. Please run DeepqMSA first!'
            return f"echo {content}"
        elif not ptree.search.deepmsa_qjaca3m.exists() and not ptree.search.deepmsa_djaca3m.exists():
            content = 'sequence does not need additional JGI search due to no jack result. skip!'
            return f"echo {content}"
        
        # query fasta
        ptree = get_pathtree(request=request)

        # get args of deepmsa_img
        args = misc.safe_get(request, ["run_config", "msa_search", "search", "deepmsa"])

        command = "".join(
            [
                f"python {pathtool.get_module_path(deepmsa_img)} ",
                f"-i {ptree.seq.fasta} ",
                f"-o {ptree.search.blast_a3m} ",
                f"-w {ptree.search.blast_whole_fa} ",
                #  parser.add_argument("-e", "--evalue", type=float, default=1e-5)
                f"-e {misc.safe_get(args, 'evalue')} "
                if misc.safe_get(args, "evalue")
                else "",
            ]
        )
        return command

    def on_run_end(self):
        if self.info_reportor is not None:
            for request in self.requests:
                tree = get_pathtree(request=request)
                if tree.search.deepmsa_da3m.exists():
                    self.info_reportor.update_state(
                        hash_id=request[info_report.HASH_ID],
                        state=State.DEEPmMSA_SUCCESS,
                    )
                else:
                    self.info_reportor.update_state(
                        hash_id=request[info_report.HASH_ID],
                        state=State.DEEPmMSA_ERROR,
                    )
