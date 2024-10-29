from celery import Celery

import os
from typing import Any, Dict, List

from lib.base import BaseCommandRunner
from lib.state import State
from lib.pathtree import get_pathtree
from lib.monitor import info_report
from pathlib import Path
from lib.utils import misc, pathtool
from lib.tool import run_chai
import lib.utils.datatool as dtool


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
    "worker.*": {"queue": "queue_chai"},
}

@celery.task(name="chai")
def chaiTask(requests: List[Dict[str, Any]]):
    ChaiRunner(requests=requests)()


def split_chain_request(request: Dict[str, Any]):
    request_list = []
    sequence = misc.safe_get(request, ["sequence"])
    seq_list = sequence.split("\n")
    for chain_id, seq in enumerate(seq_list):
        chain_request = request
        chain_request["sequence"] = seq
        chain_request["name"] = chain_request["name"] + "_chain_" + str(chain_id)
        chain_request["target"] = chain_request["target"] + "_chain_" + str(chain_id)
        request_list.append(chain_request)
    return request_list


class ChaiRunner(BaseCommandRunner):
    def __init__(
        self, requests: List[Dict[str, Any]]
    ):
        super().__init__(requests)
        self.error_code = State.CHAI_ERROR
        self.success_code = State.CHAI_SUCCESS
        self.start_code = State.CHAI_START

    @property
    def start_stage(self) -> int:
        return self.start_code
    
    def build_command(self, request: Dict[str, Any]) -> str:
        ptree = get_pathtree(request)
        chai_config = request["run_config"]["structure_prediction"]["chai"]
        
        # # get args of esm
        self.output_paths = []

        
        a3m_paths = []
        request_list = split_chain_request(request)
        for chain_request in request_list:

            ptree = get_pathtree(chain_request)
            # get msa_path
            str_dict = misc.safe_get(chain_request, ["run_config", "msa_select"])
            key_list = list(str_dict.keys())
            chain_msa_paths = []
            for idx in range(len(key_list)):
                selected_msa_path = str(ptree.strategy.strategy_list[idx]) + "_dp.a3m"
                chain_msa_paths.append(str(selected_msa_path))
            
            msa_image = ptree.alphafold.msa_coverage_image
            Path(msa_image).parent.mkdir(exist_ok=True, parents=True)

            # merge selected msa
            dtool.deduplicate_msa_a3m(chain_msa_paths, str(ptree.alphafold.input_a3m))
            a3m_paths.append(str(ptree.alphafold.input_a3m))

        command = "".join(
            [
                f"python {pathtool.get_module_path(run_chai)} ",
                f"--fasta_path {ptree.seq.fasta} ",
                f"--output_dir {ptree.chai.root} ",
                f"--msa_dir {ptree.chai.msa_dir} ",
                f"--a3m_paths {a3m_paths} ",
                # get chai params
                f"--ntr {misc.safe_get(chai_config, 'ntr')} "
                    if misc.safe_get(chai_config, "ntr")
                    else "",
                f"--ndt {misc.safe_get(chai_config, 'ndt')} "
                    if misc.safe_get(chai_config, "ndt")
                    else "",
                f"--random_seed {misc.safe_get(chai_config, 'random_seed')} "
                    if misc.safe_get(chai_config, "random_seed")
                    else "",
            ]
        )

        return command
            

    def on_run_end(self):
        if self.info_reportor is not None:
            for request in self.requests:
                if all([Path(p).exists() for p in self.output_paths]):
                    self.info_reportor.update_state(
                        hash_id=request[info_report.HASH_ID],
                        state=self.success_code,
                    )
                else:
                    self.info_reportor.update_state(
                        hash_id=request[info_report.HASH_ID],
                        state=self.error_code,
                    )
