from celery import Celery

import os
from typing import Any, Dict, List

from lib.base import BaseCommandRunner
from lib.state import State
from lib.pathtree import get_pathtree
from lib.monitor import info_report
from lib.utils import pathtool, misc
from lib.utils.execute import rlaunch_exists, rlaunch_wrapper

from lib.tool.rosettafold2 import run_predict

SEQUENCE = "sequence"
RF2_PT = "/data/protein/datasets_2024/rosettafold2/RF2_apr23.pt"

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
    "worker.*": {"queue": "queue_rosettafold"},
}

@celery.task(name="rosettafold")
def rosettafoldTask():
    RoseTTAFoldRunner()


class RoseTTAFoldRunner(BaseCommandRunner):
    def __init__(
        self, requests: List[Dict[str, Any]]
    ):
        super().__init__(requests)
        self.cpu = 8
        self.gpu = 8
        self.error_code = State.RoseTTAFold_ERROR
        self.success_code = State.RoseTTAFold_SUCCESS
        self.start_code = State.RoseTTAFold_START

    @property
    def start_stage(self) -> int:
        return self.start_code

    def build_command(self, request: Dict[str, Any]) -> str:
        
        # query fasta
        ptree = get_pathtree(request=request)
        str_dict = misc.safe_get(self.requests[0], ["run_config", "msa_select"])
        key_list = list(str_dict.keys())
        for index in range(len(key_list)):
            selected_msa_path = ptree.strategy.strategy_list[index]
        if not selected_msa_path:
            return
        
        # get args of rose
        args = misc.safe_get(request, ["run_config", "structure_prediction", "rosettafold2"])
                  
        command = "".join(
            [
                f"python {pathtool.get_module_path(run_predict)} ",
                f"--fasta_path {ptree.seq.fasta} ",
                f"--a3m_path {selected_msa_path} ",
                f"--rose_dir {ptree.rosettafold2.root} ",
                f"--rf2_pt {RF2_PT} ",
                f"--random_seed {misc.safe_get(args, 'random_seed')} " if misc.safe_get(args, "random_seed") else "",
                f"--num_models {misc.safe_get(args, 'num_models')} " if misc.safe_get(args, "num_models") else "",
                f"--msa_concat_mode {misc.safe_get(args, 'msa_concat_mode')} " if misc.safe_get(args, "msa_concat_mode") else "",
                f"--num_recycles {misc.safe_get(args, 'num_recycles')} " if misc.safe_get(args, "num_recycles") else "",
                f"--max_msa {misc.safe_get(args, 'max_msa')} " if misc.safe_get(args, "max_msa") else "",
                f"--collapse_identical {misc.safe_get(args, 'collapse_identical')} " if misc.safe_get(args, "collapse_identical") else "",
                f"--use_mlm {misc.safe_get(args, 'use_mlm')} " if misc.safe_get(args, "use_mlm") else "",
                f"--use_dropout {misc.safe_get(args, 'use_dropout')} " if misc.safe_get(args, "use_dropout") else "",

            ]
        ) 
                  
        if rlaunch_exists():
            command = rlaunch_wrapper(
                command,
                cpu=self.cpu,
                gpu=self.gpu,
            )
        random_seed = misc.safe_get(args, 'random_seed')
        num_models = misc.safe_get(args, 'num_models')
        # out_prefix=f"{out_base}/rf2_seed{seed}"
        for seed in range(random_seed):
            self.output_path = os.path.join(str(ptree.rosettafold2.root), f"/rf2_seed{seed}_00.pdb")
        return command
        

    def on_run_end(self):
        if self.info_reportor is not None:
            for request in self.requests:
                if os.path.exists(self.output_path):
                    self.info_reportor.update_state(
                        hash_id=request[info_report.HASH_ID],
                        state=self.success_code,
                    )
                else:
                    self.info_reportor.update_state(
                        hash_id=request[info_report.HASH_ID],
                        state=self.error_code,
                    )