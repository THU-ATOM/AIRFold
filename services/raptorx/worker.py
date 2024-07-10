from celery import Celery

import os
from typing import Any, Dict, List

from lib.base import BaseCommandRunner
from lib.state import State
from lib.pathtree import get_pathtree
from lib.monitor import info_report
from lib.utils import misc
from lib.utils.execute import rlaunch_exists, rlaunch_wrapper


SEQUENCE = "sequence"

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
    "worker.*": {"queue": "queue_raptorx"},
}

@celery.task(name="raptorx")
def raptorxTask(requests: List[Dict[str, Any]]):
    RaptorXRunner(requests=requests)()

class RaptorXRunner(BaseCommandRunner):
    def __init__(
        self, requests: List[Dict[str, Any]]
    ):
        super().__init__(requests)
        self.cpu = 8
        self.gpu = 8
        self.error_code = State.RaptorX_ERROR
        self.success_code = State.RaptorX_SUCCESS
        self.start_code = State.RaptorX_START

    @property
    def start_stage(self) -> int:
        return self.start_code

    def build_command(self, request: Dict[str, Any]) -> str:
        
        # query fasta
        ptree = get_pathtree(request=request)
        
        # get args of rose
        args = misc.safe_get(request, ["run_config", "structure_prediction", "esmfold"])
                  
        command = ""      
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