from celery import Celery

import os
from typing import Any, Dict, List

from lib.base import BaseCommandRunner
from lib.state import State
from lib.pathtree import get_pathtree
from lib.monitor import info_report
from pathlib import Path
from lib.utils import pathtool
from lib.tool import run_esmfold


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
    "worker.*": {"queue": "queue_esmfold"},
}

@celery.task(name="esmfold")
def singlefoldTask(requests: List[Dict[str, Any]]):
    ESMFoldRunner(requests=requests)()

class ESMFoldRunner(BaseCommandRunner):
    def __init__(
        self, requests: List[Dict[str, Any]]
    ):
        super().__init__(requests)
        self.error_code = State.ESMFold_ERROR
        self.success_code = State.ESMFold_SUCCESS
        self.start_code = State.ESMFold_START

    @property
    def start_stage(self) -> int:
        return self.start_code
    
    def build_command(self, request: Dict[str, Any]) -> str:
        ptree = get_pathtree(request)
        esm_config = request["run_config"]["structure_prediction"]["esmfold"]
        
        # # get args of esm
        # args = misc.safe_get(request, ["run_config", "structure_prediction", "esmfold"])
        models = esm_config["model_name"].split(",")
        random_seed = esm_config["random_seed"]
        self.output_paths = []
        for model_name in models:
            pdb_path = str(os.path.join(str(ptree.esmfold.root), model_name)) + "_relaxed.pdb"
            Path(pdb_path).parent.mkdir(exist_ok=True, parents=True)
            self.output_paths.append(pdb_path)
        
        model_names = " ".join(models)
        command = "".join(
            [
                f"python {pathtool.get_module_path(run_esmfold)} ",
                f"--fasta_file {ptree.seq.fasta} ",
                f"--pdb_root {ptree.esmfold.root} ",
                f"--random_seed {random_seed} ",
                f"--model_names {model_names} ",
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