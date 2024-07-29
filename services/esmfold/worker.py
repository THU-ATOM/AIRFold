from celery import Celery

import os
from typing import Any, Dict, List

from lib.base import BaseRunner
from lib.state import State
from lib.pathtree import get_pathtree
from lib.monitor import info_report
from pathlib import Path
# from lib.utils import misc
# from lib.utils.execute import rlaunch_exists, rlaunch_wrapper
from lib.tool import esmfold


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
    "worker.*": {"queue": "queue_esmfold"},
}

@celery.task(name="esmfold")
def singlefoldTask(requests: List[Dict[str, Any]]):
    ESMFoldRunner(requests=requests)()

class ESMFoldRunner(BaseRunner):
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
    
    def run(self):
        request=self.request[0]
        ptree = get_pathtree(request)
        esm_config = request["run_config"]["structure_prediction"]["esmfold"]
        
        # # get args of esm
        # args = misc.safe_get(request, ["run_config", "structure_prediction", "esmfold"])
        models = esm_config["model_name"].split(",")
        self.output_paths = []
        for idx, model_name in enumerate(models):
            pdb_path = str(os.path.join(str(ptree.alphafold.root), model_name)) + "_relaxed.pdb"
            esmfold.prediction(sequence=ptree.seq.fasta, esm_pdb_path=pdb_path, random_seed=idx)
            self.output_paths.append(pdb_path)

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