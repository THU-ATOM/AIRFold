import os
import shutil
from celery import Celery
from loguru import logger
from pathlib import Path
from typing import Any, Dict, List, Union

from lib.base import BaseRunner
from lib.state import State
from lib.pathtree import get_pathtree
import lib.utils.datatool as dtool
from lib.monitor import info_report
from lib.func_from_docker import run_relaxation
from lib.utils.systool import get_available_gpus

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
    "worker.*": {"queue": "queue_relaxation"},
}

SEQUENCE = "sequence"
TARGET = "target"
DB_PATH = Path("/data/protein/CAMEO/database/cameo_test.db")

@celery.task(name="relaxation")
def relaxationTask(requests: List[Dict[str, Any]]):
    AmberRelaxationRunner(requests=requests, db_path=DB_PATH).run()


class AmberRelaxationRunner(BaseRunner):
    def __init__(
        self,
        requests: List[Dict[str, Any]],
        db_path: Union[str, Path] = None,
    ) -> None:
        super().__init__(requests, db_path)
        self.error_code = State.RELAX_ERROR
        self.success_code = State.RELAX_SUCCESS
        self.start_code = State.RELAX_START

    @property
    def start_stage(self) -> State:
        return self.start_code

    def run(self, unrelaxed_pdb_str, model_name, *args, dry=False, **kwargs):
        ptree = get_pathtree(request=self.requests[0])
        gpu_devices = "".join([f"{i}" for i in get_available_gpus(1)])
        relaxed_pdb_str = run_relaxation(
            unrelaxed_pdb_str=unrelaxed_pdb_str, gpu_devices=gpu_devices
        )

        self.output_path = (
            os.path.join(
                str(ptree.alphafold.root),
                model_name,
            )
            + "_relaxed.pdb"
        )

        dtool.write_text_file(relaxed_pdb_str, self.output_path)

        return relaxed_pdb_str

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
