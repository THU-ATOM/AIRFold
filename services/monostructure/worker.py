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
from lib.func_from_docker import predict_structure
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
    "worker.*": {"queue": "queue_monostructure"},
}

SEQUENCE = "sequence"
TARGET = "target"
DB_PATH = Path("/data/protein/CAMEO/database/cameo_test.db")

@celery.task(name="monostructure")
def monostructureTask(requests: List[Dict[str, Any]]):
    MonoStructureRunner(requests=requests, db_path=DB_PATH).run()


class MonoStructureRunner(BaseRunner):
    def __init__(
        self,
        requests: List[Dict[str, Any]],
        db_path: Union[str, Path] = None,
    ) -> None:
        super().__init__(requests, db_path)
        self.error_code = State.STRUCTURE_ERROR
        self.success_code = State.STRUCTURE_SUCCESS
        self.start_code = State.STRUCTURE_START
        self.target_name = self.requests[0][TARGET]

    @property
    def start_stage(self) -> State:
        return self.start_code

    def run(
        self,
        processed_feat: Dict,
        af2_config: Dict,
        model_name: str,
        *args,
        dry=False,
        **kwargs,
    ):

        ptree = get_pathtree(request=self.requests[0])

        self.output_path = (
            os.path.join(
                str(ptree.alphafold.root),
                model_name,
            )
            + "_unrelaxed.pdb"
        )

        raw_output = (
            os.path.join(
                str(ptree.alphafold.root),
                model_name,
            )
            + "_output_raw.pkl"
        )

        gpu_devices = "".join([f"{i}" for i in get_available_gpus(1)])
        (prediction_result, unrelaxed_pdb_str,) = predict_structure(
            af2_config=af2_config,
            target_name=self.target_name,
            processed_feature=processed_feat,
            model_name=model_name,
            random_seed=kwargs.get("random_seed", 0),  # random.randint(0, 100000),
            gpu_devices=gpu_devices,
        )
        dtool.save_object_as_pickle(prediction_result, raw_output)
        dtool.write_text_file(plaintext=unrelaxed_pdb_str, path=self.output_path)

        return unrelaxed_pdb_str

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
