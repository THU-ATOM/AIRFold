import os
from celery import Celery

from pathlib import Path
from typing import Any, Dict, List, Union

from lib.base import BaseCommandRunner
from lib.state import State
from lib.pathtree import get_pathtree
from lib.monitor import info_report
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
    "worker.*": {"queue": "queue_msaSelect"},
}

DB_PATH = Path("/data/protein/CAMEO/database/cameo_test.db")

@celery.task(name="msaSelect")
def msaSelectTask(requests: List[Dict[str, Any]]):
    command = MSASelectRunner(requests=requests, db_path=DB_PATH).run()

    return command



class MSASelectRunner(BaseCommandRunner):
    """
    TODO not sure the error code throw part
    """

    def __init__(
        self,
        requests: List[Dict[str, Any]],
        db_path: Union[str, Path] = None,
        cpu: int = 4,
    ) -> None:
        super().__init__(requests, db_path)
        self.cpu = cpu
        self.success_code = State.SELECT_SUCCESS
        self.error_code = State.SELECT_ERROR

    @property
    def start_stage(self) -> int:
        return State.SELECT_START

    def build_command(self, request: Dict[str, Any]) -> list:
        ptree = get_pathtree(request=request)
        command_list = []
        str_dict = self.select_config
        key_list = list(str_dict.keys())
        # input fasta dir not compatible with a3m
        input_path = self.input_path
        for index in range(len(key_list)):
            """TODO current implementation could not be compatible with ABA-like strategy"""
            method_ = key_list[index]
            executed_file = (
                Path(__file__).resolve().parent / "strategy" / f"{method_}.py"
            )
            params = [f"--{k} {v} " for (k, v) in str_dict[method_].items()]

            params.append(f"--input_a3m_path {input_path} ")
            params.append(f"--output_a3m_path {ptree.strategy.strategy_list[index]} ")

            command = f"python {executed_file} " + "".join(params)
            if rlaunch_exists():
                command = rlaunch_wrapper(
                    command,
                    cpu=self.cpu,
                    gpu=0,
                    memory=5000,
                )
            command_list.append(command)
            input_path = ptree.strategy.strategy_list[index]
        self.output_path = input_path

        return "&& ".join(command_list)

    def run(self, msa_path, config, *args, dry=False, **kwargs):
        if "idle" in config:
            self.output_path = msa_path
            return msa_path
        self.select_config = config
        self.input_path = msa_path
        super().run(dry)
        return self.output_path

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
