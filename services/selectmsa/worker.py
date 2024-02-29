import os
from celery import Celery

from pathlib import Path
from typing import Any, Dict, List, Union
from loguru import logger

from lib.base import BaseCommandRunner
from lib.state import State
from lib.pathtree import get_pathtree
from lib.monitor import info_report
from lib.utils.execute import rlaunch_exists, rlaunch_wrapper
from lib.utils import misc
from lib.strategy import *

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
    "worker.*": {"queue": "queue_selectmsa"},
}

DB_PATH = Path("/data/protein/CAMEO/database/cameo_test.db")

@celery.task(name="selectmsa")
def selectmsaTask(requests: List[Dict[str, Any]]):
    MSASelectRunner(requests=requests, db_path=DB_PATH).select()


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
        str_dict = misc.safe_get(request, ["run_config", "msa_select"])
        key_list = list(str_dict.keys())
        # input fasta dir not compatible with a3m
        input_path = self.input_path
        command_list = []
        for index in range(len(key_list)):
            """TODO current implementation could not be compatible with ABA-like strategy"""
            method_ = key_list[index]
            executed_file = (
                Path(__file__).resolve().parent / "lib" / "strategy" / f"{method_}.py"
            )
            params = [f"--{k} {v} " for (k, v) in str_dict[method_].items()]

            params.append(f"--input_a3m_path {input_path} ")
            params.append(f"--output_a3m_path {ptree.strategy.strategy_list[index]} ")

            # command = f"python {pathtool.get_module_path(method_)} " + "".join(params)
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
        self.output_path = str(input_path)
        logger.info(f"The output file for msa selection procedure: {self.output_path}")
        return "&& ".join(command_list)

    def select(self, dry=False):
        # Check if the integrated_search_a3m file exists or not!
        ptree = get_pathtree(request=self.requests[0])
        integrated_search_a3m = str(ptree.search.integrated_search_a3m)
        logger.info(f"integrated_search_a3m: {integrated_search_a3m}")

        config = self.requests[0]["run_config"]["msa_select"]

        if os.path.exists(integrated_search_a3m):
            if "idle" in config:
                self.output_path = integrated_search_a3m
                return integrated_search_a3m
            self.input_path = integrated_search_a3m
            super().run(dry)
            return self.output_path

        else:
            logger.info("The integrated_search_a3m doesn't exist, please check the msa merge procedure!")
            return
            

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
