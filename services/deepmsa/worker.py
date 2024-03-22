import os
from celery import Celery

from pathlib import Path
from typing import Any, Dict, List, Union

from lib.base import BaseCommandRunner
from lib.constant import DB_PATH
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
    "worker.*": {"queue": "queue_deepmsa"},
}


@celery.task(name="deepmsa")
def deepmsaTask(requests: List[Dict[str, Any]]):
    DeepMSARunner(requests=requests, db_path=DB_PATH)()


class DeepMSARunner(BaseCommandRunner):
    def __init__(
        self, requests: List[Dict[str, Any]], db_path: Union[str, Path] = None
    ):
        super().__init__(requests, db_path)

    @property
    def start_stage(self) -> int:
        return State.DEEPMSA_START

    def build_command(self, request: Dict[str, Any]) -> str:
        ptree = get_pathtree(request=request)
        
        executed_file1 = (
                Path(__file__).resolve().parent / "lib" / "tool" / "deepmsa2" / "DeepMSA2_noIMG.py")
        executed_file2 = (
                Path(__file__).resolve().parent / "lib" / "tool" / "deepmsa2" / "DeepMSA2_IMG.py")
        # required options:
        #     -i=/home/simth/test/seq.fasta
        #     -o=/home/simth/test (This should be the same output directory as DeepMSA2_noIMG.py step)
        executed_file3 = (
                Path(__file__).resolve().parent / "lib" / "tool" / "deepmsa2" / "MSA_selection.py")
        params = []
        params.append(f"-s {self.input_path} ")
        params.append(f"-t {self.output_path} ")
        command = f"python {executed_file1} " + "".join(params) + " && " + f"python {executed_file2} " + "".join(params) + " && " + f"python {executed_file3} " + "".join(params)
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
                if tree.search.mmseqs_a3m.exists() and tree.search.mmseqs_fa.exists():
                    self.info_reportor.update_state(
                        hash_id=request[info_report.HASH_ID],
                        state=State.DEEPMSA_SUCCESS,
                    )
                else:
                    self.info_reportor.update_state(
                        hash_id=request[info_report.HASH_ID],
                        state=State.DEEPMSA_ERROR,
                    )
