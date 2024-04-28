import os
from celery import Celery
from pathlib import Path
from typing import Any, Dict, List

from lib.base import BaseCommandRunner
from lib.state import State
from lib.pathtree import get_pathtree
import lib.utils.datatool as dtool
from lib.monitor import info_report
from lib.tool import tool_utils as utils
from lib.utils.execute import rlaunch_exists, rlaunch_wrapper
import pickle as pkl

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
    "worker.*": {"queue": "queue_selecttpl"},
}

SEQUENCE = "sequence"
TARGET = "target"

@celery.task(name="selecttpl")
def selecttplTask(requests: List[Dict[str, Any]]):
    TPLTSelectRunner(requests=requests)()


class TPLTSelectRunner(BaseCommandRunner):
    def __init__(
        self,
        requests: List[Dict[str, Any]],
        cpu: int = 4,
    ) -> None:
        super().__init__(requests)
        self.cpu = cpu
        self.success_code = State.TPLT_SELECT_SUCCESS
        self.error_code = State.TPLT_SELECT_ERROR

    @property
    def start_stage(self) -> int:
        return State.TPLT_SELECT_START

    def build_command(self, request: Dict[str, Any]) -> list:
        if (
            "template_select_strategy" in request["run_config"]["template"]
            and request["run_config"]["template"]["template_select_strategy"] == "top"
        ):
            executed_file = (
                Path(__file__).resolve().parent / "lib" / "strategy" / "template_top_select.py"
            )
        elif (
            "template_select_strategy" in request["run_config"]["template"]
            and request["run_config"]["template"]["template_select_strategy"]
            == "clustering"
        ):
            executed_file = (
                Path(__file__).resolve().parent / "lib" / "strategy" / "template_clustering.py"
            )
        else:
            executed_file = (
                Path(__file__).resolve().parent / "lib" / "strategy" / "template_clustering.py"
            )
        params = []
        params.append(f"-s {self.input_path} ")
        params.append(f"-t {self.output_path} ")
        command = f"python {executed_file} " + "".join(params)

        if rlaunch_exists():
            command = rlaunch_wrapper(
                command,
                cpu=self.cpu,
                gpu=0,
                memory=5000,
            )

        return command

    def run(self, dry=False):
        # get template features
        request = self.requests[0]
        ptree = get_pathtree(request)
        template_feats_path = str(ptree.alphafold.template_feat)
        template_feats = dtool.read_pickle(template_feats_path)

        self.output_path = ptree.alphafold.selected_template_feat
        
        if (
            "template_select_strategy" in request["run_config"]["template"]
            and request["run_config"]["template"]["template_select_strategy"] == "none"
        ):
            with open(self.output_path, "wb") as fd:
                pkl.dump(template_feats, fd)
        else:
            with utils.tmpdir_manager() as tmpdir:
                self.input_path = Path(tmpdir) / "template_feat.pkl"
                dtool.save_object_as_pickle(template_feats, self.input_path)
                super().run(dry)
            

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