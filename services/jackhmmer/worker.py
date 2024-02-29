import os
from celery import Celery

from pathlib import Path
from typing import Any, Dict, List, Union

from lib.base import BaseGroupCommandRunner, PathTreeGroup
from lib.state import State
from lib.pathtree import get_pathtree
from lib.tool import jackhmmer
from lib.utils import misc
from lib.monitor import info_report
from lib.utils.execute import rlaunch_exists, rlaunch_wrapper
from lib.utils.pathtool import get_module_path
from lib.constant import TMP_ROOT

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
    "worker.*": {"queue": "queue_jackhmmer"},
}

DB_PATH = Path("/data/protein/CAMEO/database/cameo_test.db")

@celery.task(name="jackhmmer")
def jackhmmerTask(requests: List[Dict[str, Any]]):
    JackhmmerRunner(requests=requests, db_path=DB_PATH, tmpdir=TMP_ROOT).run()


class JackhmmerRunner(BaseGroupCommandRunner):
    """
    SearchRunner is a runner for search pipeline.
    """

    def __init__(
        self,
        requests: List[Dict[str, Any]],
        db_path: Union[str, Path] = None,
        tmpdir: Union[str, Path] = None,
        thread: int = 4,
        cpu_per_thread: int = 8,
        timeout: float = 3600,
    ):
        super().__init__(requests, db_path, tmpdir)
        self.thread = thread
        self.cpu_per_thread = cpu_per_thread
        self.timeout = timeout

    @property
    def start_stage(self) -> int:
        return State.JACKHMM_START

    def group_requests(self, requests: List[Dict[str, Any]]):
        trees = [get_pathtree(request=request) for request in requests]
        groups = [
            tree_group.requests
            for tree_group in PathTreeGroup.groups_from_trees(trees, "root")
        ]
        return groups

    def build_command(self, requests: List[Dict[str, Any]]) -> str:
        # input a segment list of the same sequence.
        path_requests = self.requests2file(requests)

        tmp_seq_dir = self.mk_seqs(requests)
        # get the temp directory,

        ptree = get_pathtree(request=requests[0])
        request_sample = requests[0]

        args = misc.safe_get(
            request_sample, ["run_config", "msa_search", "search", "jackhmmer"]
        )

        # TODO: parse the assigned dataset from  request, by default search all the data
        args_command = (
            f"--n_iter {misc.safe_get(args, 'n_iter')} "
            if misc.safe_get(args, "n_iter")
            else "",
            f"--e_value {misc.safe_get(args, 'e_value')} "
            if misc.safe_get(args, "e_value")
            else "",
            f"--filter_f1 {misc.safe_get(args, 'filter_f1')} "
            if misc.safe_get(args, "filter_f1")
            else "",
            f"--filter_f2 {misc.safe_get(args, 'filter_f2')} "
            if misc.safe_get(args, "filter_f2")
            else "",
            f"--filter_f3 {misc.safe_get(args, 'filter_f3')} "
            if misc.safe_get(args, "filter_f3")
            else "",
            f"--thread {misc.safe_get(args, 'thread')} "
            if misc.safe_get(args, "thread")
            else "",
            f"--cpu {misc.safe_get(args, 'cpu')} "
            if misc.safe_get(args, "cpu")
            else "",
        )

        command_uniref = (
            f"python {get_module_path(jackhmmer)} "
            f"-d {ptree.afuniref.data} "
            f"-l {path_requests} "
            f"-i {tmp_seq_dir} "
            f"--in_seq_only "
            f"-o {ptree.search.root} "
            # f"--thread {self.thread} "
            # f"--cpu {self.cpu_per_thread} "
            f"-z 135301051 "
        ) + " ".join(args_command)
        if rlaunch_exists():
            command_uniref = rlaunch_wrapper(
                command_uniref,
                cpu=self.thread * self.cpu_per_thread,
                gpu=0,
                memory=50000,
                charged_group="wangshuo_8gpu",
            )

        command_mgnify = (
            f"python {get_module_path(jackhmmer)} "
            f"-d {ptree.afmgnify.data} "
            f"-l {path_requests} "
            f"-i {tmp_seq_dir} "
            f"--in_seq_only "
            f"-o {ptree.search.root} "
            # f"--thread {self.thread} "
            # f"--cpu {self.cpu_per_thread} "
            f"-z 304820129 "
        ) + " ".join(args_command)

        if rlaunch_exists():
            command_mgnify = rlaunch_wrapper(
                command_mgnify,
                cpu=self.thread * self.cpu_per_thread,
                gpu=0,
                memory=50000,
                charged_group="wangshuo_8gpu",
            )

        return command_uniref + "&& " + command_mgnify

    def on_run_end(self):
        if self.info_reportor is not None:
            for request in self.requests:
                tree = get_pathtree(request=request)
                if (
                    tree.search.jackhammer_mgnify_fa.exists()
                    and tree.search.jackhammer_uniref90_fa.exists()
                ):
                    self.info_reportor.update_state(
                        hash_id=request[info_report.HASH_ID],
                        state=State.JACKHMM_SUCCESS,
                    )
                else:
                    self.info_reportor.update_state(
                        hash_id=request[info_report.HASH_ID],
                        state=State.JACKHMM_ERROR,
                    )