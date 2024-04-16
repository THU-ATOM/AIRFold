import os
from celery import Celery

from pathlib import Path
from typing import Any, Dict, List, Union

from lib.base import BaseGroupCommandRunner, PathTreeGroup
from lib.constant import TMP_ROOT
from lib.state import State
from lib.pathtree import get_pathtree
from lib.utils import misc
from lib.monitor import info_report
from lib.tool import hhblits
from lib.utils.execute import rlaunch_exists, rlaunch_wrapper
from lib.utils.pathtool import get_module_path


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
    "worker.*": {"queue": "queue_hhblits"},
}


@celery.task(name="hhblits")
def hhblitsTask(requests: List[Dict[str, Any]]):
    HHblitsRunner(requests=requests, tmpdir=TMP_ROOT)()


class HHblitsRunner(BaseGroupCommandRunner):
    def __init__(
        self,
        requests: List[Dict[str, Any]],
        tmpdir: Union[str, Path] = None,
    ):
        super().__init__(requests, tmpdir)

    @property
    def start_stage(self) -> int:
        return State.HHBLITS_START

    def group_requests(self, requests: List[Dict[str, Any]]):
        trees = [get_pathtree(request=request) for request in requests]
        groups = [
            tree_group.requests
            for tree_group in PathTreeGroup.groups_from_trees(trees, "root")
        ]
        return groups

    def build_command(self, requests: List[Dict[str, Any]]) -> str:
        # here we mask the uniclust30 data and test the pipeline
        path_requests = self.requests2file(requests)

        tmp_seq_dir = self.mk_seqs(requests)

        request_sample = requests[0]
        ptree = get_pathtree(request=request_sample)
        args = misc.safe_get(
            request_sample, ["run_config", "msa_search", "search", "hhblits"]
        )
        data = " ".join(
            [
                str(getattr(ptree, x).data)
                for x in misc.safe_get(args, "dataset", ["uniclust30", "bfd"])
            ]
        )
        command = "".join(
            [
                f"python {get_module_path(hhblits)} ",
                f"--data {data} ",
                f"-l {path_requests} ",
                f"-i {tmp_seq_dir} ",
                f"--in_seq_only ",
                f"-o {ptree.search.root} ",
                f"--iteration {misc.safe_get(args, 'iteration')} "
                if misc.safe_get(args, "iteration")
                else "",
                f"--e_value {misc.safe_get(args, 'e_value')} "
                if misc.safe_get(args, "e_value")
                else "",
                f"--realign_max {misc.safe_get(args, 'realign_max')} "
                if misc.safe_get(args, "realign_max")
                else "",
                f"--maxfilt {misc.safe_get(args, 'maxfilt')} "
                if misc.safe_get(args, "maxfilt")
                else "",
                f"--min_prefilter_hits {misc.safe_get(args, 'min_prefilter_hits')} "
                if misc.safe_get(args, "min_prefilter_hits")
                else "",
                f"--maxseq {misc.safe_get(args, 'maxseq')} "
                if misc.safe_get(args, "maxseq")
                else "",
                f"--diff_default {misc.safe_get(args, 'diff_default')} "
                if misc.safe_get(args, "diff_default")
                else "",
                f"--diff_fast {misc.safe_get(args, 'diff_fast')} "
                if misc.safe_get(args, "diff_fast")
                else "",
                f"--timeout {misc.safe_get(args, 'timeout')} "
                if misc.safe_get(args, "timeout")
                else "",
                f"--thread {misc.safe_get(args, 'thread')} "
                if misc.safe_get(args, "thread")
                else "",
                f"--cpu {misc.safe_get(args, 'cpu')} "
                if misc.safe_get(args, "cpu")
                else "",
            ]
        )

        if rlaunch_exists():
            command = rlaunch_wrapper(
                command,
                cpu=self.thread * self.cpu_per_thread,
                gpu=0,
                memory=50000,
                charged_group="health",
            )
        return command

    def on_run_end(self):
        if self.info_reportor is not None:
            for request in self.requests:
                tree = get_pathtree(request=request)
                if (
                    tree.search.hhblist_bfd_uniclust_a3m.exists()
                    and tree.search.hhblist_bfd_uniclust_fa.exists()
                ):
                    self.info_reportor.update_state(
                        hash_id=request[info_report.HASH_ID],
                        state=State.HHBLITS_SUCCESS,
                    )
                else:
                    self.info_reportor.update_state(
                        hash_id=request[info_report.HASH_ID],
                        state=State.HHBLITS_ERROR,
                    )