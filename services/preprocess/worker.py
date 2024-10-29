import os
from celery import Celery

from pathlib import Path
from typing import Any, Dict, List

from lib.base import BaseGroupCommandRunner, PathTreeGroup
from lib.constant import TMP_ROOT
from lib.state import State
from lib.pathtree import get_pathtree
from lib.monitor import info_report
from lib.tool import format
from lib.utils import misc

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
    "worker.*": {"queue": "queue_preprocess"},
}


@celery.task(name="preprocess")
def preprocessTask(requests: List[Dict[str, Any]]):
    PreprocessRunner(requests=requests, tmpdir=TMP_ROOT)()


class PreprocessRunner(BaseGroupCommandRunner):
    def group_requests(self, requests: List[Dict[str, Any]]):
        trees = [get_pathtree(request=request) for request in requests]
        groups = [
            tree_group.requests
            for tree_group in PathTreeGroup.groups_from_trees(trees, "root")
        ]
        return groups

    @property
    def start_stage(self) -> int:
        return State.PREPROCESS_START

    def build_command(self, requests: List[Dict[str, Any]]) -> str:
        path_requests = self.requests2file(requests)
        ptree = get_pathtree(request=requests[0])

        multimer = misc.safe_get(requests[0], ["multimer"]) if misc.safe_get(requests[0], "multimer") else False

        if not multimer:
            command = (
                f"python {Path(format.__file__).resolve()}"
                f" --list {path_requests}"
                f" --output_dir {ptree.seq.root}"
                f" --format 'list -> aln'"
                f" && python {Path(format.__file__).resolve()}"
                f" --list {path_requests}"
                f" --output_dir {ptree.seq.root}"
                f" --format 'list -> fasta'"
            )
        else:
            command = (
                f"python {Path(format.__file__).resolve()}"
                f" --list {path_requests}"
                f" --output_dir {ptree.seq.root}"
                f" --multimer 1"
                f" --format 'list -> fasta'"
            )
        return command

    def on_run_end(self):
        if self.info_reportor is not None:
            for request in self.requests:
                tree = get_pathtree(request=request)
                self.info_reportor.update_path_tree(
                    hash_id=request[info_report.HASH_ID],
                    path_tree=get_pathtree(request=request).tree,
                )
                if tree.seq.fasta.exists() and tree.seq.aln.exists():
                    self.info_reportor.update_state(
                        hash_id=request[info_report.HASH_ID],
                        state=State.PREPROCESS_SUCCESS,
                    )
                else:
                    self.info_reportor.update_state(
                        hash_id=request[info_report.HASH_ID],
                        state=State.PREPROCESS_ERROR,
                    )
