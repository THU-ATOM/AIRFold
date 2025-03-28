import os
from celery import Celery

from typing import Any, Dict, List

from lib.base import BaseCommandRunner
from lib.state import State
from lib.pathtree import get_pathtree
from lib.utils import misc, pathtool
from lib.monitor import info_report
from lib.tool import blast

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
    "worker.*": {"queue": "queue_blast"},
}


@celery.task(name="blast")
def blastTask(requests: List[Dict[str, Any]]):
    BlastRunner(requests=requests)()


class BlastRunner(BaseCommandRunner):
    def __init__(
        self, requests: List[Dict[str, Any]]
    ):
        super().__init__(requests)

    @property
    def start_stage(self) -> int:
        return State.BLAST_START

    def build_command(self, request: Dict[str, Any]) -> str:
        # get input file path for blast
        ptree = get_pathtree(request=request)

        # get args of blast
        args = misc.safe_get(request, ["run_config", "msa_search", "blast"])

        command = "".join(
            [
                f"python {pathtool.get_module_path(blast)} ",
                f"-i {ptree.seq.fasta} ",
                f"-o {ptree.search.blast_a3m} ",
                f"-w {ptree.search.blast_whole_fa} ",
                #  parser.add_argument("-e", "--evalue", type=float, default=1e-5)
                f"-e {misc.safe_get(args, 'evalue')} "
                if misc.safe_get(args, "evalue")
                else "",
                f"-n {misc.safe_get(args, 'num_iterations')} "
                if misc.safe_get(args, "num_iterations")
                else "",
                f"-b {misc.safe_get(args, 'blasttype')} "
                if misc.safe_get(args, "blasttype")
                else "",
            ]
        )

        return command

    def on_run_end(self):
        if self.info_reportor is not None:
            for request in self.requests:
                tree = get_pathtree(request=request)
                if tree.search.blast_a3m.exists():
                    self.info_reportor.update_state(
                        hash_id=request[info_report.HASH_ID],
                        state=State.BLAST_SUCCESS,
                    )
                else:
                    self.info_reportor.update_state(
                        hash_id=request[info_report.HASH_ID],
                        state=State.BLAST_ERROR,
                    )
                    