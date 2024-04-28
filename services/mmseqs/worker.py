import os
from celery import Celery

from pathlib import Path
from typing import Any, Dict, List

from lib.base import BaseCommandRunner
from lib.state import State
from lib.pathtree import get_pathtree
from lib.monitor import info_report
from lib.utils import misc

DB_BASE = "/data/protein/datasets_2024"
UniRef30 = "uniref30_2302/uniref30_2302"
ColabFoldDB = "colabfold_envdb/colabfold_envdb_202108"

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
    "worker.*": {"queue": "queue_mmseqs"},
}


@celery.task(name="mmseqs")
def mmseqsTask(requests: List[Dict[str, Any]]):
    MMseqsRunner(requests=requests)()


class MMseqsRunner(BaseCommandRunner):
    def __init__(
        self, requests: List[Dict[str, Any]]
    ):
        super().__init__(requests)

    @property
    def start_stage(self) -> int:
        return State.MMSEQS_START

    def build_command(self, request: Dict[str, Any]) -> str:

        executed_file = (
                Path(__file__).resolve().parent / "lib" / "tool" / "mmseqs" / "search.py")
        params = []
        # query fasta
        ptree = get_pathtree(request=request)
        params.append(f"--query {ptree.seq.fasta} ")
        # database dir: uniref30_2302_db and colabfold_envdb_202108_db
        # dbbase: "/data/protein/datasets_2024"
        params.append(f"--dbbase {DB_BASE} ")
        params.append(f"--db1 {UniRef30} ")
        params.append(f"--db3 {ColabFoldDB} ")
        # results dir
        params.append(f"--base {ptree.search.mmseqs_base} ")
        
        # # colabfold mmseqs config
        # SENSITIVITY=8
        # EXPAND_EVAL=inf
        # ALIGN_EVAL=10
        # DIFF=3000
        # QSC=-20.0
        # MAX_ACCEPT=1000000
        args = misc.safe_get(request, ["run_config", "msa_search", "mmseqs"])
        params.append(f"-s {misc.safe_get(args, 'sensitivity')} ")
        params.append(f"--align-eval {misc.safe_get(args, 'align_eval')} ")
        params.append(f"--diff {misc.safe_get(args, 'diff')} ")
        params.append(f"--qsc {misc.safe_get(args, 'qsc')} ")
        
        command = f"python {executed_file} " + "".join(params)
        return command

    def on_run_end(self):
        if self.info_reportor is not None:
            for request in self.requests:
                tree = get_pathtree(request=request)
                if tree.search.mmseqs_a3m.exists():
                    self.info_reportor.update_state(
                        hash_id=request[info_report.HASH_ID],
                        state=State.MMSEQS_SUCCESS,
                    )
                else:
                    self.info_reportor.update_state(
                        hash_id=request[info_report.HASH_ID],
                        state=State.MMSEQS_ERROR,
                    )
