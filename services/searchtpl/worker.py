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
from lib.func_from_docker import search_template

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
    "worker.*": {"queue": "queue_searchtpl"},
}

SEQUENCE = "sequence"
TARGET = "target"
DB_PATH = Path("/data/protein/CAMEO/database/cameo_test.db")

@celery.task(name="searchtpl")
def searchtplTask(requests: List[Dict[str, Any]]):
    TemplateSearchRunner(requests=requests, db_path=DB_PATH).run()


class TemplateSearchRunner(BaseRunner):
    def __init__(
        self,
        requests: List[Dict[str, Any]],
        db_path: Union[str, Path] = None,
    ) -> None:
        super().__init__(requests, db_path)
        self.error_code = State.TPLT_SEARCH_ERROR
        self.success_code = State.TPLT_SEARCH_SUCCESS
        self.start_code = State.TPLT_SEARCH_START
        self.sequence = self.requests[0][SEQUENCE]

    @property
    def start_stage(self) -> State:
        return self.start_code

    def run(self):
        ptree = get_pathtree(request=self.requests[0])
        template_searching_msa_path = str(ptree.search.jackhammer_uniref90_a3m)
        output_path = str(ptree.search.template_hits)
        Path(output_path).parent.mkdir(exist_ok=True, parents=True)
        self.outputs_paths = [output_path]

        copy_template_hits_from = self.requests[0]["run_config"]["template"].get(
            "copy_template_hits_from", "None"
        )
        try:
            logger.info(f"copying from {copy_template_hits_from}")
            if copy_template_hits_from == "None" or not copy_template_hits_from:
                raise ValueError("copy_template_hits_from is None")
            if Path(copy_template_hits_from).exists():
                shutil.copy(copy_template_hits_from, output_path)
                pdb_template_hits = dtool.read_pickle(output_path)
            else:
                shutil.copy(
                    os.path.join(
                        os.path.dirname(output_path),
                        copy_template_hits_from,
                    ),
                    output_path,
                )
            pdb_template_hits = dtool.read_pickle(output_path)
        except:
            if copy_template_hits_from != "None":
                logger.info(
                    f"copying from {copy_template_hits_from}  failed, fall back to searching logic"
                )

            pdb_template_hits = search_template(
                self.sequence,
                template_searching_msa_path=template_searching_msa_path,
            )

            dtool.save_object_as_pickle(pdb_template_hits, output_path)

        return pdb_template_hits

    def on_run_end(self):
        if self.info_reportor is not None:
            for request in self.requests:
                if all([Path(p).exists() for p in self.outputs_paths]):
                    self.info_reportor.update_state(
                        hash_id=request[info_report.HASH_ID],
                        state=self.success_code,
                    )
