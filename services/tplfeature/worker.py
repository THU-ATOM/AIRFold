import os
from celery import Celery
from celery.result import AsyncResult, allow_join_result
from loguru import logger
from pathlib import Path
from typing import Any, Dict, List
from collections import defaultdict

from lib.base import BaseRunner
from lib.state import State
from lib.pathtree import get_pathtree
import lib.utils.datatool as dtool
from lib.monitor import info_report
import lib.tool.parse_pdb_to_template as parse_pdb_to_template
from lib.constant import PDBMMCIF_ROOT

CELERY_RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", "rpc://")
CELERY_BROKER_URL = (
    os.environ.get("CELERY_BROKER_URL", "pyamqp://guest:guest@localhost:5672/"),
)

celery_client = Celery(
    __name__,
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
)

celery_client.conf.task_routes = {
    "worker.*": {"queue": "queue_tplfeature"},
}

SEQUENCE = "sequence"
TARGET = "target"

@celery_client.task(name="tplfeature")
def tplfeatureTask(requests: List[Dict[str, Any]]):
    TemplateFeaturizationRunner(requests=requests)()


class TemplateFeaturizationRunner(BaseRunner):
    def __init__(
        self,
        requests: List[Dict[str, Any]]
    ) -> None:
        super().__init__(requests)
        self.error_code = State.TPLT_FEAT_ERROR
        self.success_code = State.TPLT_FEAT_SUCCESS
        self.start_code = State.TPLT_FEAT_START
        self.sequence = self.requests[0][SEQUENCE]

    @property
    def start_stage(self) -> State:
        return self.start_code

    @staticmethod
    def get_cutomized_template_feature(sequence, template_pdb_paths):
        template_feature = defaultdict(list)
        for idx, pdb_path in enumerate(template_pdb_paths):
            _out = parse_pdb_to_template.parse_pdb_to_template(
                sequence=sequence, pdb_path=pdb_path, template_name=f"cust_{idx}"
            )
            for k in _out:
                template_feature[k].append(_out[k])
        return template_feature

    def run(self):

        # get template hits
        ptree = get_pathtree(request=self.requests[0])
        template_hits_path = str(ptree.search.template_hits)
        logger.info(f"template_hits_path: {template_hits_path}")
        template_hits = dtool.read_pickle(template_hits_path)

        output_path = str(ptree.alphafold.template_feat)
        Path(output_path).parent.mkdir(exist_ok=True, parents=True)
        # template_feature = af2_make_template_feature(
        #     input_sequence=self.sequence, pdb_template_hits=template_hits
        # )
        
        run_stage = "make_template_feature"
        argument_dict = {
            "input_sequence": self.sequence,
            "pdb_template_hits": template_hits,
            "max_template_hits": 20,
            "template_mmcif_dir": str(PDBMMCIF_ROOT / "mmcif_files"),
            "max_template_date": "2022-05-31",
            "obsolete_pdbs_path": str(PDBMMCIF_ROOT / "obsolete.dat"),
            "kalign_binary_path": "kalign",
        }
        
        task = celery_client.send_task("alphafold", args=[run_stage, output_path, argument_dict], queue="queue_alphafold")
        task_result = AsyncResult(task.id, app=celery_client)
        
        with allow_join_result():
            try:
                output_path = task_result.get()
                self.outputs_paths = [output_path]
                            
            except TimeoutError as exc:
                print("--- Exception: %s\n Timeout!" %exc)

    def on_run_end(self):
        if self.info_reportor is not None:
            for request in self.requests:
                if all([Path(p).exists() for p in self.outputs_paths]):
                    self.info_reportor.update_state(
                        hash_id=request[info_report.HASH_ID],
                        state=self.success_code,
                    )
