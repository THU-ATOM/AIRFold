import os
import numpy as np
from celery import Celery
from loguru import logger
from pathlib import Path
from typing import Any, Dict, List, Union
from collections import defaultdict

from lib.base import BaseRunner
from lib.state import State
from lib.pathtree import get_pathtree
import lib.utils.datatool as dtool
from lib.monitor import info_report
from lib.func_from_docker import make_template_feature
import lib.tool.parse_pdb_to_template as parse_pdb_to_template

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
    "worker.*": {"queue": "queue_tplfeature"},
}

SEQUENCE = "sequence"
TARGET = "target"
DB_PATH = Path("/data/protein/CAMEO/database/cameo_test.db")

@celery.task(name="tplfeature")
def tplfeatureTask(requests: List[Dict[str, Any]]):
    TemplateFeaturizationRunner(requests=requests, db_path=DB_PATH).run()


class TemplateFeaturizationRunner(BaseRunner):
    def __init__(
        self,
        requests: List[Dict[str, Any]],
        db_path: Union[str, Path] = None,
    ) -> None:
        super().__init__(requests, db_path)
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
        TEMPLATE_FEATURES = {
            "template_aatype": np.float32,
            "template_all_atom_masks": np.float32,
            "template_all_atom_positions": np.float32,
            "template_domain_names": np.object,
            "template_sequence": np.object,
            "template_sum_probs": np.float32,
        }

        # get template hits
        ptree = get_pathtree(request=self.requests[0])
        template_hits_path = str(ptree.search.template_hits)
        logger.info(f"template_hits_path: {template_hits_path}")
        template_hits = dtool.read_pickle(template_hits_path)

        output_path = str(ptree.alphafold.template_feat)
        Path(output_path).parent.mkdir(exist_ok=True, parents=True)
        template_feature = make_template_feature(
            input_sequence=self.sequence, pdb_template_hits=template_hits
        )

        template_config = self.requests[0]["run_config"]["template"]
        if (
            template_config["cutomized_template_pdbs"]
            and template_config["cutomized_template_pdbs"] != "None"
        ):
            cutomized_template_pdbs = (
                template_config["cutomized_template_pdbs"].strip().split(",")
            )
            logger.info(
                f"using customized template pdbs {template_config['cutomized_template_pdbs']}"
            )
            cutomized_template_feature = self.get_cutomized_template_feature(
                self.sequence,
                template_pdb_paths=cutomized_template_pdbs,
            )
            for name in TEMPLATE_FEATURES:
                try:
                    if template_feature[name].shape[0] > 0:
                        cutomized_template_feature[name].append(template_feature[name])
                        cutomized_template_feature[name] = np.concatenate(
                            cutomized_template_feature[name], axis=0
                        ).astype(TEMPLATE_FEATURES[name])
                except:
                    logger.error(
                        f"cutomized_template_feature[{name}].shape = {','.join([f'{v.shape}:{type(v)}'for v in cutomized_template_feature[name]])}"
                    )
                    raise ValueError

            template_feature = cutomized_template_feature

        dtool.save_object_as_pickle(template_feature, output_path)
        self.outputs_paths = [output_path]

        return template_feature

    def on_run_end(self):
        if self.info_reportor is not None:
            for request in self.requests:
                if all([Path(p).exists() for p in self.outputs_paths]):
                    self.info_reportor.update_state(
                        hash_id=request[info_report.HASH_ID],
                        state=self.success_code,
                    )
