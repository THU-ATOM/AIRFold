import os
from celery import Celery
from pathlib import Path
from typing import Any, Dict, List, Union
import matplotlib.pyplot as plt

from lib.base import BaseRunner
from lib.state import State
from lib.pathtree import get_pathtree
import lib.utils.datatool as dtool
from lib.monitor import info_report
from lib.tool import plot
from lib.func_from_docker import monomer_msa2feature

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
    "worker.*": {"queue": "queue_msafeature"},
}

SEQUENCE = "sequence"
TARGET = "target"
DB_PATH = Path("/data/protein/CAMEO/database/cameo_test.db")

@celery.task(name="msafeature")
def msafeatureTask(requests: List[Dict[str, Any]]):
    MonoMSA2FeatureRunner(requests=requests, db_path=DB_PATH).run()


class MonoMSA2FeatureRunner(BaseRunner):
    def __init__(
        self,
        requests: List[Dict[str, Any]],
        db_path: Union[str, Path] = None,
    ) -> None:
        super().__init__(requests, db_path)
        self.error_code = State.MSA2FEATURE_ERROR
        self.success_code = State.MSA2FEATURE_SUCCESS
        self.start_code = State.MSA2FEATURE_START
        self.sequence = self.requests[0][SEQUENCE]
        self.target_name = self.requests[0][TARGET]

    @property
    def start_stage(self) -> State:
        return self.start_code

    @staticmethod
    def save_msa_fig_from_a3m_files(msa_paths, save_path):

        delete_lowercase = lambda line: "".join(
            [t for t in list(line) if not t.islower()]
        )
        msa_collection = []
        for p in msa_paths:
            with open(p) as fd:
                _lines = fd.read().strip().split("\n")
            _lines = [
                delete_lowercase(l) for l in _lines if not l.startswith(">") and l
            ]
            msa_collection.extend(_lines)
        plot.plot_msas([msa_collection])
        plt.savefig(save_path, bbox_inches="tight", dpi=200)

    def run(
        self,
        msa_paths: List[Union[str, Path]],
        template_feat: Dict[str, Any],
        af2_config: Dict[str, Any],
        model_name: str = "model_1",
        *args,
        dry=False,
        **kwargs,
    ):
        if not isinstance(msa_paths, list):
            msa_paths = [msa_paths]
        processed_feature = monomer_msa2feature(
            sequence=self.sequence,
            target_name=self.target_name,
            msa_paths=msa_paths,
            template_feature=template_feat,
            af2_config=af2_config,
            model_name=model_name,
            random_seed=kwargs.get("random_seed", 0),  # random.randint(0, 100000),
        )

        ptree = get_pathtree(request=self.requests[0])
        dtool.deduplicate_msa_a3m(msa_paths, str(ptree.alphafold.input_a3m))

        self.save_msa_fig_from_a3m_files(
            msa_paths=msa_paths,
            save_path=ptree.alphafold.msa_coverage_image,
        )

        return processed_feature

    def on_run_end(self):
        if self.info_reportor is not None:
            for request in self.requests:
                self.info_reportor.update_state(
                    hash_id=request[info_report.HASH_ID],
                    state=self.success_code,
                )

