from celery import Celery

import os
from pathlib import Path
from typing import Any, Dict, List
import matplotlib.pyplot as plt

from lib.base import BaseRunner
from lib.state import State
from lib.pathtree import get_pathtree
from lib.monitor import info_report
from lib.utils import misc
import lib.utils.datatool as dtool
from lib.tool import plot
from lib.tool.rosettafold2 import run_predict

SEQUENCE = "sequence"
RF2_PT = "/data/protein/datasets_2024/rosettafold2/RF2_apr23.pt"

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
    "worker.*": {"queue": "queue_rosettafold"},
}

@celery.task(name="rosettafold")
def rosettafoldTask(requests: List[Dict[str, Any]]):
    RoseTTAFoldRunner(requests=requests)()


class RoseTTAFoldRunner(BaseRunner):
    def __init__(
        self, requests: List[Dict[str, Any]]
    ):
        super().__init__(requests)
        self.error_code = State.RoseTTAFold_ERROR
        self.success_code = State.RoseTTAFold_SUCCESS
        self.start_code = State.RoseTTAFold_START

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

    def run(self):
        ptree = get_pathtree(request=self.requests[0])
        # get msa_path
        str_dict = misc.safe_get(self.requests[0], ["run_config", "msa_select"])
        key_list = list(str_dict.keys())
        msa_paths = []
        for idx in range(len(key_list)):
            selected_msa_path = str(ptree.strategy.strategy_list[idx]) + "_dp.a3m"
            msa_paths.append(str(selected_msa_path))
        
        dtool.deduplicate_msa_a3m(msa_paths, str(ptree.rosettafold2.input_a3m))
        
        msa_image = ptree.rosettafold2.msa_coverage_image
        Path(msa_image).parent.mkdir(exist_ok=True, parents=True)
        self.save_msa_fig_from_a3m_files(
            msa_paths=msa_paths,
            save_path=msa_image,
        )
        
        rf2_config = self.requests[0]["run_config"]["structure_prediction"]["rosettafold2"]
        models = rf2_config["model_name"].split(",")
        
        self.output_paths = []
        for idx, model_name in enumerate(models):
            out_prefix = str(os.path.join(str(ptree.rosettafold2.root), model_name)) + "_relaxed"
            pdb_output = str(os.path.join(str(ptree.rosettafold2.root), model_name)) + "_relaxed.pdb"
            run_predict.run_tf(ptree.seq.fasta, ptree.rosettafold2.input_a3m, out_prefix, 
                               model_params="/data/protein/datasets_2024/rosettafold2/RF2_apr23.pt", 
                               run_config=rf2_config, seed=idx)
            self.output_paths.append(pdb_output)
        

    def on_run_end(self):
        if self.info_reportor is not None:
            for request in self.requests:
                if all([Path(p).exists() for p in self.output_paths]):
                    self.info_reportor.update_state(
                        hash_id=request[info_report.HASH_ID],
                        state=self.success_code,
                    )
                else:
                    self.info_reportor.update_state(
                        hash_id=request[info_report.HASH_ID],
                        state=self.error_code,
                    )