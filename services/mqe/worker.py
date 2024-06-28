import os
import json
from celery import Celery

from typing import Any, Dict, List

from lib.base import BaseRunner
from lib.state import State
from lib.pathtree import get_pathtree
from lib.utils import misc
from lib.monitor import info_report
import lib.utils.datatool as dtool
from lib.tool.enqa import enqa_msa

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
    "worker.*": {"queue": "queue_mqe"},
}


@celery.task(name="mqe")
def mqeTask(requests: List[Dict[str, Any]]):
    MQERunner(requests=requests)()


class MQERunner(BaseRunner):
    def __init__(
        self, requests: List[Dict[str, Any]]
    ):
        super().__init__(requests)
        self.error_code = State.MQE_ERROR
        self.success_code = State.MQE_SUCCESS
        self.start_code = State.MQE_START

    @property
    def start_stage(self) -> int:
        return self.start_code
    
    def run(self):
        request = self.requests[0]
        ptree = get_pathtree(request)
        struc_root = ptree.struc_root
        seq_id = request["name"]
        
        predicted_result = {}
        for select_mode in os.listdir(struc_root):
            target_dir = str(struc_root) + "/" + select_mode + "/" + seq_id + "alpha/"
            
            plddt_file = target_dir + "plddt_results.json"
            if os.path.exists(plddt_file):
                with open(plddt_file, "r") as pf:
                    plddt_dict = json.load(pf)
                
                for key, val in plddt_dict.items():
                    predicted_pdb = target_dir + key + "_relaxed.pdb"
                    predicted_result[predicted_pdb] = val
        
        mqe_method = misc.safe_get(request, ["run_config", "mse"])
        
        rank_json = {}
        if mqe_method == "enqa":
            ptree.mqe.enqa_temp.parent.mkdir(exist_ok=True, parents=True)
            for pdb_file, plddt_val in predicted_result.items():
                score = enqa_msa.evaluation(input_pdb=pdb_file, tmp_dir=ptree.mqe.enqa_temp)
                rank_json[pdb_file] = {"enqa_score": score, "plddt": plddt_val}
        dtool.write_json(ptree.mqe.enqa_rankfile, data=rank_json)
            

    def on_run_end(self):
        if self.info_reportor is not None:
            for request in self.requests:
                tree = get_pathtree(request=request)
                if tree.mqe.enqa_rankfile.exists():
                    self.info_reportor.update_state(
                        hash_id=request[info_report.HASH_ID],
                        state=self.success_code,
                    )
                else:
                    self.info_reportor.update_state(
                        hash_id=request[info_report.HASH_ID],
                        state=self.error_code,
                    )
                    