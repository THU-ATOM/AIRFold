import os
import json
from celery import Celery
from loguru import logger
from typing import Any, Dict, List

# from lib.state import State
from lib.pathtree import get_pathtree
from lib.utils import misc
# from lib.monitor import info_report
import lib.utils.datatool as dtool
from lib.tool.enqa import enqa_msa
from lib.tool.gcpl import gcpl_qa

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
    MQERunner(requests=requests, method="gcpl").run()


class MQERunner():
    def __init__(
        self, requests: List[Dict[str, Any]], method: str
    ):
        # super().__init__(requests)
        self.requests = requests
        self.mqe_method = method
    #     self.error_code = State.MQE_ERROR
    #     self.success_code = State.MQE_SUCCESS
    #     self.start_code = State.MQE_START

    # @property
    # def start_stage(self) -> int:
    #     return self.start_code
    
    def run(self):
        ptree_base = get_pathtree(self.requests[0])
        
        logger.info(f"***** MQE Method: {self.mqe_method}")
        if self.mqe_method == "enqa":
            # EnQA
            ptree_base.mqe.enqa_temp.parent.mkdir(exist_ok=True, parents=True)
            mqe_tmp_dir = ptree_base.mqe.enqa_temp
            mqe_rank_file = ptree_base.mqe.enqa_rankfile
        else:
            # GraphCPLMQA
            ptree_base.mqe.gcpl_temp.parent.mkdir(exist_ok=True, parents=True)
            mqe_tmp_dir = ptree_base.mqe.gcpl_temp
            mqe_rank_file = ptree_base.mqe.gcpl_rankfile
            
        predicted_result = []
        for request in self.requests:
            ptree = get_pathtree(request)
            struc_args = misc.safe_get(request, ["run_config", "structure_prediction"])
            logger.info(f"STRUC ARGS: {struc_args}")
            # ms_config = ptree.final_msa_fasta.parent.name
            if "alphafold" in struc_args.keys():
                target_dir = str(ptree.alphafold.root)
                
                plddt_file = os.path.join(target_dir, "plddt_results.json")
                logger.info(f"pLDDT File: {plddt_file}")
                
                if os.path.exists(plddt_file):
                    with open(plddt_file, "r") as pf:
                        plddt_dict = json.load(pf)
                    
                    for key, val in plddt_dict.items():
                        
                        predicted_pdb = os.path.join(target_dir, key + "_relaxed.pdb")
                        logger.info(f"Evaluating decoy: {predicted_pdb}")
                        
                        if self.mqe_method == "enqa":
                            score = enqa_msa.evaluation(input_pdb=predicted_pdb, tmp_dir=mqe_tmp_dir)
                        else:
                            score = gcpl_qa.evaluation(fasta_file=ptree.seq.fasta, decoy_file=predicted_pdb, tmp_dir=mqe_tmp_dir)
                        
                        logger.info(f"Evaluation score: {score}")
                        predicted_result.append({"predicted_pdb": str(predicted_pdb), "plddt": val, "score": score})
        
        
        # dtool.write_json(mqe_rank_file, data=predicted_result)
        import pickle
        with open(mqe_rank_file, 'wb') as f:
            pickle.dump(predicted_result, f)
            
                    