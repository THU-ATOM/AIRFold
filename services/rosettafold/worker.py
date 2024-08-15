from celery import Celery

import os
from pathlib import Path
from typing import Any, Dict, List
import matplotlib.pyplot as plt

from lib.base import BaseCommandRunner
from lib.state import State
from lib.pathtree import get_pathtree
from lib.monitor import info_report
from lib.utils import misc
import lib.utils.datatool as dtool
from lib.utils import misc, pathtool
from lib.tool import plot
from lib.tool.rosettafold2 import rose_predict

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


class RoseTTAFoldRunner(BaseCommandRunner):
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

    def build_command(self, request: Dict[str, Any]) -> str:
        ptree = get_pathtree(request)
        # get msa_path
        str_dict = misc.safe_get(request, ["run_config", "msa_select"])
        key_list = list(str_dict.keys())
        msa_paths = []
        for idx in range(len(key_list)):
            selected_msa_path = str(ptree.strategy.strategy_list[idx]) + "_dp.a3m"
            msa_paths.append(str(selected_msa_path))
        
        
        msa_image = ptree.rosettafold2.msa_coverage_image
        Path(msa_image).parent.mkdir(exist_ok=True, parents=True)
        dtool.deduplicate_msa_a3m(msa_paths, str(ptree.rosettafold2.input_a3m))
        self.save_msa_fig_from_a3m_files(
            msa_paths=msa_paths,
            save_path=msa_image,
        )
        
        args = misc.safe_get(request, ["run_config", "structure_prediction", "rosettafold2"])
        models = args["model_name"].split(",")
        random_seed = args["random_seed"]
        
        self.output_paths = []
        commands = []
        for idx, model_name in enumerate(models):
            out_prefix = str(os.path.join(str(ptree.rosettafold2.root), model_name)) + "_relaxed"
            pdb_output = str(os.path.join(str(ptree.rosettafold2.root), model_name)) + "_relaxed.pdb"
            # rose_predict.run_tf(ptree.seq.fasta, ptree.rosettafold2.input_a3m, out_prefix, 
            #                    model_params=RF2_PT, 
            #                    run_config=rf2_config, seed=(idx+random_seed))
            self.output_paths.append(pdb_output)
        
            command = "".join(
                [
                    f"python {pathtool.get_module_path(rose_predict)} ",
                    f"--fasta_path {str(ptree.seq.fasta)} ",
                    f"--a3m_path {str(ptree.rosettafold2.input_a3m)} ",
                    f"--rose_dir {out_prefix} ",
                    f"--rf2_pt {RF2_PT} ",
                    f"--random_seed {idx+random_seed} ",
                    f"--msa_concat_mode {misc.safe_get(args, 'msa_concat_mode')} "
                    if misc.safe_get(args, "msa_concat_mode")
                    else "",
                    f"--num_recycles {misc.safe_get(args, 'num_recycles')} "
                    if misc.safe_get(args, "num_recycles")
                    else "",
                    f"--max_msa {misc.safe_get(args, 'max_msa')} "
                    if misc.safe_get(args, "max_msa")
                    else "",
                    f"--collapse_identical {misc.safe_get(args, 'collapse_identical')} "
                    if misc.safe_get(args, "collapse_identical")
                    else "",
                    f"--use_mlm {misc.safe_get(args, 'use_mlm')} "
                    if misc.safe_get(args, "use_mlm")
                    else "",
                    f"--use_dropout {misc.safe_get(args, 'use_dropout')} "
                    if misc.safe_get(args, "use_dropout")
                    else "",
                ]
            )
            commands.append(command)
        
        return "&& ".join(commands)
        

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