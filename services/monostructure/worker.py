import os
from copy import deepcopy
from celery import Celery
from celery.result import AsyncResult, allow_join_result
from pathlib import Path
from typing import Any, Dict, List, Union
import matplotlib.pyplot as plt
from loguru import logger

from lib.base import BaseRunner
from lib.state import State
from lib.pathtree import get_pathtree
import lib.utils.datatool as dtool
from lib.monitor import info_report
from lib.tool import plot
# from lib.utils.systool import get_available_gpus
from lib.utils import misc
from lib.constant import AF_PARAMS_ROOT

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
    "worker.*": {"queue": "queue_monostructure"},
}

SEQUENCE = "sequence"
TARGET = "target"

@celery_client.task(name="monostructure")
def monostructureTask(requests: List[Dict[str, Any]]):
    AirFoldRunner(requests=requests)()


class MonoFeatureRunner(BaseRunner):
    def __init__(
        self,
        requests: List[Dict[str, Any]],
    ) -> None:
        super().__init__(requests)
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
        selected_template_feat_path: str,
        af2_config: Dict[str, Any],
        model_name: str = "model_1",
        random_seed: int = 0,
    ):
        if not isinstance(msa_paths, list):
            msa_paths = [str(msa_paths)]
        
        run_stage = "monomer_msa2feature"
        ptree = get_pathtree(request=self.requests[0])
        out_path = str(ptree.alphafold.processed_feat)
        
        argument_dict = {
            "sequence": self.sequence,
            "target_name": self.target_name,
            "msa_paths": msa_paths,
            "template_feature": selected_template_feat_path,
            "model_name": model_name,
            "random_seed": random_seed,
        }
        argument_dict = deepcopy(argument_dict)
        for k, v in af2_config.items():
            if k not in argument_dict:
                argument_dict[k] = v
        
        task = celery_client.send_task("alphafold", args=[run_stage, out_path, argument_dict], queue="queue_alphafold")
        task_result = AsyncResult(task.id, app=celery_client)
        
        with allow_join_result():
            try:
                out_path = task_result.get()
                processed_feature = dtool.read_pickle(out_path)
                
                dtool.deduplicate_msa_a3m(msa_paths, str(ptree.alphafold.input_a3m))

                self.save_msa_fig_from_a3m_files(
                    msa_paths=msa_paths,
                    save_path=ptree.alphafold.msa_coverage_image,
                )
                logger.info(
                    f"the shape of msa_feat is: {processed_feature['msa_feat'].shape}"
                )
                return out_path
            
            except TimeoutError as exc:
                print("--- Exception: %s\n Timeout!" %exc)
                return


    def on_run_end(self):
        if self.info_reportor is not None:
            for request in self.requests:
                self.info_reportor.update_state(
                    hash_id=request[info_report.HASH_ID],
                    state=self.success_code,
                )

class MonoStructureRunner(BaseRunner):
    def __init__(
        self,
        requests: List[Dict[str, Any]]
    ) -> None:
        super().__init__(requests)
        self.error_code = State.STRUCTURE_ERROR
        self.success_code = State.STRUCTURE_SUCCESS
        self.start_code = State.STRUCTURE_START
        self.target_name = self.requests[0][TARGET]

    @property
    def start_stage(self) -> State:
        return self.start_code

    def run(
        self,
        processed_feature_path: str,
        af2_config: Dict,
        random_seed: int,
        model_name: str
    ):
        

        ptree = get_pathtree(request=self.requests[0])

        self.output_path = (
            os.path.join(
                str(ptree.alphafold.root),
                model_name,
            )
            + "_unrelaxed.pdb"
        )
        
        run_stage = "predict_structure"
        argument_dict = {
            "target_name": self.target_name,
            "processed_feature": processed_feature_path,
            "model_name": model_name,
            "data_dir": str(AF_PARAMS_ROOT),
            "random_seed": random_seed,
            "return_representations": True,
        }
        argument_dict = deepcopy(argument_dict)
        for k, v in af2_config.items():
            if k not in argument_dict:
                argument_dict[k] = v
        
        out_path = str(os.path.join(str(ptree.alphafold.root), model_name))
        task = celery_client.send_task("alphafold", args=[run_stage, out_path, argument_dict], queue="queue_alphafold")
        task_result = AsyncResult(task.id, app=celery_client)

        with allow_join_result():
            try:
                un_relaxed_pdb_path = task_result.get()
                # unrelaxed_pdb_str = dtool.read_text_file(path=un_relaxed_pdb_path)
                return un_relaxed_pdb_path
            
            except TimeoutError as exc:
                print("--- Exception: %s\n Timeout!" %exc)
                return
        

    def on_run_end(self):
        if self.info_reportor is not None:
            for request in self.requests:
                if os.path.exists(self.output_path):
                    self.info_reportor.update_state(
                        hash_id=request[info_report.HASH_ID],
                        state=self.success_code,
                    )
                else:
                    self.info_reportor.update_state(
                        hash_id=request[info_report.HASH_ID],
                        state=self.error_code,
                    )


class AmberRelaxationRunner(BaseRunner):
    def __init__(
        self,
        requests: List[Dict[str, Any]]
    ) -> None:
        super().__init__(requests)
        self.error_code = State.RELAX_ERROR
        self.success_code = State.RELAX_SUCCESS
        self.start_code = State.RELAX_START

    @property
    def start_stage(self) -> State:
        return self.start_code

    def run(self, un_relaxed_pdb_path, model_name):
        ptree = get_pathtree(request=self.requests[0])
        # gpu_devices = "".join([f"{i}" for i in get_available_gpus(1)])
        # relaxed_pdb_str = run_relaxation(
        #     unrelaxed_pdb_str=unrelaxed_pdb_str, gpu_devices=gpu_devices
        # )
        run_stage = "run_relaxation"
        argument_dict = {"unrelaxed_pdb_str": un_relaxed_pdb_path}
        out_path = str(os.path.join(str(ptree.alphafold.root), model_name)) + "_relaxed.pdb"
        task = celery_client.send_task("alphafold", args=[run_stage, out_path, argument_dict], queue="queue_alphafold")
        task_result = AsyncResult(task.id, app=celery_client)
        
        with allow_join_result():
            try:
                relaxed_pdb_path = task_result.get()
                self.output_path = relaxed_pdb_path
                return relaxed_pdb_path
            
            except TimeoutError as exc:
                print("--- Exception: %s\n Timeout!" %exc)
                return


    def on_run_end(self):
        if self.info_reportor is not None:
            for request in self.requests:
                if os.path.exists(self.output_path):
                    self.info_reportor.update_state(
                        hash_id=request[info_report.HASH_ID],
                        state=self.success_code,
                    )
                else:
                    self.info_reportor.update_state(
                        hash_id=request[info_report.HASH_ID],
                        state=self.error_code,
                    )

class AirFoldRunner(BaseRunner):
    def __init__(
        self, requests: List[Dict[str, Any]]
    ) -> None:
        """_summary_

        Parameters
        ----------
        request : List[Dict], optional
            Request for the pipeline. Each item in the list includes the basic
            information about a protein sequence, e.g. name, time stamp, etc.,
            as well as the strategy for structure prediction.
            See `sample_request.jsonl` as an example.
        """
        super().__init__(requests)
        """
        Here we make a request, cut the request according to the segment part.
        """
        # self.info_reportor.update_reserved(
        #     hash_id=requests[0]["hash_id"], update_dict={"pid": os.getpid()}
        # )
        # logger.info(f"#### the process id is {os.getpid()}")

        self.mono_msa2feature =  MonoFeatureRunner(requests=self.requests)

        self.mono_structure   =  MonoStructureRunner(requests=self.requests)
        self.amber_relax      =  AmberRelaxationRunner(requests=self.requests)

    @property
    def start_stage(self) -> int:
        return State.AIRFOLD_START

    def run(self):

        af2_config = self.requests[0]["run_config"]["structure_prediction"]["alphafold"]
        models = af2_config["model_name"].split(",")
        random_seed = af2_config.get("random_seed", 0)
        af2_config = {
            k: v
            for k, v in af2_config.items()
            if v != "model_name" and v != "random_seed"
        }

        # get msa_path
        ptree = get_pathtree(request=self.requests[0])
        integrated_search_a3m = str(ptree.search.integrated_search_a3m)
        str_dict = misc.safe_get(self.requests[0], ["run_config", "msa_select"])
        key_list = list(str_dict.keys())
        selected_msa_path = integrated_search_a3m
        for index in range(len(key_list)):
            selected_msa_path = ptree.strategy.strategy_list[index]
        if not selected_msa_path:
            return

        # get selected_template_feat
        selected_template_feat_path = str(ptree.alphafold.selected_template_feat)
        # selected_template_feat = dtool.read_pickle(selected_template_feat_path)
        # if not selected_template_feat:
        #     return
        
        for m_name in models:
            processed_feature_path = self.mono_msa2feature(
                msa_paths=selected_msa_path,
                selected_template_feat_path=selected_template_feat_path,
                af2_config=af2_config,
                model_name=m_name,
                random_seed=random_seed
            )
            
            if not processed_feature_path:
                return
            un_relaxed_pdb_path = self.mono_structure(
                processed_feature_path=processed_feature_path,
                af2_config=af2_config,
                model_name=m_name,
                random_seed=random_seed,
            )
            if not un_relaxed_pdb_path:
                return

            relaxed_pdb_path = self.amber_relax(
                un_relaxed_pdb_path=un_relaxed_pdb_path, model_name=m_name
            )
            if not relaxed_pdb_path:
                return
