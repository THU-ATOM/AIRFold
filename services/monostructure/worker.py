import os
from celery import Celery
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
from lib.func_from_docker import monomer_msa2feature, predict_structure, run_relaxation
from lib.utils.systool import get_available_gpus
from lib.utils import misc

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
    "worker.*": {"queue": "queue_monostructure"},
}

SEQUENCE = "sequence"
TARGET = "target"

@celery.task(name="monostructure")
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
        template_feat: Dict[str, Any],
        af2_config: Dict[str, Any],
        model_name: str = "model_1",
        random_seed: int = 0,
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
            random_seed=random_seed  # random.randint(0, 100000),
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
        processed_feat: Dict,
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

        raw_output = (
            os.path.join(
                str(ptree.alphafold.root),
                model_name,
            )
            + "_output_raw.pkl"
        )

        gpu_devices = "".join([f"{i}" for i in get_available_gpus(1)])
        (prediction_result, unrelaxed_pdb_str,) = predict_structure(
            af2_config=af2_config,
            target_name=self.target_name,
            processed_feature=processed_feat,
            model_name=model_name,
            random_seed=random_seed,  # random.randint(0, 100000),
            gpu_devices=gpu_devices,
        )
        dtool.save_object_as_pickle(prediction_result, raw_output)
        dtool.write_text_file(plaintext=unrelaxed_pdb_str, path=self.output_path)

        return unrelaxed_pdb_str

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

    def run(self, unrelaxed_pdb_str, model_name):
        ptree = get_pathtree(request=self.requests[0])
        gpu_devices = "".join([f"{i}" for i in get_available_gpus(1)])
        relaxed_pdb_str = run_relaxation(
            unrelaxed_pdb_str=unrelaxed_pdb_str, gpu_devices=gpu_devices
        )

        self.output_path = (
            os.path.join(
                str(ptree.alphafold.root),
                model_name,
            )
            + "_relaxed.pdb"
        )

        dtool.write_text_file(relaxed_pdb_str, self.output_path)

        return relaxed_pdb_str

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
        selected_template_feat_path = ptree.alphafold.selected_template_feat
        selected_template_feat = dtool.read_pickle(selected_template_feat_path)
        if not selected_template_feat:
            return
        
        for m_name in models:
            processed_feature = self.mono_msa2feature(
                msa_paths=selected_msa_path,
                template_feat=selected_template_feat,
                af2_config=af2_config,
                model_name=m_name,
                random_seed=random_seed
            )
            logger.info(
                f"the shape of msa_feat is: {processed_feature['msa_feat'].shape}"
            )
            if not processed_feature:
                return
            unrelaxed_pdb_str = self.mono_structure(
                processed_feat=processed_feature,
                af2_config=af2_config,
                model_name=m_name,
                random_seed=random_seed,
            )
            if not unrelaxed_pdb_str:
                return

            relaxed_pdb_str = self.amber_relax(
                unrelaxed_pdb_str=unrelaxed_pdb_str, model_name=m_name
            )
            if not relaxed_pdb_str:
                return
