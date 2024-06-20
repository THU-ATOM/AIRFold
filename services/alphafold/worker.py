import os
from loguru import logger
from copy import deepcopy
from celery import Celery
from pathlib import Path
from typing import Any, Dict, List, Union
from loguru import logger

import pickle as pkl
import matplotlib.pyplot as plt

from lib.base import BaseRunner, BaseCommandRunner
from lib.state import State
from lib.pathtree import get_pathtree
import lib.utils.datatool as dtool
from lib.monitor import info_report
from lib.utils.execute import rlaunch_exists, rlaunch_wrapper
from lib.tool import plot, tool_utils
from lib.utils import misc
from lib.constant import AF_PARAMS_ROOT, PDB70_ROOT, PDBMMCIF_ROOT
from lib.utils.systool import get_available_gpus
from lib.tool.run_af2_stage import (
    search_template, 
    make_template_feature, 
    monomer_msa2feature, 
    predict_structure,
    run_relaxation,
)

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
    "worker.*": {"queue": "queue_alphafold"},
}

SEQUENCE = "sequence"
TARGET = "target"

@celery_client.task(name="alphafold")
def alphafoldTask(requests: List[Dict[str, Any]]):
    TemplateSearchRunner(requests=requests)()
    TPLTSelectRunner(requests=requests)()
    TemplateFeaturizationRunner(requests=requests)()
    AlphaFoldRunner(requests=requests)()


class TemplateSearchRunner(BaseRunner):
    def __init__(
        self,
        requests: List[Dict[str, Any]]
    ) -> None:
        super().__init__(requests)
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

        argument_dict = {
            "input_sequence": self.sequence,
            "template_searching_msa_path": template_searching_msa_path,
            "pdb70_database_path": str(PDB70_ROOT / "pdb70"),
            "hhsearch_binary_path": "hhsearch",
        }
        
        try:
            out_tpl_path = alphafold_func(run_stage="search_template", 
                                          output_path=output_path, 
                                          argument_dict=argument_dict
                                          )
            return out_tpl_path
        except TimeoutError as exc:
            logger.exception(exc)
            return False

    def on_run_end(self):
        if self.info_reportor is not None:
            for request in self.requests:
                if all([Path(p).exists() for p in self.outputs_paths]):
                    self.info_reportor.update_state(
                        hash_id=request[info_report.HASH_ID],
                        state=self.success_code,
                    )

class TPLTSelectRunner(BaseCommandRunner):
    def __init__(
        self,
        requests: List[Dict[str, Any]],
        cpu: int = 4,
    ) -> None:
        super().__init__(requests)
        self.cpu = cpu
        self.success_code = State.TPLT_SELECT_SUCCESS
        self.error_code = State.TPLT_SELECT_ERROR

    @property
    def start_stage(self) -> int:
        return State.TPLT_SELECT_START

    def build_command(self, request: Dict[str, Any]) -> list:
        if (
            "template_select_strategy" in request["run_config"]["template"]
            and request["run_config"]["template"]["template_select_strategy"] == "top"
        ):
            executed_file = (
                Path(__file__).resolve().parent / "lib" / "strategy" / "template_top_select.py"
            )
        elif (
            "template_select_strategy" in request["run_config"]["template"]
            and request["run_config"]["template"]["template_select_strategy"]
            == "clustering"
        ):
            executed_file = (
                Path(__file__).resolve().parent / "lib" / "strategy" / "template_clustering.py"
            )
        else:
            executed_file = (
                Path(__file__).resolve().parent / "lib" / "strategy" / "template_clustering.py"
            )
        params = []
        params.append(f"-s {self.input_path} ")
        params.append(f"-t {self.output_path} ")
        command = f"python {executed_file} " + "".join(params)

        if rlaunch_exists():
            command = rlaunch_wrapper(
                command,
                cpu=self.cpu,
                gpu=0,
                memory=5000,
            )

        return command

    def run(self, dry=False):
        # get template features
        request = self.requests[0]
        ptree = get_pathtree(request)
        template_feats_path = str(ptree.alphafold.template_feat)
        template_feats = dtool.read_pickle(template_feats_path)

        self.output_path = ptree.alphafold.selected_template_feat
        
        if (
            "template_select_strategy" in request["run_config"]["template"]
            and request["run_config"]["template"]["template_select_strategy"] == "none"
        ):
            with open(self.output_path, "wb") as fd:
                pkl.dump(template_feats, fd)
        else:
            with tool_utils.tmpdir_manager() as tmpdir:
                self.input_path = Path(tmpdir) / "template_feat.pkl"
                dtool.save_object_as_pickle(template_feats, self.input_path)
                super().run(dry)
            

    def on_run_end(self):
        request = self.requests[0]
        if self.info_reportor is not None:
            if Path(self.output_path).exists():
                self.info_reportor.update_state(
                    hash_id=request[info_report.HASH_ID],
                    state=self.success_code,
                )
            else:
                self.info_reportor.update_state(
                    hash_id=request[info_report.HASH_ID],
                    state=self.error_code,
                )


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

    def run(self):

        # get template hits
        ptree = get_pathtree(request=self.requests[0])
        template_hits_path = str(ptree.search.template_hits)
        logger.info(f"template_hits_path: {template_hits_path}")
        template_hits = dtool.read_pickle(template_hits_path)

        output_path = str(ptree.alphafold.template_feat)
        self.outputs_paths = [output_path]
        Path(output_path).parent.mkdir(exist_ok=True, parents=True)

        argument_dict = {
            "input_sequence": self.sequence,
            "pdb_template_hits": template_hits,
            "max_template_hits": 20,
            "template_mmcif_dir": str(PDBMMCIF_ROOT / "mmcif_files"),
            "max_template_date": "2022-05-31",
            "obsolete_pdbs_path": str(PDBMMCIF_ROOT / "obsolete.dat"),
            "kalign_binary_path": "kalign",
        }
        
        try:
            output_path = alphafold_func(run_stage="make_template_feature", 
                                          output_path=output_path, 
                                          argument_dict=argument_dict
                                          )

            return output_path
        except TimeoutError as exc:
            logger.exception(exc)
            return False


    def on_run_end(self):
        if self.info_reportor is not None:
            for request in self.requests:
                if all([Path(p).exists() for p in self.outputs_paths]):
                    self.info_reportor.update_state(
                        hash_id=request[info_report.HASH_ID],
                        state=self.success_code,
                    )

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
        
        ptree = get_pathtree(request=self.requests[0])
        self.output_path = str(ptree.alphafold.processed_feat)
        
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
        
        try:
            out_fea_path = alphafold_func(run_stage="monomer_msa2feature", 
                                          output_path=self.output_path, 
                                          argument_dict=argument_dict
                                          )
            return out_fea_path
        except TimeoutError as exc:
            logger.exception(exc)
            return False


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
        try:
            pdb_output = alphafold_func(run_stage="predict_structure", 
                                          output_path=out_path, 
                                          argument_dict=argument_dict
                                          )
            return pdb_output
        except TimeoutError as exc:
            logger.exception(exc)
            return False
        

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
        argument_dict = {"unrelaxed_pdb_str": un_relaxed_pdb_path}
        out_path = str(os.path.join(str(ptree.alphafold.root), model_name)) + "_relaxed.pdb"
        
        try:
            relaxed_pdb_path = alphafold_func(run_stage="predict_structure", 
                                          output_path=out_path, 
                                          argument_dict=argument_dict
                                          )
            self.output_path = relaxed_pdb_path
            return relaxed_pdb_path
        except TimeoutError as exc:
            logger.exception(exc)
            return False


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

class AlphaFoldRunner(BaseRunner):
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

        self.mono_msa2feature =  MonoFeatureRunner(requests=self.requests)

        self.mono_structure   =  MonoStructureRunner(requests=self.requests)
        self.amber_relax      =  AmberRelaxationRunner(requests=self.requests)

    @property
    def start_stage(self) -> int:
        return State.AIRFOLD_START

    def run_one_model(self, m_name):

        processed_feature_path = self.mono_msa2feature(
            msa_paths=self.selected_msa_path,
            selected_template_feat_path=self.selected_template_feat_path,
            af2_config=self.af2_config,
            model_name=m_name,
            random_seed=self.random_seed
        )
            
        if not processed_feature_path:
            return
        un_relaxed_pdb_path = self.mono_structure(
            processed_feature_path=processed_feature_path,
            af2_config=self.af2_config,
            model_name=m_name,
            random_seed=self.random_seed,
        )
        if not os.path.exists(un_relaxed_pdb_path):
            logger.info(f"{un_relaxed_pdb_path} doesn't exist, please check")
            return

        relaxed_pdb_path = self.amber_relax(
            un_relaxed_pdb_path=un_relaxed_pdb_path, model_name=m_name
        )
        if not os.path.exists(relaxed_pdb_path):
            logger.info(f"{relaxed_pdb_path} doesn't exist, please check")
            return

    def run(self):

        af2_config = self.requests[0]["run_config"]["structure_prediction"]["alphafold"]
        models = af2_config["model_name"].split(",")
        self.random_seed = af2_config.get("random_seed", 0)
        self.af2_config = {
            k: v
            for k, v in af2_config.items()
            if v != "model_name" and v != "random_seed"
        }

        # get msa_path
        ptree = get_pathtree(request=self.requests[0])

        str_dict = misc.safe_get(self.requests[0], ["run_config", "msa_select"])
        key_list = list(str_dict.keys())
        for index in range(len(key_list)):
            self.selected_msa_path = ptree.strategy.strategy_list[index]
        if not os.path.exists(self.selected_msa_path):
            logger.info(f"{self.selected_msa_path} doesn't exist, please check")
            return

        # get selected_template_feat
        self.selected_template_feat_path = str(ptree.alphafold.selected_template_feat)

        # run prediction
        # pool = Pool(processes=5)
        # for m_name in models:
        #     pool.apply_async(AirFoldRunner.run_one_model, (m_name))
        # logger.info(f"AirFoldRunner.run_one_model start!")
        # pool.close()
        # pool.join()
        # logger.info(f"AirFoldRunner.run_one_model finished.")
        # Q for multi processing: https://blog.csdn.net/lenfranky/article/details/103975566
        for m_name in models:
            self.run_one_model(m_name=m_name)
            

def alphafold_func(run_stage: str, output_path: str, argument_dict: Dict[str, Any]):
    
    print("------- running stage: %s" % run_stage)
    # set visible gpu device
    gpu_devices = "".join([f"{i}" for i in get_available_gpus(1)])
    logger.info(f"The gpu device used for {run_stage}: {gpu_devices}")
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_devices
    # ref: https://github.com/google-deepmind/alphafold/issues/140
    # for CUDA_ERROR_ILLEGAL_ADDRESS error
    os.system("unset TF_FORCE_UNIFIED_MEMORY")
    
    if run_stage == "search_template":
        pdb_template_hits = search_template(**argument_dict)
        dtool.save_object_as_pickle(pdb_template_hits, output_path)
        return  output_path
    elif run_stage == "make_template_feature":
        template_feature = make_template_feature(**argument_dict)
        dtool.save_object_as_pickle(template_feature, output_path)
        return output_path
    elif run_stage == "monomer_msa2feature":
        template_feat = dtool.read_pickle(argument_dict["template_feature"])
        argument_dict["template_feature"] = template_feat
        processed_feature, _ = monomer_msa2feature(**argument_dict)
        dtool.save_object_as_pickle(processed_feature, output_path)
        return output_path
    elif run_stage == "predict_structure":
        pkl_output = output_path + "_output_raw.pkl"
        pdb_output = output_path + "_unrelaxed.pdb"
        processed_feature = dtool.read_pickle(argument_dict["processed_feature"])
        argument_dict["processed_feature"] = processed_feature
        prediction_results, unrelaxed_pdb_str, _ = predict_structure(**argument_dict)
        dtool.save_object_as_pickle(prediction_results, pkl_output)
        dtool.write_text_file(plaintext=unrelaxed_pdb_str, path=pdb_output)
        return pdb_output
    elif run_stage == "run_relaxation":
        unrelaxed_pdb_str = dtool.read_text_file(argument_dict["unrelaxed_pdb_str"])
        argument_dict["unrelaxed_pdb_str"] = unrelaxed_pdb_str
        relaxed_pdb_str, _ = run_relaxation(**argument_dict)
        dtool.write_text_file(relaxed_pdb_str, output_path)
        return output_path
    else:
        return None
