import os
from loguru import logger
from copy import deepcopy
from celery import Celery
from pathlib import Path
from typing import Any, Dict, List

import pickle as pkl
import matplotlib.pyplot as plt

from lib.base import BaseRunner, BaseCommandRunner
from lib.state import State
from lib.pathtree import get_pathtree
import lib.utils.datatool as dtool
from lib.monitor import info_report
from lib.utils.execute import rlaunch_exists, rlaunch_wrapper
from lib.tool import plot
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
    TemplateFeaturizationRunner(requests=requests)()
    TPLTSelectRunner(requests=requests)()
    AlphaStrucRunner(requests=requests)()
    AmberRelaxationRunner(requests=requests)()


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
        self.output_path = str(ptree.search.template_hits)
        if not Path(self.output_path).exists():
            Path(self.output_path).parent.mkdir(exist_ok=True, parents=True)

            argument_dict = {
                "input_sequence": self.sequence,
                "template_searching_msa_path": template_searching_msa_path,
                "pdb70_database_path": str(PDB70_ROOT / "pdb70"),
                "hhsearch_binary_path": "hhsearch",
            }
            
            try:
                alphafold_func(run_stage="search_template", 
                            output_path=self.output_path, 
                            argument_dict=argument_dict
                            )
                return True
            except TimeoutError as exc:
                logger.exception(exc)
                return False

    def on_run_end(self):
        if self.info_reportor is not None:
            for request in self.requests:
                if Path(self.output_path).exists():
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
        template_feats_path = str(ptree.search.template_feat)
        self.output_path = ptree.search.selected_template_feat
        Path(self.output_path).parent.mkdir(exist_ok=True, parents=True)
        if not Path(self.output_path).exists():
            if (
                "template_select_strategy" in request["run_config"]["template"]
                and request["run_config"]["template"]["template_select_strategy"] == "none"
            ):
                template_feats = dtool.read_pickle(template_feats_path)
                with open(self.output_path, "wb") as fd:
                    pkl.dump(template_feats, fd)
            else:
                self.input_path = template_feats_path
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

        self.output_path = str(ptree.search.template_feat)
        Path(self.output_path).parent.mkdir(exist_ok=True, parents=True)
        if not Path(self.output_path).exists():

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
                alphafold_func(run_stage="make_template_feature", 
                            output_path=self.output_path, 
                            argument_dict=argument_dict
                            )

                return True
            except TimeoutError as exc:
                logger.exception(exc)
                return False


    def on_run_end(self):
        if self.info_reportor is not None:
            for request in self.requests:
                if Path(self.output_path).exists():
                    self.info_reportor.update_state(
                        hash_id=request[info_report.HASH_ID],
                        state=self.success_code,
                    )

class AlphaStrucRunner(BaseRunner):
    def __init__(
        self,
        requests: List[Dict[str, Any]]
    ) -> None:
        super().__init__(requests)
        self.error_code = State.AlphaFold_ERROR
        self.success_code = State.AlphaFold_SUCCESS
        self.start_code = State.AlphaFold_START
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

    def run(self):
        ptree = get_pathtree(request=self.requests[0])
        # get msa_path
        str_dict = misc.safe_get(self.requests[0], ["run_config", "msa_select"])
        key_list = list(str_dict.keys())
        msa_paths = []
        for idx in range(len(key_list)):
            selected_msa_path = str(ptree.strategy.strategy_list[idx]) + "_dp.a3m"
            msa_paths.append(str(selected_msa_path))
        
        dtool.deduplicate_msa_a3m(msa_paths, str(ptree.alphafold.input_a3m))
        
        msa_image = ptree.alphafold.msa_coverage_image
        Path(msa_image).parent.mkdir(exist_ok=True, parents=True)
        self.save_msa_fig_from_a3m_files(
            msa_paths=msa_paths,
            save_path=msa_image,
        )
        # get selected_template_feat
        selected_template_feat_path = str(ptree.search.selected_template_feat)
        
        af2_config = self.requests[0]["run_config"]["structure_prediction"]["alphafold"]
        models = af2_config["model_name"].split(",")
        random_seed = af2_config.get("random_seed", 0)
        
        self.output_paths = []
        for model_name in models:
            out_preffix = str(os.path.join(str(ptree.alphafold.root), model_name))
            out_path = str(os.path.join(str(ptree.alphafold.root), model_name)) + "_unrelaxed.pdb"
            if not os.path.exists(out_path):
                fea_output_path = str(ptree.alphafold.processed_feat) + f"_{model_name}.pkl"
                template_feat = dtool.read_pickle(selected_template_feat_path)
                argument_dict1 = {
                    "sequence": self.sequence,
                    "target_name": self.target_name,
                    "msa_paths": msa_paths,
                    "template_feature": template_feat,
                    "model_name": model_name,
                    "random_seed": random_seed,
                }
                argument_dict1 = deepcopy(argument_dict1)
                for k, v in af2_config.items():
                    if k not in argument_dict1:
                        argument_dict1[k] = v
                
                processed_feature = alphafold_func(run_stage="monomer_msa2feature", 
                                                output_path=fea_output_path, 
                                                argument_dict=argument_dict1
                                                )
                
                argument_dict2 = {
                    "target_name": self.target_name,
                    "processed_feature": processed_feature,
                    "model_name": model_name,
                    "data_dir": str(AF_PARAMS_ROOT),
                    "random_seed": random_seed,
                    "return_representations": True,
                }
                argument_dict2 = deepcopy(argument_dict2)
                for k, v in af2_config.items():
                    if k not in argument_dict2:
                        argument_dict2[k] = v
                try:
                    pdb_output = alphafold_func(run_stage="predict_structure", 
                                                output_path=out_preffix, 
                                                argument_dict=argument_dict2
                                                )
                    self.output_paths.append(pdb_output)
                except TimeoutError as exc:
                    logger.exception(exc)
                    return False
        

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

    def run(self):
        ptree = get_pathtree(request=self.requests[0])
        af2_config = self.requests[0]["run_config"]["structure_prediction"]["alphafold"]
        models = af2_config["model_name"].split(",")
        self.output_paths = []
        for model_name in models:
            input_path = str(os.path.join(str(ptree.alphafold.root), model_name)) + "_unrelaxed.pdb"
            unrelaxed_pdb_str = dtool.read_text_file(input_path)
            output_path = str(os.path.join(str(ptree.alphafold.root), model_name)) + "_relaxed.pdb"
            argument_dict = {"unrelaxed_pdb_str": unrelaxed_pdb_str}
            if not os.path.exists(output_path):
                try:
                    relaxed_pdb_path = alphafold_func(run_stage="run_relaxation", 
                                                output_path=output_path, 
                                                argument_dict=argument_dict
                                                )
                    self.output_paths.append(relaxed_pdb_path)
                except TimeoutError as exc:
                    logger.exception(exc)
                    return False


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



def alphafold_func(run_stage: str, output_path: str, argument_dict: Dict[str, Any]):
    
    print("------- running stage: %s" % run_stage)
    # set visible gpu device
    gpu_devices = "".join([f"{i}" for i in get_available_gpus(1)])
    logger.info(f"The gpu device used for {run_stage}: {gpu_devices}")
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_devices
    # ref: https://github.com/google-deepmind/alphafold/issues/140
    # for CUDA_ERROR_ILLEGAL_ADDRESS error
    # os.system("unset TF_FORCE_UNIFIED_MEMORY")
    
    if run_stage == "search_template":
        pdb_template_hits = search_template(**argument_dict)
        dtool.save_object_as_pickle(pdb_template_hits, output_path)
        return  output_path
    elif run_stage == "make_template_feature":
        template_feature = make_template_feature(**argument_dict)
        dtool.save_object_as_pickle(template_feature, output_path)
        return output_path
    elif run_stage == "monomer_msa2feature":
        processed_feature, _ = monomer_msa2feature(**argument_dict)
        # dtool.save_object_as_pickle(processed_feature, output_path)
        return processed_feature
    elif run_stage == "predict_structure":
        pkl_output = output_path + "_output_raw.pkl"
        pdb_output = output_path + "_unrelaxed.pdb"
        prediction_results, unrelaxed_pdb_str, _ = predict_structure(**argument_dict)
        dtool.save_object_as_pickle(prediction_results, pkl_output)
        dtool.write_text_file(plaintext=unrelaxed_pdb_str, path=pdb_output)
        return pdb_output
    elif run_stage == "run_relaxation":
        relaxed_pdb_str, _ = run_relaxation(**argument_dict)
        dtool.write_text_file(relaxed_pdb_str, output_path)
        return output_path
    else:
        return None
