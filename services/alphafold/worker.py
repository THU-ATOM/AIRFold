import os
from loguru import logger
from celery import Celery
from copy import deepcopy
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
from lib.utils import misc, pathtool
from lib.constant import PDB70_ROOT, PDBMMCIF_ROOT
from lib.tool.run_af2_stage import (
    search_template, 
    make_template_feature
)
from lib.tool import run_af2_monomer, run_af2_multimer

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
    request = requests[0]
    multimer = misc.safe_get(requests[0], ["multimer"]) if misc.safe_get(requests[0], "multimer") else False
    if not multimer:
        TemplateSearchRunner(requests=requests)()
        TemplateFeaturizationRunner(requests=requests)()
        TPLTSelectRunner(requests=requests)()
        AlphaMonomerStrucRunner(requests=requests)()
        # AmberRelaxationRunner(requests=requests)()
    else:
        chain_requests_list = split_chain_requests(requests)
        for chain_requests in chain_requests_list:
            TemplateSearchRunner(requests=chain_requests)()
            TemplateFeaturizationRunner(requests=chain_requests)()
            TPLTSelectRunner(requests=chain_requests)()
        AlphaMultimerStrucRunner(requests=requests)()


def split_chain_requests(requests: List[Dict[str, Any]]):
    requests_list = []
    request = requests[0]
    sequence = misc.safe_get(request, ["sequence"])
    seq_list = sequence.split("\n")
    for chain_id, seq in enumerate(seq_list):
        chain_request = deepcopy(request)
        chain_request["sequence"] = seq
        chain_request["multimer"] = False
        chain_request["name"] = chain_request["name"] + "_chain_" + str(chain_id)
        chain_request["target"] = chain_request["target"] + "_chain_" + str(chain_id)
        requests_list.append([chain_request])
    return requests_list


def split_chain_request(request: Dict[str, Any]):
    request_list = []
    sequence = misc.safe_get(request, ["sequence"])
    seq_list = sequence.split("\n")
    for chain_id, seq in enumerate(seq_list):
        chain_request = deepcopy(request)
        chain_request["sequence"] = seq
        chain_request["multimer"] = False
        chain_request["name"] = chain_request["name"] + "_chain_" + str(chain_id)
        chain_request["target"] = chain_request["target"] + "_chain_" + str(chain_id)
        request_list.append(chain_request)
    return request_list


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
                pdb_template_hits = search_template(**argument_dict)
                dtool.save_object_as_pickle(pdb_template_hits, self.output_path)
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
                template_feature = make_template_feature(**argument_dict)
                dtool.save_object_as_pickle(template_feature, self.output_path)
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

class AlphaMonomerStrucRunner(BaseCommandRunner):
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

    def build_command(self, request: Dict[str, Any]) -> str:
        ptree = get_pathtree(request)
        # get msa_path
        str_dict = misc.safe_get(request, ["run_config", "msa_select"])
        key_list = list(str_dict.keys())
        msa_paths = []
        for idx in range(len(key_list)):
            selected_msa_path = str(ptree.strategy.strategy_list[idx]) + "_dp.a3m"
            msa_paths.append(str(selected_msa_path))
        
        
        msa_image = ptree.alphafold.msa_coverage_image
        Path(msa_image).parent.mkdir(exist_ok=True, parents=True)
        dtool.deduplicate_msa_a3m(msa_paths, str(ptree.alphafold.input_a3m))
        self.save_msa_fig_from_a3m_files(
            msa_paths=msa_paths,
            save_path=msa_image,
        )
        
        af2_config = request["run_config"]["structure_prediction"]["alphafold"]
        models = af2_config["model_name"].split(",")
        random_seed = af2_config.get("random_seed", 0)
        
        self.output_paths = []
        commands = []
        for model_name in models:
            pdb_output = str(os.path.join(ptree.alphafold.root, model_name)) + "_relaxed.pdb"
            self.output_paths.append(pdb_output)
            command = "".join(
                [
                    f"python {pathtool.get_module_path(run_af2_monomer)} ",
                    f"--sequence {self.sequence} ",
                    f"--target_name {self.target_name} ",
                    f"--model_name {model_name} ",
                    f"--root_path {str(ptree.alphafold.root)} ",
                    f"--a3m_path {str(ptree.alphafold.input_a3m)} ",
                    f"--template_feat {str(ptree.search.selected_template_feat)} ",
                    f"--random_seed {random_seed} ",
                    # AF2 Params
                    f"--seqcov {misc.safe_get(af2_config, 'seqcov')} "
                    if misc.safe_get(af2_config, "seqcov")
                    else "",
                    f"--seqqid {misc.safe_get(af2_config, 'seqqid')} "
                    if misc.safe_get(af2_config, "seqqid")
                    else "",
                    f"--max_recycles {misc.safe_get(af2_config, 'max_recycles')} "
                    if misc.safe_get(af2_config, "max_recycles")
                    else "",
                    f"--max_msa_clusters {misc.safe_get(af2_config, 'max_msa_clusters')} "
                    if misc.safe_get(af2_config, "max_msa_clusters")
                    else "",
                    f"--max_extra_msa {misc.safe_get(af2_config, 'max_extra_msa')} "
                    if misc.safe_get(af2_config, "max_extra_msa")
                    else "",
                    f"--num_ensemble {misc.safe_get(af2_config, 'num_ensemble')} "
                    if misc.safe_get(af2_config, "num_ensemble")
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


class AlphaMultimerStrucRunner(BaseCommandRunner):
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

    def build_command(self, request: Dict[str, Any]) -> str:
        request_list = split_chain_request(request)
        sequences = []
        targets = []
        input_msa_list = []
        input_uniprot_msa_list = []
        input_template_fea_list = []
        for chain_request in request_list:
            sequences.append(chain_request[SEQUENCE])
            targets.append(chain_request[TARGET])

            ptree = get_pathtree(chain_request)
            # get msa_path
            str_dict = misc.safe_get(chain_request, ["run_config", "msa_select"])
            key_list = list(str_dict.keys())
            chain_msa_paths = []
            for idx in range(len(key_list)):
                selected_msa_path = str(ptree.strategy.strategy_list[idx]) + "_dp.a3m"
                chain_msa_paths.append(str(selected_msa_path))
            
            msa_image = ptree.alphafold.msa_coverage_image
            Path(msa_image).parent.mkdir(exist_ok=True, parents=True)

            # merge selected msa
            dtool.deduplicate_msa_a3m(chain_msa_paths, str(ptree.alphafold.input_a3m))
            input_msa_list.append(str(ptree.alphafold.input_a3m))

            # get uniprot msa
            dtool.deduplicate_msa_a3m([str(ptree.search.jackhammer_uniprot_a3m)], str(ptree.alphafold.input_uniprot_a3m))
            input_uniprot_msa_list.append(str(ptree.alphafold.input_uniprot_a3m))

            # get selected template feature
            input_template_fea_list.append(str(ptree.search.selected_template_feat))

            self.save_msa_fig_from_a3m_files(
                msa_paths=chain_msa_paths,
                save_path=msa_image,
            )
        
        af2_config = request["run_config"]["structure_prediction"]["alphafold"]
        models = af2_config["model_name"].split(",")
        random_seed = af2_config.get("random_seed", 0)
        
        self.output_paths = []
        commands = []
        for model_name in models:
            pdb_output = str(os.path.join(ptree.alphafold.root, model_name)) + "_relaxed.pdb"
            self.output_paths.append(pdb_output)
            command = "".join(
                [
                    f"python {pathtool.get_module_path(run_af2_multimer)} ",
                    f"--target_name {self.target_name} ",
                    f"--chain_sequences {' '.join(sequences)} ",
                    f"--chain_targets {' '.join(targets)} ",
                    f"--model_name {model_name} ",
                    f"--root_path {str(ptree.alphafold.root)} ",
                    f"--a3m_paths {' '.join(input_msa_list)} ",
                    f"--uniprot_a3m_paths {' '.join(input_uniprot_msa_list)} ",
                    f"--template_feats {' '.join(input_template_fea_list)} ",
                    f"--random_seed {random_seed} ",
                    # AF2 Params
                    f"--seqcov {misc.safe_get(af2_config, 'seqcov')} "
                    if misc.safe_get(af2_config, "seqcov")
                    else "",
                    f"--seqqid {misc.safe_get(af2_config, 'seqqid')} "
                    if misc.safe_get(af2_config, "seqqid")
                    else "",
                    f"--max_recycles {misc.safe_get(af2_config, 'max_recycles')} "
                    if misc.safe_get(af2_config, "max_recycles")
                    else "",
                    f"--max_msa_clusters {misc.safe_get(af2_config, 'max_msa_clusters')} "
                    if misc.safe_get(af2_config, "max_msa_clusters")
                    else "",
                    f"--max_extra_msa {misc.safe_get(af2_config, 'max_extra_msa')} "
                    if misc.safe_get(af2_config, "max_extra_msa")
                    else "",
                    f"--num_ensemble {misc.safe_get(af2_config, 'num_ensemble')} "
                    if misc.safe_get(af2_config, "num_ensemble")
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