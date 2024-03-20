import os
import glob
from celery import Celery

from pathlib import Path
from typing import Any, Dict, List, OrderedDict, Union
import matplotlib.pyplot as plt
import numpy as np
import Bio.PDB
from scipy.special import softmax
from loguru import logger

from lib.base import BaseRunner
from lib.constant import DB_PATH
from lib.state import State
from lib.pathtree import get_pathtree
from lib.monitor import info_report
from lib.tool import plot
import lib.utils.datatool as dtool
from lib.tool.colabfold.alphafold.common import protein



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
    "worker.*": {"queue": "queue_analysis"},
}


@celery.task(name="analysis")
def analysisTask(requests: List[Dict[str, Any]]):
    GenAnalysisRunner(requests=requests, db_path=DB_PATH)()


class GenAnalysisRunner(BaseRunner):
    def __init__(
        self,
        requests: List[Dict[str, Any]],
        db_path: Union[str, Path] = None,
    ) -> None:
        super().__init__(requests, db_path)
        self.stage = State.ANALYSIS_GEN

    @property
    def start_stage(self) -> State:
        return self.stage

    @staticmethod
    def parse_save_plddt_from_dir(dirname, pdb_pattern="rank_*_relaxed.pdb"):
        paths = glob.glob(os.path.join(dirname, pdb_pattern))
        if len(paths) == 0:
            raise ValueError
        model2plddts = {}
        m2plddt = {}
        for p in paths:
            model_name = os.path.basename(p).replace("_relaxed.pdb", "")
            with open(p) as fd:
                prot = protein.from_pdb_string(fd.read())
            model2plddts[model_name] = prot.b_factors[:, 0]
            m2plddt[model_name] = np.mean(prot.b_factors[:, 0])
        model_names, plddt = list(
            zip(*sorted(m2plddt.items(), key=lambda x: x[-1], reverse=True))
        )
        model2plddts = OrderedDict(
            [(mname, model2plddts[mname]) for mname in model_names]
        )

        plot.plot_plddts(model2plddts)
        plt.savefig(
            f'{os.path.join(dirname, "predicted_LDDT.png")}',
            bbox_inches="tight",
            dpi=200,
        )

    @staticmethod
    def parse_save_disto_contact_from_dir(dirname, file_pattern, num_res):
        to_np = lambda a: np.asarray(a)
        dist_matrices = []
        contact_matrices = []
        plddt_scores = []
        for p in glob.glob(os.path.join(dirname, file_pattern)):
            prediction_result = dtool.read_pickle(p)
            dist_bins = np.append(0, prediction_result["distogram"]["bin_edges"])
            dist_logits = prediction_result["distogram"]["logits"][:num_res, :][
                :, :num_res
            ]
            dist_mtx = dist_bins[dist_logits.argmax(-1)]
            contact_mtx = softmax(dist_logits, axis=-1)[:, :, dist_bins < 8].sum(-1)

            dist_mtx = to_np(dist_mtx)
            contact_mtx = to_np(contact_mtx)
            dist_matrices.append(dist_mtx)
            contact_matrices.append(contact_mtx)
            plddt_scores.append(prediction_result["ranking_confidence"])

        orderd_indexes = sorted(
            [(i, plddt) for i, plddt in enumerate(plddt_scores)],
            key=lambda x: x[-1],
            reverse=True,
        )
        dist_matrices = [dist_matrices[i] for i, _ in orderd_indexes]
        contact_matrices = [contact_matrices[i] for i, _ in orderd_indexes]
        plot.plot_adjs(contact_matrices)
        plt.savefig(
            os.path.join(dirname, f"predicted_contacts.png"),
            bbox_inches="tight",
            dpi=300,
        )
        plt.show()

        plot.plot_dists(dist_matrices)
        plt.savefig(
            os.path.join(dirname, f"predicted_distogram.png"),
            bbox_inches="tight",
            dpi=300,
        )
        plt.show()

    @staticmethod
    def get_conformation_from_dir(dirname):
        def get_disogram(pkl_path):
            data = dtool.read_pickle(pkl_path)
            return data["distogram"]

        def distogram_plot(disto, num_res):
            to_np = lambda a: np.asarray(a)
            dist_bins = np.append(0, disto["bin_edges"])
            dist_logits = disto["logits"][:num_res, :][:, :num_res]
            dist_mtx = dist_bins[dist_logits.argmax(-1)]
            dist_mtx = to_np(dist_mtx)
            return dist_mtx

        def plot_dists(dists, name, dpi=100, fig=True):
            title_dict = [
                "Distances (from distogram)",
                "Distances (from structure)",
                "Unrealized distances",
            ]
            num_models = len(dists)
            if fig:
                plt.figure(figsize=(3 * num_models, 2), dpi=dpi)
            for n, dist in enumerate(dists):
                plt.subplot(1, num_models, n + 1)
                plt.title(title_dict[n])
                Ln = dist.shape[0]
                plt.imshow(dist, extent=(0, Ln, Ln, 0))
                plt.colorbar()
            plt.savefig(
                os.path.join(dirname, f"{name}_conformation.png"),
                bbox_inches="tight",
                dpi=300,
            )
            return plt

        def calc_dist_matrix(chain):
            """Returns a matrix of C-alpha distances between two chains"""
            answer = np.zeros((len(chain), len(chain)), dtype=float)
            for row, residue_one in enumerate(chain):
                for col, residue_two in enumerate(chain):
                    if (
                        residue_one.get_resname() != "GLY"
                        and residue_two.get_resname() != "GLY"
                    ):
                        answer[row, col] = residue_one["CB"] - residue_two["CB"]
                    else:
                        answer[row, col] = residue_one["CA"] - residue_two["CA"]
            return answer

        def pdb2distogram(pdb, bin_edges):
            structure = Bio.PDB.PDBParser(QUIET=True).get_structure("rank1", pdb)
            for model in structure:
                for chain in model:
                    # info_chain(chain)
                    dist6d = calc_dist_matrix(chain)
            dist_bins = np.append(0, bin_edges)
            digit = np.digitize(dist6d, dist_bins) - 1
            return dist_bins[digit], dist_bins[digit].shape[0]

        def conformation_pipeline(pkl_path, pdb_path, name):

            pred_disto = get_disogram(pkl_path)
            structure_mtx, num_res = pdb2distogram(pdb_path, pred_disto["bin_edges"])
            dist_mtx = distogram_plot(pred_disto, num_res)  # expected distogram plot

            # print(structure_mtx.shape)
            plot_dists(
                [dist_mtx, structure_mtx, np.abs(dist_mtx - structure_mtx)],
                name,
            )

        pdb_list = Path(dirname).glob("rank_*_relaxed.pdb")
        for pdb_path in pdb_list:
            model_num = pdb_path.name.split("_")[3]
            pkl_path = "model_{}_output_raw.pkl".format(model_num)
            conformation_pipeline(Path(dirname) / pkl_path, pdb_path, pdb_path.stem)

    @staticmethod
    def get_template_info_from_ptree(ptree):
        def get_selected_template_info(pkl_path):
            data = dtool.read_pickle(pkl_path)
            return [
                (n.decode("utf-8") if not isinstance(n, str) else n, s[0])
                for n, s in zip(
                    data["template_domain_names"].tolist(),
                    data["template_sum_probs"].tolist(),
                )
            ]

        def get_searched_template_info(pkl_path):
            data = dtool.read_pickle(pkl_path)
            return [(k["name"].split()[0], k["sum_probs"]) for k in data]

        template_feat_pkl_path = str(ptree.alphafold.template_feat)
        selected_template_pkl_path = str(ptree.alphafold.selected_template_feat)
        all_searched_template_pkl_path = str(ptree.search.template_hits)

        ret_dict = {}
        if os.path.exists(template_feat_pkl_path):
            ret_dict["template_info"] = get_selected_template_info(
                template_feat_pkl_path
            )
        if os.path.exists(selected_template_pkl_path):
            ret_dict["selected_template_info"] = get_selected_template_info(
                selected_template_pkl_path
            )
        if os.path.exists(all_searched_template_pkl_path):
            ret_dict["searched_template_info"] = get_searched_template_info(
                all_searched_template_pkl_path
            )
        return ret_dict

    @staticmethod
    def get_plddts_from_dir(dirname, pdb_pattern="rank_*_relaxed.pdb"):
        paths = glob.glob(os.path.join(dirname, pdb_pattern))
        if len(paths) == 0:
            raise ValueError
        model2plddts = {}
        model2pdbpaths = {}
        for p in paths:
            model_name = os.path.basename(p).replace("_relaxed.pdb", "")
            with open(p) as fd:
                prot = protein.from_pdb_string(fd.read())
            model2plddts[model_name] = np.mean(prot.b_factors[:, 0])
            model2pdbpaths[model_name] = p
        # logger.info(json.dumps(paths))
        # logger.info(f'metric update with: {json.dumps(model2plddts)}')
        return model2plddts, model2pdbpaths

    def rerank_relaxed_plddts(self, dirname):
        os.system(f'rm -rf {os.path.join(dirname, "rank_*_relaxed.pdb")}')
        model2plddts, model2pdbpaths = self.get_plddts_from_dir(
            dirname=dirname, pdb_pattern="model_*_relaxed.pdb"
        )
        model_plddt_tuple = sorted(
            model2plddts.items(), key=lambda x: x[-1], reverse=True
        )
        for i, (model_name, plddt) in enumerate(model_plddt_tuple):
            src_path = model2pdbpaths[model_name]
            tgt_path = os.path.join(
                dirname, f"rank_{i+1}_{model_name}_seed_0_relaxed.pdb"
            )
            os.system(f"cp {src_path} {tgt_path}")

    def run(self):
        ptree = get_pathtree(request=self.requests[0])
        num_res = len(self.requests[0]["sequence"])
        dirname = str(ptree.alphafold.root)
        logger.info(f"Analysis dirname: {dirname}")
        self.rerank_relaxed_plddts(dirname=dirname)
        self.parse_save_plddt_from_dir(dirname=dirname)
        self.get_conformation_from_dir(dirname=dirname)
        self.parse_save_disto_contact_from_dir(
            dirname=dirname,
            file_pattern="model_*_output_raw.pkl",
            num_res=num_res,
        )
        model2plddts, _ = self.get_plddts_from_dir(
            dirname=dirname, pdb_pattern="rank_*.pdb"
        )
        template_infos = self.get_template_info_from_ptree(ptree)
        self.info_reportor.update_reserved(
            hash_id=self.requests[0][info_report.HASH_ID], update_dict=template_infos
        )
        self.info_reportor.update_metric(
            hash_id=self.requests[0][info_report.HASH_ID],
            value=model2plddts,
            metric="plddt",
        )

        self.info_reportor.update_path_tree(
            hash_id=self.requests[0][info_report.HASH_ID], path_tree=ptree.tree
        )
        result_path = os.path.join(dirname, "plddt_results.json")
        dtool.write_json(result_path, data=model2plddts)
        return str(result_path)