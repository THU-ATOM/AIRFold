import os
from celery import Celery

from pathlib import Path
from typing import Any, Dict, List, Union

from lib.base import BaseCommandRunner
from lib.state import State
from lib.pathtree import get_pathtree
from lib.utils import misc, pathtool
from lib.monitor import info_report

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
    "worker.*": {"queue": "queue_blast"},
}

DB_PATH = Path("/data/protein/CAMEO/database/cameo_test.db")

@celery.task(name="blast")
# def blast(request: Dict[str, Any]):
def blast(requess: Dict[str, Any]):
    request = {
  "email": "sd-m__2024-01-06_00000180__2-127@proteinmodelportal.org",
  "sender": "cameo",
  "sequence": "MDFFNKFSQGLAESSTPKSSIYYSEEKDPDTKKDEAIEIGLKSQESYYQRQLREQLARDNMTVASRQPIQPLQPTIHITPQPVPTATPAPILLPSSTVPTPKPRQQTNTSSDMSNLFDWLSEDTDAPASSLLPALTPSNAVQDIISKFNKDQKTTTPPSTQPSQTLPTTTCTQQSDGNISCTTPTVTPPQPPIVATVCTPTPTGGTVCTTAQQNPNPGAASQQNLDDMALKDLMSNVERDMHQLQAETNDLVTNVYDAREYTRRAIDQILQLVKGFERFQK",
  "name": "2024-01-06_00000180_2_127___rdTu",
  "target": "2024-01-06_00000180_2_127",
  "run_config": {
    "name": "cameo",
    "msa_search": {
      "segment": 'null',
      "copy_int_msa_from": 'null',
      "hhblits": {
        "iteration": 3,
        "e_value": 0.001,
        "realign_max": 100000,
        "maxfilt": 100000,
        "min_prefilter_hits": 1000,
        "maxseq": 100000,
        "dataset": [
          "uniclust30",
          "bfd"
        ],
        "diff_default": "inf",
        "diff_fast": 1000,
        "timeout": 7200,
        "thread": 8,
        "cpu": 8
      },
      "jackhmmer": {
        "n_iter": 1,
        "e_value": 0.0001,
        "filter_f1": 0.0005,
        "filter_f2": 0.00005,
        "filter_f3": 0.000005,
        "thread": 8,
        "cpu": 8
      },
      "blast": {
        "blasttype": "psiblast",
        "evalue": 0.001,
        "num_iterations": 3
      }
    },
    "template": {
      "copy_template_hits_from": 'null',
      "cutomized_template_pdbs": 'null',
      "template_select_strategy": "top"
    },
    "msa_select": {
      "seq_entropy": {
        "reduce_ratio": 0.1,
        "least_seqs": 5000
      }
    },
    "structure_prediction": {
      "alphafold": {
        "seqcov": 0,
        "seqqid": 0,
        "max_recycles": 128,
        "max_msa_clusters": 508,
        "max_extra_msa": 5120,
        "num_ensemble": 1,
        "model_name": "model_1,model_2,model_3,model_4,model_5",
        "random_seed": 0
      }
    }
  },
  "submit": "False"
}
    requests = [request]
    command = BlastRunner(requests=requests, db_path=DB_PATH).run()
    return command



class BlastRunner(BaseCommandRunner):
    def __init__(
        self, requests: List[Dict[str, Any]], db_path: Union[str, Path] = None
    ):
        super().__init__(requests, db_path)

    @property
    def start_stage(self) -> int:
        return State.BLAST_START

    def build_command(self, request: Dict[str, Any]) -> str:
        ptree = get_pathtree(request=request)

        args = misc.safe_get(request, ["run_config", "msa_search", "search", "blast"])

        command = "".join(
            [
                f"python {pathtool.get_module_path(blast)} ",
                f"-i {ptree.seq.fasta} ",
                f"-o {ptree.search.blast_a3m} ",
                f"-w {ptree.search.blast_whole_fa} ",
                #  parser.add_argument("-e", "--evalue", type=float, default=1e-5)
                f"-e {misc.safe_get(args, 'evalue')} "
                if misc.safe_get(args, "evalue")
                else "",
                f"-n {misc.safe_get(args, 'num_iterations')} "
                if misc.safe_get(args, "num_iterations")
                else "",
                f"-b {misc.safe_get(args, 'blasttype')} "
                if misc.safe_get(args, "blasttype")
                else "",
            ]
        )
        # command = (
        #     f"python {get_module_path(blast)} "
        #     f"-i={ptree.seq.fasta} "
        #     f"-o={ptree.search.blast_a3m} "
        #     # f"-fo={ptree.search.mmseqs_fa} "
        # )
        return command

    def on_run_end(self):
        if self.info_reportor is not None:
            for request in self.requests:
                tree = get_pathtree(request=request)
                if tree.search.blast_a3m.exists():
                    self.info_reportor.update_state(
                        hash_id=request[info_report.HASH_ID],
                        state=State.BLAST_SUCCESS,
                    )
                else:
                    self.info_reportor.update_state(
                        hash_id=request[info_report.HASH_ID],
                        state=State.BLAST_ERROR,
                    )