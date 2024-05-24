import os
from celery import Celery

from typing import Any, Dict
import lib.utils.datatool as dtool
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

celery = Celery(
    __name__,
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
)

celery.conf.task_routes = {
    "worker.*": {"queue": "queue_alphafold"},
}


@celery.task(name="alphafold")
def alphafoldTask(run_stage: str, output_path: str, argument_dict: Dict[str, Any]):
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
        dtool.save_object_as_pickle(processed_feature, output_path)
        return output_path
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
