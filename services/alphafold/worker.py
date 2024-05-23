import os
from celery import Celery

from typing import Any, Dict
# from loguru import logger

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
def alphafoldTask(run_stage: str, argument_dict: Dict[str, Any]):
    # run_stage
    # enum_values=[
    #     "search_template",
    #     "make_template_feature",
    #     "monomer_msa2feature",
    #     "predict_structure",
    #     "run_relaxation",
    # ]
    if run_stage == "search_template":
        results = search_template(**argument_dict)
    elif run_stage == "make_template_feature":
        results = make_template_feature(**argument_dict)
    elif run_stage == "monomer_msa2feature":
        results = monomer_msa2feature(**argument_dict)
    elif run_stage == "predict_structure":
        results = predict_structure(**argument_dict)
    elif run_stage == "run_relaxation":
        results = run_relaxation(**argument_dict)
    else:
        results = None
    return results
