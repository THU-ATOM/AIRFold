import os
import random
import time
from loguru import logger
from celery import Celery

from typing import Any, Dict
import lib.utils.datatool as dtool
from gpustat import new_query

from lib.tool.run_af2_stage import (
    search_template, 
    make_template_feature, 
    monomer_msa2feature, 
    predict_structure,
    run_relaxation,
)

REFRESH_SECONDS = 30
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
    print("------- running stage: %s" % run_stage)
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
        # set visible gpu device
        gpu_devices = "".join([f"{i}" for i in get_available_gpus(1)])
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_devices
        
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


def get_available_gpus(
    num: int = -1,
    min_memory: int = 20000,
    random_select: bool = True,
    wait_time: float = float("inf"),
):
    """Get available GPUs.

    Parameters
    ----------
    num : int, optional
        Number of GPUs to get. The default is -1.
    min_memory : int, optional
        Minimum memory available in GB. The default is 20000.
    random_select : bool, optional
        Random select a GPU. The default is True.
    wait_time : float, optional
        Wait time in seconds. The default is inf.
    """

    start = time.time()
    while time.time() - start < wait_time:
        gpu_list = new_query().gpus
        if random_select:
            random.shuffle(gpu_list)
        sorted_gpu_list = sorted(
            gpu_list,
            key=lambda card: (
                card.entry["utilization.gpu"],
                card.entry["memory.used"],
            ),
        )
        available_gpus = [
            gpu.entry["index"]
            for gpu in sorted_gpu_list
            if gpu.entry["memory.total"] - gpu.entry["memory.used"]
            >= min_memory
        ]
        if num > 0:
            available_gpus = available_gpus[:num]
        if len(available_gpus) > 0:
            return available_gpus
        else:
            logger.info(
                f"No GPU available, having waited {time.time() - start} seconds"
            )
            time.sleep(REFRESH_SECONDS)
    raise Exception("No GPU available")