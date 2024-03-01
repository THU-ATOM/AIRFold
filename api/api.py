from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request
from celery import group, signature
from celery.result import AsyncResult, GroupResult
from worker import celery_client

from pathlib import Path
from typing import Any, Dict, Union, List

import json


# configuration
DEBUG = True
app = FastAPI()
# enable CORS
origins = ["*"]
app.add_middleware(CORSMiddleware, allow_origins=origins)

# ----------------------------
# Single task
# ----------------------------

@app.post("/blast")
async def blast_task(requests: List[Dict[str, Any]]):
    task = celery_client.send_task("blast", args=[requests], queue="queue_blast")
    return {"task_id": task.id}

@app.post("/jackhmmer")
async def jackhmmer_task(requests: List[Dict[str, Any]]):
    task = celery_client.send_task("jackhmmer", args=[requests], queue="queue_jackhmmer")
    return {"task_id": task.id}

@app.post("/hhblits")
async def hhblits_task(requests: List[Dict[str, Any]]):
    task = celery_client.send_task("hhblits", args=[requests], queue="queue_hhblits")
    return {"task_id": task.id}

@app.post("/mergemsa")
async def mergemsa_task(requests: List[Dict[str, Any]]):
    task = celery_client.send_task("mergemsa", args=[requests], queue="queue_mergemsa")
    return {"task_id": task.id}

@app.post("/selectmsa")
async def selectmsa_task(requests: List[Dict[str, Any]]):
    task = celery_client.send_task("selectmsa", args=[requests], queue="queue_selectmsa")
    return {"task_id": task.id}

@app.post("/searchtpl")
async def searchtpl_task(requests: List[Dict[str, Any]]):
    task = celery_client.send_task("searchtpl", args=[requests], queue="queue_searchtpl")
    return {"task_id": task.id}

@app.post("/tplfeature")
async def tplfeature_task(requests: List[Dict[str, Any]]):
    task = celery_client.send_task("tplfeature", args=[requests], queue="queue_tplfeature")
    return {"task_id": task.id}

@app.post("/selecttpl")
async def selecttpl_task(requests: List[Dict[str, Any]]):
    task = celery_client.send_task("selecttpl", args=[requests], queue="queue_selecttpl")
    return {"task_id": task.id}

@app.post("/msafeature")
async def msafeature_task(requests: List[Dict[str, Any]]):
    task = celery_client.send_task("msafeature", args=[requests], queue="queue_msafeature")
    return {"task_id": task.id}

@app.post("/monostructure")
async def monostructure_task(requests: List[Dict[str, Any]]):
    task = celery_client.send_task("monostructure", args=[requests], queue="queue_monostructure")
    return {"task_id": task.id}

@app.post("/relaxation")
async def relaxation_task(requests: List[Dict[str, Any]]):
    task = celery_client.send_task("relaxation", args=[requests], queue="queue_relaxation")
    return {"task_id": task.id}

@app.post("/analysis")
async def analysis_task(requests: List[Dict[str, Any]]):
    task = celery_client.send_task("analysis", args=[requests], queue="queue_analysis")
    return {"task_id": task.id}

@app.post(f"/submit")
async def submit_task(
    request: Request,
    pdb_path: Union[str, Path] = None,
    plddt: float = 0.0,
    target_addresses: str = "",
):
    _params = request.json()
    print("Request: ", _params)
    HASH_ID = "hash_id"
    if _params is None or HASH_ID not in _params:
        return json.dumps([])
    hash_ids = _params[HASH_ID]
    if not isinstance(hash_ids, list):
        hash_ids = [hash_ids]

    task = celery_client.send_task(
        "submit", args=[pdb_path, plddt, target_addresses], queue="queue_submit"
    )
    return {"task_id": task.id}


@app.get("/check/{task_id}")
async def get_task_result(task_id: str):
    task_result = AsyncResult(task_id, app=celery_client)
    if task_result.ready():
        return {
            "task_id": task_id,
            "status": task_result.status,
            "result": task_result.get(),
        }
    else:
        return {
            "task_id": task_id,
            "status": task_result.status,
            "result": "Not ready",
        }


# ----------------------------
# Group task
# ----------------------------


@app.post("/msaGen")
async def msaGen_task(requests: List[Dict[str, Any]]):
    group_task = group(
        signature("blast", args=[requests], queue="queue_blast"),
        signature("jackhmmer", args=[requests], queue="queue_jackhmmer"),
        signature("hhblits", args=[requests], queue="queue_hhblits"),
        # signature("mmseqs", args=[requests], queue="queue_mmseqs"),
    )()
    group_task.save()
    return {"group_task_id": group_task.id}


@app.get("/check_group/{group_task_id}")
async def get_group_task_result(group_task_id: str):
    task_result = GroupResult.restore(group_task_id, app=celery_client)
    if task_result.ready():
        return {
            "group_task_id": group_task_id,
            "status": [task.status for task in task_result],
            "result": task_result.get(),
        }
    else:
        return {
            "group_task_id": group_task_id,
            "status": [task.status for task in task_result],
            "result": "Not ready",
        }
