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


@app.post("/add")
async def add_task(x: int, y: int):
    task = celery_client.send_task("add", args=[x, y], queue="queue_add")
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


@app.post("/blast")
async def blast_task(requests: List[Dict[str, Any]]):
    task = celery_client.send_task("blast", args=[requests], queue="queue_blast")
    return {"task_id": task.id}


@app.post("/jackhmmer")
async def jackhmmer_task(requests: List[Dict[str, Any]]):
    task = celery_client.send_task("jackhmmer", args=[requests], queue="queue_jackhmmer")
    return {"task_id": task.id}


@app.post("/hhblits")
async def blast_task(requests: List[Dict[str, Any]]):
    task = celery_client.send_task("hhblits", args=[requests], queue="queue_hhblits")
    return {"task_id": task.id}


@app.post("/mmseqs")
async def jackhmmer_task(requests: List[Dict[str, Any]]):
    task = celery_client.send_task("mmseqs", args=[requests], queue="queue_mmseqs")
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
async def multiply_and_add_task(request: Dict[str, Any]):
    group_task = group(
        signature("blast", args=[request], queue="queue_blast"),
        signature("jackhmmer", args=[request], queue="queue_jackhmmer"),
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
