from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request
from celery.result import AsyncResult
from worker import celery

from pathlib import Path
from typing import Any, Dict, Union

import json


# configuration
DEBUG = True
app = FastAPI()
# enable CORS
origins = ["*"]
app.add_middleware(CORSMiddleware, allow_origins=origins)


@app.post("/add")
async def add_task(x: int, y: int):
    task = celery.send_task("add", args=[x, y], queue="queue_add")
    return {"task_id": task.id}


@app.post(f"/update/submit")
async def submit_task(
    request: Request, pdb_path: Union[str, Path] = None, plddt: float = 0.0, target_addresses: str = ""
):
    _params = request.json()
    print("Request: ", _params)
    HASH_ID = "hash_id"
    if _params is None or HASH_ID not in _params:
        return json.dumps([])
    hash_ids = _params[HASH_ID]
    if not isinstance(hash_ids, list):
        hash_ids = [hash_ids]

    task = celery.send_task("submit", args=[pdb_path, plddt, target_addresses], queue="queue_submit")
    return {"task_id": task.id}


@app.post("/blast")
async def blast_task(request: Dict[str, Any]):
    task = celery.send_task("blast", args=[request], queue="queue_blast")
    return {"task_id": task.id}


@app.get("/check/{task_id}")
async def get_task_result(task_id: str):
    task_result = AsyncResult(task_id, app=celery)
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
