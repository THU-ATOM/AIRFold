from celery import group, signature, chord
from celery.result import AsyncResult, GroupResult
from worker import celery_client

from pathlib import Path
from typing import Any, Dict, List
from loguru import logger
from hashlib import sha256

# import asyncio
import requests
import json, sqlite3, psutil
from io import StringIO
from pathlib import Path
import pandas as pd

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request, Body
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

from lib.monitor.database_mgr import StateRecord
from lib.monitor.extend_config import (
    extend_run_config,
    generate_default_config,
)
from lib.state import State
from lib.monitor.info_report import *
from lib.constant import *
from lib.tool.align import align_pdbs
import lib.monitor.download_pdb as download_pdb

info_retriever = InfoRetrieve(db_path=DB_PATH)
info_report = InfoReport(db_path=DB_PATH)

cameo_api = "https://www.cameo3d.org/modeling/targets/1-month/ajax/?to_date="
casp_target_list = "https://predictioncenter.org/casp15/targetlist.cgi?type=csv"

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

@app.post("/monostructure")
async def monostructure_task(requests: List[Dict[str, Any]]):
    task = celery_client.send_task("monostructure", args=[requests], queue="queue_monostructure")
    return {"task_id": task.id}

@app.post("/analysis")
async def analysis_task(requests: List[Dict[str, Any]]):
    task = celery_client.send_task("analysis", args=[requests], queue="queue_analysis")
    return {"task_id": task.id}

@app.post(f"/submit")
async def submit_task(requests: List[Dict[str, Any]]):
    task = celery_client.send_task("submit", args=[requests], queue="queue_submit")
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
# Group/Graph task
# ----------------------------


@app.post("/msaGen")
async def msaGen_task(requests: List[Dict[str, Any]]):
    # msaTasks
    msaSearchTasks = group(
        signature("blast", args=[requests], queue="queue_blast"),
        signature("jackhmmer", args=[requests], queue="queue_jackhmmer"),
        signature("hhblits", args=[requests], queue="queue_hhblits"),
        # signature("mmseqs", args=[requests], queue="queue_mmseqs"),
    )
    msaMergeTask = signature("mergemsa", args=[requests], queue="queue_mergemsa", immutable=True)
    msaGenTask = chord(msaSearchTasks)(msaMergeTask)()
    msaGenTask.save()
    return {"msaGenTask_id": msaGenTask.id}


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


@app.post("/pipeline")
async def pipeline_task(requests: List[Dict[str, Any]] = Body(..., embed=True)):
    # msaTasks
    msaSearchTasks = group(
        signature("blast", args=[requests], queue="queue_blast"), 
        signature("jackhmmer", args=[requests], queue="queue_jackhmmer"),
        signature("hhblits", args=[requests], queue="queue_hhblits"),
    )
    msaMergeTask = signature("mergemsa", args=[requests], queue="queue_mergemsa", immutable=True)
    msaSelctTask = signature("selectmsa", args=[requests], queue="queue_selectmsa", immutable=True)

    # templateTasks
    templateSearchTask  = signature("searchtpl", args=[requests], queue="queue_searchtpl", immutable=True)
    templateFeatureTask = signature("tplfeature", args=[requests], queue="queue_tplfeature", immutable=True)
    templateSelectTask  = signature("selecttpl", args=[requests], queue="queue_selecttpl", immutable=True)

    # structureTask
    structureTask = signature("monostructure", args=[requests], queue="queue_monostructure", immutable=True)

    # analysisTask
    analysisTask = signature("analysis", args=[requests], queue="queue_analysis", immutable=True)

    # submitTask
    submitTask = signature("submit", args=[requests], queue="queue_submit", immutable=True)


    # pipelineTask
    pipelineTask = (msaSearchTasks | msaMergeTask | msaSelctTask | templateSearchTask | templateFeatureTask | templateSelectTask | 
                    structureTask | analysisTask | submitTask)()

    # pipelineTask.save()
    task_id = pipelineTask.id
    info_reportor = info_report.InfoReport(db_path=DB_PATH)
    info_reportor.update_reserved(
            hash_id=requests[0]["hash_id"], update_dict={"task_id": task_id}
    )
    logger.info(f"------- the task id is {task_id}")

    return {"pipelineTask_id": task_id}


# ----------------------------
# API BACKEND
# ----------------------------


def prefix_ip(message: str, request: Request):
    client_host = request.client.host
    return f"{client_host} - {message}"


def try_json_loads(x):
    if not isinstance(x, str):
        return x
    try:
        return json.loads(x)
    except:
        return x


@app.get("/file/png")
async def get_png(request: Request):
    _params = request.query_params
    file_path = _params["file_path"]
    file_name = os.path.basename(file_path)
    if Path(file_path).exists():
        logger.info(prefix_ip(f"Sending {file_path}", request))
        return FileResponse(path=file_path, filename=file_name)
    else:
        logger.warning(prefix_ip(f"{file_path} does not exist", request))
        return "File not found", 404


@app.get("/file/text")
async def get_file(request: Request):
    _params = request.query_params
    file_path = _params["file_path"]
    file_name = os.path.basename(file_path)
    if Path(file_path).exists():
        logger.info(prefix_ip(f"Sending {file_path}", request))
        return FileResponse(path=file_path, filename=file_name)
    else:
        logger.warning(prefix_ip(f"{file_path} does not exist", request))
        return "File not found", 404


@app.get("/file/download")
async def get_file_download(request: Request):
    _params = request.query_params
    file_path = _params["file_path"]
    file_name = os.path.basename(file_path)
    if Path(file_path).exists():
        logger.info(prefix_ip(f"Sending {file_path}", request))
        return FileResponse(path=file_path, filename=file_name)   # as_attachment=True?
    else:
        logger.warning(prefix_ip(f"{file_path} does not exist", request))
        return "File not found", 404


@app.get("/query/hash_id/{hash_id}")
async def pull_hash_id(hash_id: str, request: Request):
    records = info_retriever.pull_hash_id(hash_id=hash_id)
    records = [r._asdict() for r in records]
    logger.info(prefix_ip(f"query {hash_id}", request))
    results = [{k: try_json_loads(r[k]) for k in r} for r in records]
    return JSONResponse(content=jsonable_encoder(results))

        
@app.get("/query")
async def pull_with_condition(request: Request):
    _params = request.query_params
    _params = {
        k: _params[k]
        for k in _params
        if k in StateRecord._fields
        or "_".join(k.split("_")[:-1]) in StateRecord._fields
        or k == "limit"
    }
    _params = {k: _params[k].replace(".*", "%") for k in _params}

    records = info_retriever.pull_with_condition(_params)
    records = [r._asdict() for r in records]
    logger.info(prefix_ip("sending all records.", request))
    results = [{k: try_json_loads(r[k]) for k in r} for r in records]
    return JSONResponse(content=jsonable_encoder(results))


@app.post("/update/visible/{hash_id}")
async def set_visible(hash_id: str, request: Request):
    _params = request.query_params
    visible = _params.get(VISIBLE, 1)
    info_report.update_visible(hash_id=hash_id, visible=visible)
    results = []
    try:
        ret = info_retriever.pull_hash_id(hash_id=hash_id)[0]
        rcd = ret._asdict()
        results.append({k: try_json_loads(rcd[k]) for k in rcd})
    except sqlite3.IntegrityError as e:
        results.append({HASH_ID: hash_id, ERROR: f"IntegrityError: {str(e)}"})
    except Exception as e:
        results.append({HASH_ID: hash_id, ERROR: f"UnknownError: {str(e)}"})
    return JSONResponse(content=jsonable_encoder(results))

@app.get("/update/visible/{hash_id}")
async def set_visible(hash_id: str, request: Request):
    _params = request.query_params
    visible = _params.get(VISIBLE, 1)
    info_report.update_visible(hash_id=hash_id, visible=visible)
    results = []
    try:
        ret = info_retriever.pull_hash_id(hash_id=hash_id)[0]
        rcd = ret._asdict()
        results.append({k: try_json_loads(rcd[k]) for k in rcd})
    except sqlite3.IntegrityError as e:
        results.append({HASH_ID: hash_id, ERROR: f"IntegrityError: {str(e)}"})
    except Exception as e:
        results.append({HASH_ID: hash_id, ERROR: f"UnknownError: {str(e)}"})
    return JSONResponse(content=jsonable_encoder(results))


@app.options("/update/visible/{hash_id}") 
async def set_visible(hash_id: str, request: Request):
    _params = request.query_params
    visible = _params.get(VISIBLE, 1)
    info_report.update_visible(hash_id=hash_id, visible=visible)
    results = []
    try:
        ret = info_retriever.pull_hash_id(hash_id=hash_id)[0]
        rcd = ret._asdict()
        results.append({k: try_json_loads(rcd[k]) for k in rcd})
    except sqlite3.IntegrityError as e:
        results.append({HASH_ID: hash_id, ERROR: f"IntegrityError: {str(e)}"})
    except Exception as e:
        results.append({HASH_ID: hash_id, ERROR: f"UnknownError: {str(e)}"})
    return JSONResponse(content=jsonable_encoder(results))


@app.post(f"/update/lddt")
async def batch_get_lddt(request: Request):
    _params = request.query_params
    
    results = []
    if _params is None or HASH_ID not in _params:
        return JSONResponse(content=jsonable_encoder(results))
    hash_ids = _params[HASH_ID]
    if not isinstance(hash_ids, list):
        hash_ids = [hash_ids]

    logger.info(prefix_ip(f"update lddt for {hash_ids}", request))
    for hash_id in hash_ids:
        try:
            info_report.update_lddt_metric(hash_id=hash_id)
            ret = info_retriever.pull_hash_id(hash_id=hash_id)[0]
            rcd = ret._asdict()
            results.append({k: try_json_loads(rcd[k]) for k in rcd})
        except sqlite3.IntegrityError as e:
            results.append(
                {
                    HASH_ID: hash_id,
                    ERROR: f"IntegrityError: {str(e)} when update lddt metric",
                }
            )
            logger.exception("update lddt error")
        except Exception as e:
            results.append(
                {
                    HASH_ID: hash_id,
                    ERROR: f"UnknownError: {str(e)} when update lddt metric",
                }
            )
            logger.exception("update lddt error")

    return JSONResponse(content=jsonable_encoder(results))


@app.post(f"/update/rerun")
async def batch_rerun(request: Request):
    _params = request.query_params
    
    results = []
    if _params is None or HASH_ID not in _params:
        return JSONResponse(content=jsonable_encoder(results))
    hash_ids = _params[HASH_ID]
    if not isinstance(hash_ids, list):
        hash_ids = [hash_ids]
    
    logger.info(prefix_ip(f"rerurn {hash_ids}", request))
    for hash_id in hash_ids:
        try:
            info_report.update_state(hash_id=hash_id, state=State.RECEIVED)
            ret = info_retriever.pull_hash_id(hash_id=hash_id)[0]
            r = try_json_loads(ret.request_json)
            if "run_config" not in r:
                r = extend_run_config(r)
            r["submit"] = False
            info_report.update_request(hash_id=hash_id, request=r)
            info_report.update_visible(hash_id=hash_id, visible=1)
            ret = info_retriever.pull_hash_id(hash_id=hash_id)[0]
            rcd = ret._asdict()
            results.append({k: try_json_loads(rcd[k]) for k in rcd})
        except sqlite3.IntegrityError as e:
            results.append({HASH_ID: hash_id, ERROR: f"IntegrityError: {str(e)}"})
        except Exception as e:
            results.append({HASH_ID: hash_id, ERROR: f"UnknownError: {str(e)}"})

    return JSONResponse(content=jsonable_encoder(results))


@app.post(f"/update/submit")
async def batch_submit(request: Request):
    _params = request.query_params
    
    results = []
    if _params is None or HASH_ID not in _params:
        return JSONResponse(content=jsonable_encoder(results))
    hash_ids = _params[HASH_ID]
    if not isinstance(hash_ids, list):
        hash_ids = [hash_ids]

    logger.info(prefix_ip(f"request to email records: {hash_ids}", request))
    _requests = []
    for hash_id in hash_ids:
        try:
            ret = info_retriever.pull_hash_id(hash_id=hash_id)[0]
            rcd = ret._asdict()
            results.append({k: try_json_loads(rcd[k]) for k in rcd})
            r = try_json_loads(ret.request_json)
            r["submit"] = True
            _requests.append(r)
        except sqlite3.IntegrityError as e:
            results.append({HASH_ID: hash_id, ERROR: f"IntegrityError: {str(e)}"})
        except Exception as e:
            results.append({HASH_ID: hash_id, ERROR: f"UnknownError: {str(e)}"})
    logger.info(f"Email requests: \n{json.dumps(_requests, indent=2)}")
    # submit
    # UniforSubmitRunner(requests=_requests, db_path=DB_PATH, loop_forever=False)()
    celery_client.send_task("submit", args=[_requests], queue="queue_submit")
    for hash_id in hash_ids:
        try:
            ret = info_retriever.pull_hash_id(hash_id=hash_id)[0]
            rcd = ret._asdict()
            results.append({k: try_json_loads(rcd[k]) for k in rcd})
        except sqlite3.IntegrityError as e:
            results.append({HASH_ID: hash_id, ERROR: f"IntegrityError: {str(e)}"})
        except Exception as e:
            results.append({HASH_ID: hash_id, ERROR: f"UnknownError: {str(e)}"})
    return JSONResponse(content=jsonable_encoder(results))


@app.post(f"/update/gen_analysis")
async def batch_gen_analysis(request: Request):
    _params = request.query_params
    
    results = []
    if _params is None or HASH_ID not in _params:
        return JSONResponse(content=jsonable_encoder(results))
    hash_ids = _params[HASH_ID]
    if not isinstance(hash_ids, list):
        hash_ids = [hash_ids]

    logger.info(prefix_ip(f"request to email records: {hash_ids}", request))
    _requests = []
    for hash_id in hash_ids:
        try:
            ret = info_retriever.pull_hash_id(hash_id=hash_id)[0]
            rcd = ret._asdict()
            results.append({k: try_json_loads(rcd[k]) for k in rcd})
            r = try_json_loads(ret.request_json)
            r["submit"] = True
            _requests.append(r)
        except sqlite3.IntegrityError as e:
            results.append({HASH_ID: hash_id, ERROR: f"IntegrityError: {str(e)}"})
        except Exception as e:
            results.append({HASH_ID: hash_id, ERROR: f"UnknownError: {str(e)}"})
    
    logger.info(f"Email requests: \n{json.dumps(_requests, indent=2)}")
    for r in _requests:
        # analysis
        # GenAnalysisRunner(requests=[r], db_path=DB_PATH)()
        celery_client.send_task("analysis", args=[[r]], queue="queue_analysis")
    for hash_id in hash_ids:
        try:
            ret = info_retriever.pull_hash_id(hash_id=hash_id)[0]
            rcd = ret._asdict()
            results.append({k: try_json_loads(rcd[k]) for k in rcd})
        except sqlite3.IntegrityError as e:
            results.append({HASH_ID: hash_id, ERROR: f"IntegrityError: {str(e)}"})
        except Exception as e:
            results.append({HASH_ID: hash_id, ERROR: f"UnknownError: {str(e)}"})
    return JSONResponse(content=jsonable_encoder(results))


@app.get("/cameo_data/{to_date}")
async def get_cameo_data(to_date: str, request: Request):
    logger.info(prefix_ip(f"get recent cameo data to {to_date}", request))
    try:
        results = requests.get(cameo_api + to_date).json()
        return JSONResponse(content=jsonable_encoder(results))
    except Exception:
        results = {"aaData": []}
        return JSONResponse(content=jsonable_encoder(results))


@app.get(f"/casp_data")
async def get_casp_targets(request: Request):
    logger.info(prefix_ip(f"get casp targets", request))
    with requests.Session() as s:
        content = s.get(casp_target_list).content.decode("utf-8")
        content = "\n".join(
            [line.replace(";", "\t", 8) for line in content.split("\n")]
        )
    data = pd.read_csv(StringIO(content), sep="\t")
    results = data.to_dict(orient="records")
    return JSONResponse(content=jsonable_encoder(results))


@app.post(f"/insert/request")
async def insert_request(request: Request):
    _received = request.query_params
    logger.info(f"Received request: \n{json.dumps(_received, indent=2)}")
    time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # _received = sorted_dict(_received)
    hash_id = sha256(json.dumps(_received).encode()).hexdigest()
    _received[TIME_STAMP] = time
    _received[HASH_ID] = hash_id

    results = []
    try:
        _request = (
            extend_run_config(_received) if "run_config" not in _received else _received
        )
        info_report.insert_new_request(_request)

        target = _request["target"]
        hash_ids = info_report.get_hash_ids(query_dict={"name": f"{target}"})
        if hash_ids:
            ref_hash = hash_ids[0]
            ref_rcd = info_retriever.pull_hash_id(hash_id=ref_hash)[0]
            ref_reserved = json.loads(ref_rcd.reserved)
            exp_pdb_path = ref_reserved.get("exp_pdb_path", None)
            if exp_pdb_path:
                info_report.update_reserved(
                    hash_id=hash_id, update_dict={"exp_pdb_path": exp_pdb_path}
                )
        ret = info_retriever.pull_hash_id(hash_id=hash_id)[0]
        rcd = ret._asdict()
        results.append({k: try_json_loads(rcd[k]) for k in rcd})
    except sqlite3.IntegrityError as e:
        results.append(
            {
                HASH_ID: hash_id,
                ERROR: f"IntegrityError: {str(e)} when insert new request",
            }
        )
    except Exception as e:
        results.append(
            {
                HASH_ID: hash_id,
                ERROR: f"UnknownError: {str(e)} when insert new request",
            }
        )

    return JSONResponse(content=jsonable_encoder(results))


@app.post(f"/align")
async def align_structures(request: Request):
    _params = request.query_params
    logger.info(f"Received align request: \n{json.dumps(_params, indent=2)}")

    PDBS = "pdbs"
    if _params is None or PDBS not in _params:
        results = []
        return JSONResponse(content=jsonable_encoder(results))
    pdbs = _params[PDBS]
    if not isinstance(pdbs, list):
        pdbs = [pdbs]

    results = align_pdbs(*pdbs)
    for key, val in results.items():
        if hasattr(val, "tolist"):
            results[key] = val.tolist()
    return JSONResponse(content=jsonable_encoder(results))


@app.get("/genconf/{conf_name}")
async def gen_default_conf(conf_name: str, request: Request):
    logger.info(prefix_ip(f"generate default config for {conf_name}", request))
    results = generate_default_config(conf_name=conf_name)
    return JSONResponse(content=jsonable_encoder(results))


@app.get(f"/genconf")
async def gen_conf_default(request: Request):
    logger.info(prefix_ip(f"generate default config", request))
    results = generate_default_config()
    return JSONResponse(content=jsonable_encoder(results))


@app.get("/stop/{hash_id}")
async def stop_process(hash_id: str, request: Request):
    reserved_dict = info_retriever.get_reserved(hash_id=hash_id)
    # logger.info(f"reserved_dict: {reserved_dict}")
    results = []
    if "task_id" not in reserved_dict:
        results.append({HASH_ID: hash_id, ERROR: f"kill task for {hash_id} failed"})
        return JSONResponse(content=jsonable_encoder(results))

    # stop task via revoke
    task_id = reserved_dict["task_id"]
    result = AsyncResult(task_id)
    try:
        result.revoke(terminate=True)
        info_report.update_state(hash_id=hash_id, state=State.KILLED)
        results.append(
            {
                HASH_ID: hash_id,
                ERROR: f"kill process {task_id} for {hash_id} success",
            }
        )
        logger.info(prefix_ip(f"kill task {task_id} for {hash_id} success", request))
    except Exception as e:
        info_report.update_state(hash_id=hash_id, state=State.RUNTIME_ERROR)
        info_report.update_error_message(
            hash_id=hash_id,
            error_msg=f"kill task {task_id} for {hash_id} failed",
        )
        results.append(
            {
                HASH_ID: hash_id,
                ERROR: f"kill process {task_id} for {hash_id} failed because of {e}",
            }
        )
        logger.info(prefix_ip(f"kill task {task_id} for {hash_id} failed", request))
        
    return JSONResponse(content=jsonable_encoder(results))


@app.post("/update/reserved/{hash_id}")
async def update_reserved(hash_id: str, request: Request):
    _params = request.query_params
    
    results = []
    if _params is None:
        return JSONResponse(content=jsonable_encoder(results))
    logger.info(
        prefix_ip(f"update reserved for {hash_id}: \n{json.dumps(_params, indent=2)}", request)
    )
    
    try:
        rcd = info_retriever.pull_hash_id(hash_id=hash_id)[0]
        reserved_dict = json.loads(rcd.reserved) if rcd.reserved else {}
        reserved_dict.update(_params)
        info_report.update_reserved(hash_id=hash_id, update_dict=reserved_dict)
        ret = info_retriever.pull_hash_id(hash_id=hash_id)[0]
        rcd = ret._asdict()
        results.append({k: try_json_loads(rcd[k]) for k in rcd})
    except sqlite3.IntegrityError as e:
        results.append({HASH_ID: hash_id, ERROR: f"IntegrityError: {str(e)}"})
    except Exception as e:
        results.append({HASH_ID: hash_id, ERROR: f"UnknownError: {str(e)}"})
        logger.exception("update reserved failed")
    return JSONResponse(content=jsonable_encoder(results))


@app.post("/update/tags")
async def batch_update_tags(request: Request):
    _params = request.query_params
    
    results = []
    if _params is None or HASH_ID not in _params:
        return JSONResponse(content=jsonable_encoder(results))
    hash_ids = _params[HASH_ID]
    if not isinstance(hash_ids, list):
        hash_ids = [hash_ids]

    def strip_tags(tags):
        return [tag.strip() for tag in tags if tag.strip()]

    logger.info(prefix_ip(f"update tags {hash_ids}", request))
    for hash_id in hash_ids:
        mode = _params.get(TAG_EDIT_MODE, "add")  # add, remove, or replace
        try:
            rcd = info_retriever.pull_hash_id(hash_id=hash_id)[0]
            reserved_dict = json.loads(rcd.reserved) if rcd.reserved else {}
            tags = set(strip_tags(_params.get(TAGS, "").split(TAG_SEP)))
            old_tags = set(
                strip_tags(reserved_dict.get(TAGS).split(TAG_SEP))
                if TAGS in reserved_dict
                else []
            )
            if mode == "add":
                tags = tags | old_tags
            elif mode == "remove":
                tags = tags - old_tags
            elif mode == "replace":
                tags = tags
            reserved_dict[TAGS] = TAG_SEP.join(tags)
            info_report.update_reserved(hash_id=hash_id, update_dict=reserved_dict)
            ret = info_retriever.pull_hash_id(hash_id=hash_id)[0]
            rcd = ret._asdict()
            results.append({k: try_json_loads(rcd[k]) for k in rcd})
        except sqlite3.IntegrityError as e:
            results.append({HASH_ID: hash_id, ERROR: f"IntegrityError: {str(e)}"})
        except Exception as e:
            results.append({HASH_ID: hash_id, ERROR: f"UnknownError: {str(e)}"})

    return JSONResponse(content=jsonable_encoder(results))


@app.get("/update/cameo_gt/{to_date}")
async def cameo_gt_download(to_date: str):
    downloader = download_pdb.CameoPDBDownloader(to_date=to_date)
    downloader.start()
    results = {"status": "in progress"}
    return JSONResponse(content=jsonable_encoder(results))


