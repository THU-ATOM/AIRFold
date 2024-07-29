from celery import group, signature, chord
from celery.result import AsyncResult, GroupResult
from worker import celery_client

from pathlib import Path
from typing import Any, Dict, List
from loguru import logger
from hashlib import sha256

# import asyncio
import requests
import json, pymongo
from io import StringIO
from pathlib import Path
import pandas as pd

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request, Body
from fastapi.responses import FileResponse
# from fastapi.encoders import jsonable_encoder
from bson import json_util

from lib.monitor.database_mgr import StateRecord
from lib.monitor.extend_config import (
    extend_run_config,
    generate_default_config,
)
from lib.state import State
from lib.utils import misc
from lib.monitor.info_report import *
from lib.constant import *
from lib.tool.align import align_pdbs
import lib.monitor.download_pdb as download_pdb

# configuration
DEBUG = True
app = FastAPI()
# enable CORS
app.add_middleware(
CORSMiddleware,
allow_origins=["*"], # Allows all origins
allow_credentials=True,
allow_methods=["*"], # Allows all methods
allow_headers=["*"], # Allows all headers
)

info_retriever = InfoRetrieve()
info_report = InfoReport()

# ----------------------------
# Single task
# ----------------------------

@app.post("/monitor/")
async def monitor_task():
    task = celery_client.send_task("monitor", args=[], queue="queue_monitor")
    return {"task_id": task.id}

@app.post("/preprocess/")
async def preprocess_task(requests: List[Dict[str, Any]]):
    task = celery_client.send_task("preprocess", args=[requests], queue="queue_preprocess")
    return {"task_id": task.id}

@app.post("/blast/")
async def blast_task(requests: List[Dict[str, Any]]):
    task = celery_client.send_task("blast", args=[requests], queue="queue_blast")
    return {"task_id": task.id}

@app.post("/jackhmmer/")
async def jackhmmer_task(requests: List[Dict[str, Any]]):
    task = celery_client.send_task("jackhmmer", args=[requests], queue="queue_jackhmmer")
    return {"task_id": task.id}

@app.post("/hhblits/")
async def hhblits_task(requests: List[Dict[str, Any]]):
    task = celery_client.send_task("hhblits", args=[requests], queue="queue_hhblits")
    return {"task_id": task.id}


@app.post("/mmseqs/")
async def mmseqs_task(requests: List[Dict[str, Any]]):
    task = celery_client.send_task("mmseqs", args=[requests], queue="queue_mmseqs")
    return {"task_id": task.id}

@app.post("/deepmsa/")
async def deepmsa_task(requests: List[Dict[str, Any]]):
    task = celery_client.send_task("deepmsa", args=[requests], queue="queue_deepmsa")
    return {"task_id": task.id}

@app.post("/mergemsa/")
async def mergemsa_task(requests: List[Dict[str, Any]]):
    task = celery_client.send_task("mergemsa", args=[requests], queue="queue_mergemsa")
    return {"task_id": task.id}

@app.post("/selectmsa/")
async def selectmsa_task(requests: List[Dict[str, Any]]):
    task = celery_client.send_task("selectmsa", args=[requests], queue="queue_selectmsa")
    return {"task_id": task.id}

@app.post("/alphafold/")
async def monostructure_task(requests: List[Dict[str, Any]]):
    task = celery_client.send_task("alphafold", args=[requests], queue="queue_alphafold")
    return {"task_id": task.id}

@app.post("/analysis/")
async def analysis_task(requests: List[Dict[str, Any]]):
    task = celery_client.send_task("analysis", args=[requests], queue="queue_analysis")
    return {"task_id": task.id}

@app.post(f"/submit/")
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

@app.post("/msaGen/")
async def msaGen_task(requests: List[Dict[str, Any]]):
    # msaTasks
    msaSearchTasks = group(
        signature("blast", args=[requests], queue="queue_blast"),
        signature("jackhmmer", args=[requests], queue="queue_jackhmmer"),
        signature("hhblits", args=[requests], queue="queue_hhblits"),
        signature("mmseqs", args=[requests], queue="queue_mmseqs"),
        signature("deepmsa", args=[requests], queue="queue_deepmsa"),
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


@app.post("/mqe/")
async def mqe_task(requests: List[Dict[str, Any]] = Body(..., embed=True)):
    task = celery_client.send_task("mqe", args=[requests], queue="queue_mqe")
    return {"task_id": task.id}


@app.post("/pipeline/")
async def pipeline_task(requests: List[Dict[str, Any]] = Body(..., embed=True)):
    # preprocessTask
    preprocessTask = signature("preprocess", args=[requests], queue="queue_preprocess", immutable=True)
    # msaTasks
    request = requests[0]
    search_args = misc.safe_get(request, ["run_config", "msa_search"])
    if "deepmsa" in search_args.keys() and "mmseqs" in search_args.keys():
        msaSearchTasks = group(
            signature("blast", args=[requests], queue="queue_blast", immutable=True), 
            signature("jackhmmer", args=[requests], queue="queue_jackhmmer", immutable=True),
            signature("hhblits", args=[requests], queue="queue_hhblits", immutable=True),
            signature("deepmsa", args=[requests], queue="queue_deepmsa", immutable=True),
            signature("mmseqs", args=[requests], queue="queue_mmseqs", immutable=True),
        )
    elif "deepmsa" in search_args.keys() and "mmseqs" not in search_args.keys():
        msaSearchTasks = group(
            signature("blast", args=[requests], queue="queue_blast", immutable=True), 
            signature("jackhmmer", args=[requests], queue="queue_jackhmmer", immutable=True),
            signature("hhblits", args=[requests], queue="queue_hhblits", immutable=True),
            signature("deepmsa", args=[requests], queue="queue_deepmsa", immutable=True),
        )
    elif "deepmsa" not in search_args.keys() and "mmseqs" in search_args.keys():
        msaSearchTasks = group(
            signature("blast", args=[requests], queue="queue_blast", immutable=True), 
            signature("jackhmmer", args=[requests], queue="queue_jackhmmer", immutable=True),
            signature("hhblits", args=[requests], queue="queue_hhblits", immutable=True),
            signature("mmseqs", args=[requests], queue="queue_mmseqs", immutable=True),
        )
    else:
        msaSearchTasks = group(
            signature("blast", args=[requests], queue="queue_blast", immutable=True), 
            signature("jackhmmer", args=[requests], queue="queue_jackhmmer", immutable=True),
            signature("hhblits", args=[requests], queue="queue_hhblits", immutable=True),
        )
        
    msaMergeTask = signature("mergemsa", args=[requests], queue="queue_mergemsa", immutable=True)
    msaSelctTask = signature("selectmsa", args=[requests], queue="queue_selectmsa", immutable=True)

    # structureTask
    alphafoldTask = signature("alphafold", args=[requests], queue="queue_alphafold", immutable=True)

    # analysisTask
    analysisTask = signature("analysis", args=[requests], queue="queue_analysis", immutable=True)

    # submitTask
    submitTask = signature("submit", args=[requests], queue="queue_submit", immutable=True)


    # pipelineTask
    pipelineTask = (preprocessTask | msaSearchTasks | msaMergeTask | msaSelctTask | 
                    alphafoldTask | analysisTask | submitTask)()

    # pipelineTask.save()
    task_id = pipelineTask.id
    info_report.update_reserved(
            hash_id=requests[0]["hash_id"], update_dict={"task_id": task_id}
    )
    logger.info(f"------- the task id is {task_id}")

    return {"pipelineTask_id": task_id}


# ----------------------------
# API BACKEND
# ----------------------------

async def get_request(request): 
    try :
        ret = await request.json()
        print(f'-------- request json: {ret}')
        return ret
    except Exception as err:
        # could not parse json
        print(f'------- request body: {await request.body()}')
        return None


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
    # records = [r._asdict() for r in records]
    logger.info(prefix_ip(f"query {hash_id}", request))
    results = [{k: try_json_loads(r[k]) for k in r} for r in records]
    return json.loads(json_util.dumps(results))

        
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
    
    # to do:
    limit_num = 250
    records = info_retriever.pull_with_limit(limit_num, _params)
    # records = [r._asdict() for r in records]
    logger.info(prefix_ip("sending all records.", request))
    results = [{k: try_json_load4query(r, k) for k in r} for r in records]
    return json.loads(json_util.dumps(results))


def try_json_load4query(record, key):
    x = record[key]
    if key == "_id":
        return str(x)
    if not isinstance(x, str):
        return x
    try:
        return json.loads(x)
    except:
        return x
    
@app.post("/update/visible/{hash_id}")
async def set_visible(hash_id: str, request: Request):
    try:
        _params = await get_request(request)
    except Exception as err:
        logger.error(f'could not print REQUEST: {err}')
    
    visible = _params.get(VISIBLE, 1)
    info_report.update_visible(hash_id=hash_id, visible=visible)
    results = []
    try:
        ret = info_retriever.pull_hash_id(hash_id=hash_id)[0]
        # rcd = ret._asdict()
        results.append({k: try_json_loads(ret[k]) for k in ret})
    except pymongo.errors.PyMongoError as e:
        results.append({HASH_ID: hash_id, ERROR: f"IntegrityError: {str(e)}"})
    except Exception as e:
        results.append({HASH_ID: hash_id, ERROR: f"UnknownError: {str(e)}"})
    return json.loads(json_util.dumps(results))

@app.get("/update/visible/{hash_id}")
async def set_visible(hash_id: str, request: Request):
    _params = request.query_params
    
    visible = _params.get(VISIBLE, 1)
    info_report.update_visible(hash_id=hash_id, visible=visible)
    results = []
    try:
        ret = info_retriever.pull_hash_id(hash_id=hash_id)[0]
        # rcd = ret._asdict()
        results.append({k: try_json_loads(ret[k]) for k in ret})
    except pymongo.errors.PyMongoError as e:
        results.append({HASH_ID: hash_id, ERROR: f"IntegrityError: {str(e)}"})
    except Exception as e:
        results.append({HASH_ID: hash_id, ERROR: f"UnknownError: {str(e)}"})
    return json.loads(json_util.dumps(results))


@app.options("/update/visible/{hash_id}") 
async def set_visible(hash_id: str, request: Request):
    try:
        _params = await get_request(request)
    except Exception as err:
        logger.error(f'could not print REQUEST: {err}')
    
    visible = _params.get(VISIBLE, 1)
    info_report.update_visible(hash_id=hash_id, visible=visible)
    results = []
    try:
        ret = info_retriever.pull_hash_id(hash_id=hash_id)[0]
        # rcd = ret._asdict()
        results.append({k: try_json_loads(ret[k]) for k in ret})
    except pymongo.errors.PyMongoError as e:
        results.append({HASH_ID: hash_id, ERROR: f"IntegrityError: {str(e)}"})
    except Exception as e:
        results.append({HASH_ID: hash_id, ERROR: f"UnknownError: {str(e)}"})
    return json.loads(json_util.dumps(results))


@app.post(f"/update/lddt/")
async def batch_get_lddt(request: Request):
    try:
        _params = await get_request(request)
    except Exception as err:
        logger.error(f'could not print REQUEST: {err}')
    
    results = []
    if _params is None or HASH_ID not in _params:
        return json.loads(json_util.dumps(results))
    hash_ids = _params[HASH_ID]
    if not isinstance(hash_ids, list):
        hash_ids = [hash_ids]

    logger.info(prefix_ip(f"update lddt for {hash_ids}", request))
    for hash_id in hash_ids:
        try:
            info_report.update_lddt_metric(hash_id=hash_id)
            ret = info_retriever.pull_hash_id(hash_id=hash_id)[0]
            # rcd = ret._asdict()
            results.append({k: try_json_loads(ret[k]) for k in ret})
        except pymongo.errors.PyMongoError as e:
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

    return json.loads(json_util.dumps(results))


@app.post(f"/update/rerun/")
async def batch_rerun(request: Request):
        
    try:
        _params = await get_request(request)
    except Exception as err:
        logger.error(f'could not print REQUEST: {err}')
    
    results = []
    if _params is None or HASH_ID not in _params:
        return json.loads(json_util.dumps(results))
    hash_ids = _params[HASH_ID]
    if not isinstance(hash_ids, list):
        hash_ids = [hash_ids]
    
    logger.info(prefix_ip(f"rerurn {hash_ids}", request))
    for hash_id in hash_ids:
        logger.info(f"hash_id: {hash_id}")
        try:
            info_report.update_state(hash_id=hash_id, state=State.RECEIVED)
            ret = info_retriever.pull_hash_id(hash_id=hash_id)[0]
            r = try_json_loads(ret['request_json'])
            if "run_config" not in r:
                r = extend_run_config(r)
            r["submit"] = False
            info_report.update_request(hash_id=hash_id, request=r)
            info_report.update_visible(hash_id=hash_id, visible=1)
            ret = info_retriever.pull_hash_id(hash_id=hash_id)[0]
            # rcd = ret._asdict()
            results.append({k: try_json_loads(ret[k]) for k in ret})
        except pymongo.errors.PyMongoError as e:
            results.append({HASH_ID: hash_id, ERROR: f"IntegrityError: {str(e)}"})
        except Exception as e:
            results.append({HASH_ID: hash_id, ERROR: f"UnknownError: {str(e)}"})

    return json.loads(json_util.dumps(results))


@app.post(f"/update/submit/")
async def batch_submit(request: Request):
    try:
        _params = await get_request(request)
    except Exception as err:
        logger.error(f'could not print REQUEST: {err}')
    
    results = []
    if _params is None or HASH_ID not in _params:
        return json.loads(json_util.dumps(results))
    hash_ids = _params[HASH_ID]
    if not isinstance(hash_ids, list):
        hash_ids = [hash_ids]

    logger.info(prefix_ip(f"request to email records: {hash_ids}", request))
    _requests = []
    for hash_id in hash_ids:
        try:
            ret = info_retriever.pull_hash_id(hash_id=hash_id)[0]
            results.append({k: try_json_loads(ret[k]) for k in ret})
            r = try_json_loads(ret['request_json'])
            r["submit"] = True
            _requests.append(r)
        except pymongo.errors.PyMongoError as e:
            results.append({HASH_ID: hash_id, ERROR: f"IntegrityError: {str(e)}"})
        except Exception as e:
            results.append({HASH_ID: hash_id, ERROR: f"UnknownError: {str(e)}"})
    logger.info(f"Email requests: \n{json.dumps(_requests, indent=2)}")
    # submit
    celery_client.send_task("submit", args=[_requests], queue="queue_submit")
    for hash_id in hash_ids:
        try:
            ret = info_retriever.pull_hash_id(hash_id=hash_id)[0]
            # rcd = ret._asdict()
            results.append({k: try_json_loads(ret[k]) for k in ret})
        except pymongo.errors.PyMongoError as e:
            results.append({HASH_ID: hash_id, ERROR: f"IntegrityError: {str(e)}"})
        except Exception as e:
            results.append({HASH_ID: hash_id, ERROR: f"UnknownError: {str(e)}"})
    return json.loads(json_util.dumps(results))


@app.post(f"/update/gen_analysis/")
async def batch_gen_analysis(request: Request):
    try:
        _params = await get_request(request)
    except Exception as err:
        logger.error(f'could not print REQUEST: {err}')
    
    results = []
    if _params is None or HASH_ID not in _params:
        return json.loads(json_util.dumps(results))
    hash_ids = _params[HASH_ID]
    if not isinstance(hash_ids, list):
        hash_ids = [hash_ids]

    logger.info(prefix_ip(f"request to email records: {hash_ids}", request))
    _requests = []
    for hash_id in hash_ids:
        try:
            ret = info_retriever.pull_hash_id(hash_id=hash_id)[0]
            results.append({k: try_json_loads(ret[k]) for k in ret})
            r = try_json_loads(ret['request_json'])
            r["submit"] = True
            _requests.append(r)
        except pymongo.errors.PyMongoError as e:
            results.append({HASH_ID: hash_id, ERROR: f"IntegrityError: {str(e)}"})
        except Exception as e:
            results.append({HASH_ID: hash_id, ERROR: f"UnknownError: {str(e)}"})
    
    logger.info(f"Email requests: \n{json.dumps(_requests, indent=2)}")
    for r in _requests:
        # analysis
        celery_client.send_task("analysis", args=[[r]], queue="queue_analysis")
    for hash_id in hash_ids:
        try:
            ret = info_retriever.pull_hash_id(hash_id=hash_id)[0]
            # rcd = ret._asdict()
            results.append({k: try_json_loads(ret[k]) for k in ret})
        except pymongo.errors.PyMongoError as e:
            results.append({HASH_ID: hash_id, ERROR: f"IntegrityError: {str(e)}"})
        except Exception as e:
            results.append({HASH_ID: hash_id, ERROR: f"UnknownError: {str(e)}"})
    return json.loads(json_util.dumps(results))


@app.get("/cameo_data/{to_date}")
async def get_cameo_data(to_date: str, request: Request):
    cameo_api = "https://www.cameo3d.org/modeling/targets/1-month/ajax/?to_date="
    logger.info(prefix_ip(f"get recent cameo data to {to_date}", request))
    try:
        results = requests.get(cameo_api + to_date).json()
        return json.loads(json_util.dumps(results))
    except Exception:
        results = {"aaData": []}
        return json.loads(json_util.dumps(results))


@app.get(f"/casp_data")
async def get_casp_targets(request: Request):
    casp_target_list = "https://predictioncenter.org/casp15/targetlist.cgi?type=csv"
    logger.info(prefix_ip(f"get casp targets", request))
    with requests.Session() as s:
        content = s.get(casp_target_list).content.decode("utf-8")
        content = "\n".join(
            [line.replace(";", "\t", 8) for line in content.split("\n")]
        )
    data = pd.read_csv(StringIO(content), sep="\t")
    results = data.to_dict(orient="records")
    return json.loads(json_util.dumps(results))


@app.post(f"/insert/request/")
async def insert_request(request: Request):
    try:
        _params = await get_request(request)
    except Exception as err:
        logger.error(f'could not print REQUEST: {err}')

    time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # _received = sorted_dict(_received)
    hash_id = sha256(json.dumps(_params).encode()).hexdigest()
    _params[TIME_STAMP] = time
    _params[HASH_ID] = hash_id

    results = []
    try:
        _request = extend_run_config(_params)
        info_report.insert_new_request(_request)

        target = _request["target"]
        hash_ids = info_report.get_hash_ids(query_dict={"name": f"{target}"})
        if hash_ids:
            ref_hash = hash_ids[0]
            ref_rcd = info_retriever.pull_hash_id(hash_id=ref_hash)[0]
            ref_reserved = json.loads(ref_rcd['reserved'])
            exp_pdb_path = ref_reserved.get("exp_pdb_path", None)
            if exp_pdb_path:
                info_report.update_reserved(
                    hash_id=hash_id, update_dict={"exp_pdb_path": exp_pdb_path}
                )
        ret = info_retriever.pull_hash_id(hash_id=hash_id)[0]
        # rcd = ret._asdict()
        results.append({k: try_json_loads(ret[k]) for k in ret})
    except pymongo.errors.PyMongoError as e:
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

    return json.loads(json_util.dumps(results))


@app.post(f"/align/")
async def align_structures(request: Request):
    try:
        _params = await get_request(request)
    except Exception as err:
        logger.error(f'could not print REQUEST: {err}')

    PDBS = "pdbs"
    if _params is None or PDBS not in _params:
        results = []
        return json.loads(json_util.dumps(results))
    pdbs = _params[PDBS]
    if not isinstance(pdbs, list):
        pdbs = [pdbs]

    results = align_pdbs(*pdbs)
    for key, val in results.items():
        if hasattr(val, "tolist"):
            results[key] = val.tolist()
    return json.loads(json_util.dumps(results))


@app.get("/genconf/{conf_name}")
async def gen_default_conf(conf_name: str, request: Request):
    logger.info(prefix_ip(f"generate default config for {conf_name}", request))
    results = generate_default_config(conf_name=conf_name)
    return json.loads(json_util.dumps(results))


@app.get(f"/genconf")
async def gen_conf_default(request: Request):
    logger.info(prefix_ip(f"generate default config", request))
    results = generate_default_config()
    return json.loads(json_util.dumps(results))


@app.get("/stop/{hash_id}")
async def stop_process(hash_id: str, request: Request):
    reserved_dict = info_retriever.get_reserved(hash_id=hash_id)
    # logger.info(f"reserved_dict: {reserved_dict}")
    results = []
    if "task_id" not in reserved_dict:
        results.append({HASH_ID: hash_id, ERROR: f"kill task for {hash_id} failed"})
        return json.loads(json_util.dumps(results))

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
        
    return json.loads(json_util.dumps(results))


@app.post("/update/reserved/{hash_id}")
async def update_reserved(hash_id: str, request: Request):
    try:
        _params = await get_request(request)
    except Exception as err:
        logger.error(f'could not print REQUEST: {err}')
    
    results = []
    if _params is None:
        return json.loads(json_util.dumps(results))
    logger.info(
        prefix_ip(f"update reserved for {hash_id}: \n{json.dumps(_params, indent=2)}", request)
    )
    
    try:
        rcd = info_retriever.pull_hash_id(hash_id=hash_id)[0]
        reserved_dict = json.loads(rcd['reserved']) if rcd['reserved'] else {}
        reserved_dict.update(_params)
        info_report.update_reserved(hash_id=hash_id, update_dict=reserved_dict)
        ret = info_retriever.pull_hash_id(hash_id=hash_id)[0]
        # rcd = ret._asdict()
        results.append({k: try_json_loads(ret[k]) for k in ret})
    except pymongo.errors.PyMongoError as e:
        results.append({HASH_ID: hash_id, ERROR: f"IntegrityError: {str(e)}"})
    except Exception as e:
        results.append({HASH_ID: hash_id, ERROR: f"UnknownError: {str(e)}"})
        logger.exception("update reserved failed")
    return json.loads(json_util.dumps(results))


@app.post("/update/tags/")
async def batch_update_tags(request: Request):
    try:
        _params = await get_request(request)
    except Exception as err:
        logger.error(f'could not print REQUEST: {err}')
    
    results = []
    if _params is None or HASH_ID not in _params:
        return json.loads(json_util.dumps(results))
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
            reserved_dict = json.loads(rcd['reserved']) if rcd['reserved'] else {}
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
            # rcd = ret._asdict()
            results.append({k: try_json_loads(ret[k]) for k in ret})
        except pymongo.errors.PyMongoError as e:
            results.append({HASH_ID: hash_id, ERROR: f"IntegrityError: {str(e)}"})
        except Exception as e:
            results.append({HASH_ID: hash_id, ERROR: f"UnknownError: {str(e)}"})

    return json.loads(json_util.dumps(results))


@app.get("/update/cameo_gt/{to_date}")
async def cameo_gt_download(to_date: str):
    downloader = download_pdb.CameoPDBDownloader(to_date=to_date)
    downloader.start()
    results = {"status": "in progress"}
    return json.loads(json_util.dumps(results))


