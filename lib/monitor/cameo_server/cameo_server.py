import sqlite3
import json

from flask import Flask, request, render_template
from datetime import datetime
from hashlib import sha256
from flask_cors import CORS

from DBManager import RequestRecordDBMGR, RequestRecord
from post_utils import *
from ack_casp15 import casp15_submit_ack


app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
CORS(app, resources={r"/*": {"origins": "*"}})

TIME_STAMP = "time_stamp"
TITLE = "title"
SENDER = "sender"
TARGET = "target"
NAME = "name"
ERROR = "error"
SEQUENCE = "sequence"
RAW_SEQUENCE = "raw_sequence"

request_db_path = "request.db"
keymap = {TITLE: NAME}
dbmgr = RequestRecordDBMGR(request_db_path)


@app.route(f"{PUSH_PATH}", methods=["POST", "GET", "OPTIONS"])
def _cameo_server():
    time = datetime.now().strftime("%Y%m%d_%H%M%S")
    _received = request.values.to_dict()
    _received = sorted_dict(_received)
    hash_id = sha256(json.dumps(_received).encode()).hexdigest()
    _received[TIME_STAMP] = time
    _received[HASH_ID] = hash_id
    received = dict_keyrename(_received, keymap)
    received[TARGET] = received[NAME]
    if SENDER not in received:
        raise ValueError(f"{SENDER} not in the request")
    request_json = json.dumps(received)
    record = RequestRecord(
        hash_id=hash_id,
        name=received[NAME],
        sender=received[SENDER],
        request_json=request_json,
        time_stamp=time,
        reserved="",
        visible=1,
    )
    try:
        dbmgr.insert(record)
    except sqlite3.IntegrityError as e:
        return json.dumps(
            {
                ERROR: f"Error when insert record under the primary key {record.hash_id} : {str(e)}"
            }
        )

    return json.dumps(received)


@app.route(f"/casp", methods=["POST", "GET", "OPTIONS"])
def _casp_server():
    time = datetime.now().strftime("%Y%m%d_%H%M%S")
    today = datetime.now().strftime("%Y-%m-%d")
    _received = request.values.to_dict()

    _received[TARGET] = _received[NAME]
    _received[NAME] = f"{today}_{_received[NAME]}"
    _received[RAW_SEQUENCE] = _received[SEQUENCE]
    _received[SEQUENCE] = _received[SEQUENCE].split("|")[-1].strip()
    try:
        casp15_submit_ack(target=_received[TARGET])
    except:
        print("email ack failed")

    hash_id = sha256(json.dumps(_received).encode()).hexdigest()
    _received[TIME_STAMP] = time
    _received[HASH_ID] = hash_id
    received = dict_keyrename(_received, keymap)
    if SENDER not in received:
        raise ValueError(f"{SENDER} not in the request")
    request_json = json.dumps(received)
    record = RequestRecord(
        hash_id=hash_id,
        name=received[NAME],
        sender=received[SENDER],
        request_json=request_json,
        time_stamp=time,
        reserved="",
        visible=1,
    )
    try:
        dbmgr.insert(record)
    except sqlite3.IntegrityError as e:
        return json.dumps(
            {
                ERROR: f"Error when insert record under the primary key {record.hash_id} : {str(e)}"
            }
        )

    return json.dumps(received)


@app.route(f"{PULL_PATH}", methods=["POST", "GET", "OPTIONS"])
def _query_record():
    _params = request.values.to_dict()
    _params = {k: _params[k] for k in _params if k in RequestRecord._fields}
    _params = {k: _params[k].replace(".*", "%") for k in _params}

    try:
        req_res = dbmgr.query(_params)
    except sqlite3.IntegrityError as e:
        return json.dumps(
            {ERROR: f"Error when query with params {json.dumps(_params)} : {str(e)}"}
        )

    results = []
    for item in req_res:
        request_dict = json.loads(item.request_json)
        results.append(request_dict)

    return json.dumps(results)


@app.route(
    f"{PULL_BY_HASH_PATH}/<string:{HASH_ID}>",
    methods=["POST", "GET", "OPTIONS"],
)
def _query_by_hash(hash_id: str):
    _params = request.values.to_dict()
    _params = {k: _params[k] for k in _params if k in RequestRecord._fields}
    _params[HASH_ID] = hash_id
    try:
        req_res = dbmgr.query(_params)
    except sqlite3.IntegrityError as e:
        return json.dumps(
            {ERROR: f"Error when query with params {json.dumps(_params)} : {str(e)}"}
        )

    results = []
    for item in req_res:
        request_dict = json.loads(item.request_json)
        results.append(request_dict)

    return json.dumps(results)


@app.route(f"{UPDATE_PATH}", methods=["POST", "GET", "OPTIONS"])
def _update_record():
    _params = request.values.to_dict()
    _params = {k: _params[k] for k in _params if k in RequestRecord._fields}

    try:
        dbmgr.update(
            hash_id=_params[HASH_ID],
            update_dict={k: _params[k] for k in _params if k != HASH_ID},
        )
        req_res = dbmgr.query(query_dict={HASH_ID: _params[HASH_ID]})
    except sqlite3.IntegrityError as e:
        return json.dumps(
            {ERROR: f"Error when update with params {json.dumps(_params)} : {str(e)}"}
        )

    results = []
    for item in req_res:
        request_dict = json.loads(item.request_json)
        request_dict["visible"] = item.visible
        results.append(request_dict)

    return json.dumps(results)


@app.route(
    f"{UPDATE_BY_HASH_PATH}/<string:{HASH_ID}>",
    methods=["POST", "GET", "OPTIONS"],
)
def _update_by_hash(hash_id: str):
    _params = request.values.to_dict()
    _params = {k: _params[k] for k in _params if k in RequestRecord._fields}
    _params[HASH_ID] = hash_id
    try:
        dbmgr.update(
            hash_id=hash_id,
            update_dict={k: _params[k] for k in _params if k != HASH_ID},
        )
        req_res = dbmgr.query(query_dict={HASH_ID: hash_id})
    except sqlite3.IntegrityError as e:
        return json.dumps(
            {ERROR: f"Error when update with params {json.dumps(_params)} : {str(e)}"}
        )

    results = []
    for item in req_res:
        request_dict = json.loads(item.request_json)
        request_dict["visible"] = item.visible
        results.append(request_dict)

    return json.dumps(results)


@app.route(f"/html/<string:html_file>", methods=["POST", "GET", "OPTIONS"])
def render_html(html_file: str):
    _params = request.values.to_dict()
    _params = {k: _params[k] for k in _params if k in RequestRecord._fields}
    _str_params = "&".join([f"{k}={v}" for k, v in _params.items()])
    str_params = f"?{_str_params}" if len(_str_params) > 0 else ""
    return render_template(html_file, str_params=str_params)
