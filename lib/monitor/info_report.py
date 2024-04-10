from ctypes import Union
import json, copy, base64
from datetime import datetime
import os
from typing import Union, Dict, Any, List
from pathlib import Path

import pymongo
import matplotlib.pyplot as plt
import numpy as np
import requests
from loguru import logger

import lib.utils.datatool as dtool
from lib.monitor.database_mgr import DBManager
from lib.state import State, get_state2message
from lib.tool import metrics, plot
from lib.constant import FEISHU_WEBHOOK


SENDER = "sender"
HASH_ID = "hash_id"
NAME = "name"
STATE = "state"
VISIBLE = "visible"
STATE_MESSAGE = "state_msg"
ERROR = "error"
REQUEST_JSON = "request_json"
PATH_TREE = "path_tree"
RECEIVE_TIME = "receive_time"
RESERVED = "reserved"
TIME_STAMP = "time_stamp"
TAGS = "tags"
TAG_SEP = ","
TAG_EDIT_MODE = "mode"


def report_to_feishu_bot(content, msg_type="text"):
    message = {"msg_type": msg_type, "content": {msg_type: content}}
    ret = requests.post(FEISHU_WEBHOOK, json=message).json()
    logger.info(f"report_to_feishu_bot: {ret}")


class InfoReport:
    def __init__(self) -> None:
        self.dbmgr = DBManager()

    def get_state_msg(self, hash_id: str) -> list:
        _res = self.dbmgr.query({HASH_ID: hash_id})
        if len(_res) != 1:
            raise pymongo.errors.PyMongoError(f"no record with hash_id={hash_id}")
        state_msg = json.loads(_res[0].state_msg)
        return state_msg[-100:]

    def get_hash_ids(self, query_dict: dict) -> List[str]:
        _res = self.dbmgr.query(query_dict=query_dict)
        return [item.hash_id for item in _res]

    def get_request(self, hash_id: str) -> dict:
        _res = self.dbmgr.query({HASH_ID: hash_id})
        if len(_res) != 1:
            raise pymongo.errors.PyMongoError(f"no record with hash_id={hash_id}")
        request = json.loads(_res[0].request_json)
        return request

    def update_lddt_metric(self, hash_id: str) -> None:
        exp_pdb_path = None

        _res = self.dbmgr.query(query_dict={HASH_ID: hash_id})
        if len(_res) != 1:
            logger.error(f"no such hash: {hash_id}")
            return None

        record = _res[0]
        if record.reserved:
            reserved = json.loads(record.reserved)
            if "exp_pdb_path" in reserved:
                exp_pdb_path = reserved["exp_pdb_path"]
        else:
            # try read exp_pdb_path from original pdb
            POSTFIX_SPLT = "___"
            original_name = record.name.split(POSTFIX_SPLT)[0]
            logger.info(f"Find pdb path from {original_name}.")
            result = self.dbmgr.query(query_dict={NAME: original_name})
            if len(result) > 0:
                original_record = result[0]
                original_reserved = json.loads(original_record.reserved)
                if "exp_pdb_path" in original_reserved:
                    exp_pdb_path = original_reserved["exp_pdb_path"]
                    self.update_reserved(
                        hash_id=record.hash_id,
                        update_dict={"exp_pdb_path": exp_pdb_path},
                    )

        if exp_pdb_path is not None:
            logger.info(f"Using {exp_pdb_path} to evaluate.")
            path_tree = json.loads(record.path_tree)
            target_dir = path_tree["alphafold"]["root"]
            logger.info(
                f"ground truth: {exp_pdb_path}, computing lddt for pdbs in {target_dir}"
            )

            global_lddts = {}
            local_lddts = {}
            for pred_pdb_path in path_tree["alphafold"]["relaxed_pdbs"]:
                model_name = os.path.basename(pred_pdb_path).replace("_relaxed.pdb", "")
                global_lddt, local_lddt = metrics.get_detailed_lddt(
                    pred_pdb=pred_pdb_path, target_pdb=exp_pdb_path
                )
                global_lddts[model_name] = global_lddt * 100
                local_lddts[model_name] = [v * 100 for v in local_lddt[1]]

            dtool.write_json(
                path_tree["alphafold"]["lddt"],
                {"global": global_lddts, "local": local_lddts},
            )

            self.update_metric(hash_id=hash_id, value=global_lddts, metric="lddt")
            plot.plot_lddts(local_lddts)
            plt.savefig(
                os.path.join(target_dir, f"LDDT.png"),
                bbox_inches="tight",
                dpi=np.maximum(200, 300),
            )
            return global_lddts
        else:
            logger.info("No available ground truth structure for evaluation.")
            return None

    def update_state(self, hash_id: str, state: State) -> None:
        state_msg = self.get_state_msg(hash_id)
        state_msg.append(get_state2message(state))
        update_dict = {STATE: state.name, STATE_MESSAGE: state_msg}
        self.dbmgr.update(hash_id=hash_id, update_dict=update_dict)
        _r = self.get_request(hash_id=hash_id)
        if state.value >= State.ERROR.value or state in {
            State.SUBMIT_SUCCESS,
            State.SUBMIT_SKIP,
            State.RECEIVED,
        }:
            report_to_feishu_bot(
                # f'{hash_id}\t{_r["name"]}\t update state to: {state.name}\n full reuqest is:{json.dumps(_r)}'
                f'{hash_id}\t{_r["name"]}\t update state to: {state.name}'
            )

    def update_reserved(self, hash_id: str, update_dict: dict):
        _res = self.dbmgr.query({HASH_ID: hash_id})
        if len(_res) != 1:
            raise pymongo.errors.PyMongoError(f"no record with hash_id={hash_id}")
        reserved_string = _res[0].reserved
        reserved_dict = json.loads(reserved_string) if len(reserved_string) > 0 else {}
        reserved_dict.update(update_dict)
        reserved_string = json.dumps(reserved_dict)
        self.dbmgr.update(hash_id=hash_id, update_dict={RESERVED: reserved_string})

    def update_request(self, hash_id: str, request: dict):
        self.dbmgr.update(hash_id=hash_id, update_dict={REQUEST_JSON: request})

    def update_metric(self, hash_id, value, metric="plddt"):
        update_dict = {metric: value}
        self.dbmgr.update(hash_id=hash_id, update_dict=update_dict)

    def update_error_message(self, hash_id: str, error_msg: str) -> None:
        update_dict = {
            ERROR: base64.b64encode(bytes(error_msg, "utf-8")).decode("utf-8")
        }
        self.dbmgr.update(hash_id=hash_id, update_dict=update_dict)
        report_to_feishu_bot(f"ERROR: {hash_id}:\n {error_msg}")

    def update_visible(self, hash_id: str, visible: int = 1) -> None:
        update_dict = {VISIBLE: visible}
        self.dbmgr.update(hash_id=hash_id, update_dict=update_dict)

    def update_path_tree(self, hash_id: str, path_tree: Dict[str, Any]) -> None:
        update_dict = {PATH_TREE: path_tree}
        self.dbmgr.update(hash_id=hash_id, update_dict=update_dict)

    def insert_new_request(self, r: dict) -> None:
        receive_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        insert_dict = copy.copy(r)
        insert_dict[REQUEST_JSON] = r
        insert_dict[STATE] = State.RECEIVED.name
        insert_dict[STATE_MESSAGE] = [get_state2message(State.RECEIVED)]
        insert_dict[VISIBLE] = 1
        insert_dict[RECEIVE_TIME] = receive_time
        insert_dict[SENDER] = r[SENDER]
        insert_dict = {
            k: insert_dict[k] for k in insert_dict if k in self.dbmgr.stcolnames._fields
        }
        self.dbmgr.insert(insert_dict=insert_dict)
        report_to_feishu_bot(
            f'{insert_dict["hash_id"]}\t{insert_dict["name"]}\t update state to: {State.RECEIVED.name}\n '
            # f"full reuqest is:{json.dumps(insert_dict)}"
        )


class InfoRetrieve:
    def __init__(self) -> None:
        self.dbmgr = DBManager()

    def pull_all(self) -> dict:
        records = self.dbmgr.query({})
        return records

    def pull_hash_id(self, hash_id) -> dict:
        records = self.dbmgr.query({HASH_ID: hash_id})
        return records

    def pull_with_condition(self, cond_dict) -> dict:
        records = self.dbmgr.query(cond_dict)
        return records

    def get_reserved(self, hash_id: str):
        _res = self.dbmgr.query({HASH_ID: hash_id})
        if len(_res) != 1:
            raise pymongo.errors.PyMongoError(f"no record with hash_id={hash_id}")
        reserved_string = _res[0].reserved
        reserved_dict = json.loads(reserved_string) if len(reserved_string) > 0 else {}
        return reserved_dict
