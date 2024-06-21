import argparse
import json
import sys
import pymongo
import requests
from traceback import print_exception
from typing import List
from loguru import logger
from hashlib import sha256

from lib.constant import *
from lib.state import State
from lib.tool import tool_utils

from lib.monitor.database_mgr import StateRecord
from lib.monitor.info_report import *
from lib.utils import misc
from lib.monitor.extend_config import extend_run_config


def compose_requests(records: List[StateRecord], info_report: InfoReport) -> List[dict]:
    if len(records) == 0:
        return []
    res = []
    for r in records:
        r_dict = json.loads(r['request_json'])
        try:
            info_report.update_state(hash_id=r_dict[HASH_ID], state=State.POST_RECEIVE)
            res.append(r_dict)
        except Exception as e:
            error_message = str(e)
            logger.error(error_message)
            info_report.update_state(hash_id=r_dict[HASH_ID], state=State.RECEIVE_ERROR)
            info_report.update_error_message(
                hash_id=r_dict[HASH_ID], error_msg=error_message
            )
            print_exception(*sys.exc_info())
    return res


def insert_request(r: dict, info_report: InfoReport):
    try:
        if not r[SENDER].startswith("test"):
            time = datetime.now().strftime("%Y%m%d_%H%M%S")
            # _received = sorted_dict(_received)
            hash_id = sha256(json.dumps(r).encode()).hexdigest()
            r[TIME_STAMP] = time
            r[HASH_ID] = hash_id
            new_r = extend_run_config(r)
            info_report.insert_new_request(new_r)
    except pymongo.errors.PyMongoError as e:
        logger.warning(
            f"Error when update with hash_id {r[HASH_ID]} : {str(e)}\n "
            f"retrying to reset existing record",
        )
        info_report.update_state(hash_id=r[HASH_ID], state=State.RECEIVED)
        if "run_config" not in r:
            logger.warning(
                f"run_config not in {json.dumps(r)}, \n"
                f"trying extending the request to "
                f"{json.dumps(extend_run_config(r))}"
            )
            info_report.update_request(
                hash_id=r[HASH_ID], request=extend_run_config(r)
            )
        info_report.update_visible(hash_id=r[HASH_ID], visible=1)


def call_pipeline(info_report: InfoReport):
    records = info_report.dbmgr.query(
        {VISIBLE: 1, STATE: State.RECEIVED.name}
                )
    records = list(records)
    logger.info(f"------- Received records: {records}")
                
    if len(records) > 0:
        for rcds in misc.chunk_generate(records, chunk_size=1):
            try:
                request_dicts = compose_requests(
                    rcds, info_report=info_report
                )
                logger.info(
                    f"start processing {len(request_dicts)} requests"
                    f"\n{json.dumps(request_dicts)}"
                )
                pipelineWorker(request_dicts)
                            
            except:
                print_exception(*sys.exc_info())
            break


def load_fasta(file_path):
    # data_suffix: 2024-04-09
    seq_name = ""
    sequence = ""
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith(">"):
                seq_name = line.strip()[1:]
            else:
                sequence = line.strip()
                
    return seq_name, sequence
                

def pipelineWorker(request_dicts, mode):
    
    with tool_utils.tmpdir_manager(base_dir="/tmp") as tmpdir:
        os.path.join(tmpdir, "requests.pkl")
        # pip_request= {"requests" : request_dicts}
        if mode == "msa":
            pipeline_url = f"http://10.0.0.12:8081/msaGen"
        if mode == "eomega":
            pipeline_url = f"http://10.0.0.12:8081/omegafold"
        if mode == "disgram":
            pipeline_url = f"http://10.0.0.12:8081/analysis"
        if mode == "pipeline":
            pipeline_url = f"http://10.0.0.12:8081/pipeline"

        try:
            logger.info(f"------- Requests of pipeline task: {request_dicts}")
            requests.post(pipeline_url , json={'requests': request_dicts})
        except Exception as e:
            logger.error(str(e))

def main(args):

    info_report = InfoReport()

    with open("./tmp/temp_10000_128_1_plmsim.json", 'r') as jf:
        request_dict = json.load(jf)
    
    seq_name, sequence = load_fasta(args.seq_file)
    request_dict["sequence"] = sequence
    request_dict["name"] = seq_name
    request_dict["target"] = seq_name
    logger.info(f"------- Received request: {request_dict}")
    insert_request(r=request_dict, info_report=info_report)
    call_pipeline(info_report=info_report)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True)
    args = parser.parse_args()
    
    logger.configure(**MONITOR_LOGGING_CONFIG)
    logger.info("------- Start to monitor -------")
    
    main()
    