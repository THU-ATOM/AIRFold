import json
import sys
import pymongo
import requests
from time import sleep, time
from traceback import print_exception
from typing import List
from datetime import datetime, timedelta
from loguru import logger

from lib.constant import *
from lib.state import State
from lib.tool import tool_utils

from lib.monitor.database_mgr import StateRecord
from lib.monitor.info_report import *
from lib.monitor.cameo_server import post_utils
from lib.utils import misc
from lib.monitor.extend_config import extend_run_config


# MAX_CONCURRENT_PIPELINE_NUM = 20
# WAIT_UNTIL_START = 15 * 60
WAIT_UNTIL_START = 30
# REQUEST_PERIOD = 60
REQUEST_PERIOD = 10


def request_due(r, d=3):
    time = datetime.strptime(r["time_stamp"], "%Y%m%d_%H%M%S")
    return datetime.now() - time > timedelta(days=d)


def compose_requests(records: List[StateRecord], info_report: InfoReport) -> List[dict]:
    if len(records) == 0:
        return []
    res = []
    for r in records:
        r_dict = json.loads(r['request_json'])
        try:
            # _r_dict = extend_run_config(r_dict)
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


def pipelineWorker(request_dicts):
    
    with tool_utils.tmpdir_manager(base_dir="/tmp") as tmpdir:
        os.path.join(tmpdir, "requests.pkl")
        # pip_request= {"requests" : request_dicts}
        pipeline_url = f"http://10.0.0.12:8081/pipeline"

        try:
            logger.info(f"------- Requests of pipeline task: {request_dicts}")
            requests.post(pipeline_url , json={'requests': request_dicts})
        except Exception as e:
            logger.error(str(e))


if __name__ == "__main__":
    logger.configure(**MONITOR_LOGGING_CONFIG)
    logger.info("------- Start to monitor...")

    info_report = InfoReport()

    last_received_time = time()
    while True:
        _cur_time = time()
        try:
            _requests = post_utils.pull_visible()  # todo what if this fails
            if len(_requests) > 0:
                logger.info(f"------- The number of requests: {len(_requests)}")
        except requests.exceptions.RequestException:
            sleep(REQUEST_PERIOD)
            continue
        if len(_requests) == 0:
            waited_time = _cur_time - last_received_time
            # logger.info(f"------- The time has waited: {waited_time}")
            if waited_time >= WAIT_UNTIL_START:
                records = info_report.dbmgr.query(
                    {VISIBLE: 1, STATE: State.RECEIVED.name}
                )
                records = list(records)
                logger.info(f"------- Received records: {records}")
                
                if len(records) > 0:
                    for rcds in misc.chunk_generate(records, chunk_size=1):
                        try:
                            # todo
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
                        
                        # For test
                        break
                last_received_time = _cur_time
            sleep(REQUEST_PERIOD)
            continue
        for r in _requests:
            if request_due(r) and "submit" not in r:
                r["submit"] = False
            try:
                if not r[SENDER].startswith("test"):
                    info_report.insert_new_request(extend_run_config(r))
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
            try:
                post_utils.set_invisible(hash_id=r[HASH_ID])
            except requests.exceptions.RequestException:
                sleep(REQUEST_PERIOD)
                continue
            last_received_time = _cur_time

        sleep(REQUEST_PERIOD)
