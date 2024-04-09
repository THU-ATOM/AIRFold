from loguru import logger

from lib.constant import *
from lib.state import State

from lib.monitor.info_report import *


def delete_error_request(info_report: InfoReport):
    info_report.dbmgr.delete(name="2024-04-09_O95800")
    records = info_report.dbmgr.query(
        {VISIBLE: 1, STATE: State.RECEIVED.name}
                )
    logger.info(f"------- Received records: {records}")


if __name__ == "__main__":
    info_report = InfoReport(db_path=DB_PATH)
    delete_error_request(info_report=info_report)
    
    


