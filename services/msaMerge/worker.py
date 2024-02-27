import os
import glob
import json
import shutil
import sys
from celery import Celery

from pathlib import Path
from typing import Any, Dict, List, Union
from loguru import logger
from traceback import print_exception

from lib.base import BaseRunner
from lib.state import State
from lib.pathtree import get_pathtree
from lib.monitor import info_report

SEQUENCE = "sequence"
TARGET = "target"

CELERY_RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", "rpc://")
CELERY_BROKER_URL = (
    os.environ.get("CELERY_BROKER_URL", "pyamqp://guest:guest@localhost:5672/"),
)

celery = Celery(
    __name__,
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
)

celery.conf.task_routes = {
    "worker.*": {"queue": "queue_msaMerge"},
}

DB_PATH = Path("/data/protein/CAMEO/database/cameo_test.db")

@celery.task(name="msaMerge")
def msaMergeTask(requests: List[Dict[str, Any]]):
    command = MSAMergeRunner(requests=requests, db_path=DB_PATH).run()

    return command



class MSAMergeRunner(BaseRunner):
    """
    SearchRunner is a runner for search pipeline.
    TODO: here we could make the runner more flexible by make it compatible with parameters.
    """

    def __init__(
        self,
        requests: List[Dict[str, Any]],
        search_requests: List[Dict[str, Any]],
        db_path: Union[str, Path] = None,
    ) -> None:
        super().__init__(requests, db_path)
        self.search_requests = search_requests

    @property
    def start_stage(self) -> int:
        return State.SEARCH_START

    @staticmethod
    def merge_segmented_a3m_files(target_path, target_sequence, target="input_0"):
        target_len = len(target_sequence)
        target_path = str(target_path)
        segmented_paths = glob.glob(f"{os.path.splitext(target_path)[0]}@*.a3m")
        logger.info(f"segmented_paths:{segmented_paths}")
        collections = [f">{target}", target_sequence]
        if Path(target_path).exists():
            with open(target_path) as fd:
                for line in fd.readlines()[2:]:
                    collections.append(line.strip())
        for path in segmented_paths:
            _, _, start, _, end = os.path.basename(path).replace(".a3m", "").split("@")
            start, end = int(start), int(end)
            logger.info(f"padding a3m file: {path}, with start={start} end={end}")
            with open(path) as fd:
                buff = fd.read()
            paded = lambda line: "".join(
                ["-" * start, line, "-" * (target_len - end - 1)]
            )
            for line in buff.strip().split("\n")[2:]:
                collections.append(paded(line) if not line.startswith(">") else line)
        with open(target_path, "w") as fd:
            wstring = "\n".join(collections)
            fd.write(wstring)
        return target_path

    @staticmethod
    def merge_a3m_files(
        target_path,
        msa_paths: Dict[str, str],
        target_sequence,
        max_seq_per_file=100000,
        target="input_0",
    ):
        collections = [f">{target}", target_sequence]

        statistics = {}
        for name, path in msa_paths.items():
            with open(path) as fd:
                buff = fd.read()
            lines = buff.strip().split("\n")[2:][:max_seq_per_file]
            lines = [
                f"{line} src={name}" if line.startswith(">") else line for line in lines
            ]
            collections.extend(lines)
            statistics[name] = len(lines) // 2
        logger.info(f"{target} search statistics: {json.dumps(statistics)}")
        with open(target_path, "w") as fd:
            wstring = "\n".join(collections)
            fd.write(wstring)
        return True

    @staticmethod
    def segmented_copy(msa_path, msa_path_to, target_sequence, start):
        """only support one line sequence"""

        def indicing(in_a3m_seq, start, length):
            ret = ""
            count = 0
            for ch in in_a3m_seq:
                if not ch.islower():
                    count += 1
                if count > start and count <= start + length:
                    ret += ch

                if count >= start + length:
                    return ret
            return ret

        length = len(target_sequence)
        with open(msa_path) as fd:
            buff = fd.read()
        lines = buff.strip().split("\n")
        wlines = [
            line if line.startswith(">") else indicing(line, start, length)
            for line in lines
        ]
        lines = []
        comment = None
        for i, l in enumerate(wlines):
            if i % 2 == 0:
                comment = l
            else:
                if l.strip("-"):
                    lines.append(comment)
                    lines.append(l)
                else:
                    continue

        primary_seq = lines[1]
        if primary_seq != target_sequence:
            raise ValueError(
                f"primary sequence mismatch with target sequence: {primary_seq} vs {target_sequence}"
            )
        with open(msa_path_to, "w") as fd:
            wstring = "\n".join(lines)
            fd.write(wstring)
        return True

    def run(self, dry=False):
        ptree = get_pathtree(request=self.requests[0])
        copy_int_msa_from = self.requests[0]["run_config"]["msa_search"].get(
            "copy_int_msa_from", "None"
        )

        ptree.search.integrated_search_a3m.parent.mkdir(exist_ok=True, parents=True)
        ptree.search.jackhammer_uniref90_a3m.parent.mkdir(exist_ok=True, parents=True)
        integrated_search_a3m = str(ptree.search.integrated_search_a3m)
        jackhammer_uniref90_a3m = str(ptree.search.jackhammer_uniref90_a3m)
        template_msa_a3m = jackhammer_uniref90_a3m
        try:
            logger.info(f"copying from {copy_int_msa_from}")
            if copy_int_msa_from == "None" or not copy_int_msa_from:
                raise ValueError("copy_int_msa_from is None")
            start_idx = None
            if len(copy_int_msa_from.split(":start_index_")) > 1:
                copy_int_msa_from, start_idx = copy_int_msa_from.split(":start_index_")
                start_idx = int(start_idx)
                logger.info(
                    f"segment copy from {copy_int_msa_from}, start from {start_idx}"
                )
            if Path(copy_int_msa_from).exists():
                copy_int_msa_from = copy_int_msa_from
                copy_tplt_msa_from = copy_int_msa_from
            else:
                copy_int_msa_from = os.path.join(
                    os.path.dirname(integrated_search_a3m),
                    copy_int_msa_from,
                )
                copy_tplt_msa_from = os.path.join(
                    os.path.dirname(template_msa_a3m),
                    copy_int_msa_from,
                )
            if start_idx != None:
                logger.info(f"segmented copy from {copy_int_msa_from}")
                self.segmented_copy(
                    copy_int_msa_from,
                    msa_path_to=integrated_search_a3m,
                    target_sequence=self.requests[0][SEQUENCE],
                    start=start_idx,
                )
                self.segmented_copy(
                    copy_tplt_msa_from,
                    msa_path_to=template_msa_a3m,
                    target_sequence=self.requests[0][SEQUENCE],
                    start=start_idx,
                )
            else:
                logger.info(f"standard copy from {copy_int_msa_from}")
                shutil.copy(copy_int_msa_from, integrated_search_a3m)
                shutil.copy(copy_int_msa_from, template_msa_a3m)
        except:
            print_exception(*sys.exc_info())
            if copy_int_msa_from != "None":
                logger.info(
                    f"copying from {copy_int_msa_from}  failed, fall back to searching logic"
                )
            # # excute the runner
            # for G_c in self.Groupc:
            #     G_c.run(dry=dry)
            # for S_c in self.Singlec:
            #     S_c.run(dry=dry)

            segment_merge = lambda path: self.merge_segmented_a3m_files(
                target_path=path,
                target_sequence=self.requests[0][SEQUENCE],
                target=self.requests[0][TARGET],
            )
            self.merge_a3m_files(
                target_path=integrated_search_a3m,
                msa_paths={
                    "hh_bfd_uni": segment_merge(ptree.search.hhblist_bfd_uniclust_a3m),
                    "jh_mgn": segment_merge(ptree.search.jackhammer_mgnify_a3m),
                    "jh_uni": segment_merge(ptree.search.jackhammer_uniref90_a3m),
                    "bl": segment_merge(ptree.search.blast_a3m),
                },
                target_sequence=self.requests[0][SEQUENCE],
                target=self.requests[0][TARGET],
            )

        self.outputs_paths = [integrated_search_a3m, template_msa_a3m]

        return integrated_search_a3m, template_msa_a3m
        # except:
        #     print(f"runner failed")

    def on_run_end(self):
        request = self.requests[0]
        if self.info_reportor is not None:
            if all([Path(p).exists() for p in self.outputs_paths]):
                self.info_reportor.update_state(
                    hash_id=request[info_report.HASH_ID],
                    state=State.SEARCH_SUCCESS,
                )
            else:
                self.info_reportor.update_state(
                    hash_id=request[info_report.HASH_ID],
                    state=State.SEARCH_ERROR,
                )
