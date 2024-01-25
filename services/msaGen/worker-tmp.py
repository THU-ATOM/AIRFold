import os

from celery import Celery
import json
import os
import glob
import shutil
import sys
import time
import traceback
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, OrderedDict, Union

from loguru import logger
from traceback import print_exception


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
    "worker.*": {"queue": "queue_msaGen"},
}


@celery.task(name="msaGen")
def msaGen():
    pass


class BaseRunner:
    def __init__(
        self, requests: List[Dict[str, Any]], db_path: Union[str, Path] = None
    ) -> None:
        self.requests = requests
        self.db_path = db_path
        self._runners = OrderedDict()
        self.run_time = 0.0

        self._info_reportor = None

    # @property
    # def info_reportor(self) -> info_report.InfoReport:
    #     if self._info_reportor is None and self.db_path is not None:
    #         self._info_reportor = info_report.InfoReport(db_path=self.db_path)
    #     return self._info_reportor

    def add_runner(self, name, runner: "BaseRunner"):
        # set child runner's db_path to parent's db_path
        self.set_defaults(runner, "db_path", self.db_path)
        self._runners[name] = runner

    def set_defaults(self, runner, key, value):
        setattr(runner, key, value)
        for r in runner._runners.values():
            self.set_defaults(r, key, value)

    def remove_runner(self, name):
        del self._runners[name]

    def __setattr__(self, name: str, value: Union["BaseRunner", Any]) -> None:
        runners = self.__dict__.get("_runners")
        if isinstance(value, BaseRunner):
            self.add_runner(name, value)
        else:
            if runners is not None and name in runners:
                self.remove_runner(name)
        object.__setattr__(self, name, value)

    def __repr__(self) -> str:
        str_repr = ""
        str_repr += f"{self.__class__.__name__} ({len(self.requests)}, {self.db_path})"
        if len(self._runners) > 0:
            str_repr += "\n["
            for runner in self._runners.values():
                str_repr += f"\n{runner}".replace("\n", "\n  ")
            str_repr += "\n]"
        return str_repr

    def on_run_start(self):
        pass

    def run(self, *args, **kwargs):
        raise NotImplementedError

    def on_run_end(self):
        pass

    # @property
    # def start_stage(self) -> State:
    #     raise NotImplementedError

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        self.on_run_start()
        for r in self.requests:
            if self.info_reportor is not None:
                self.info_reportor.update_state(
                    hash_id=r[info_report.HASH_ID], state=self.start_stage
                )
        t_start = time.time()
        res = None
        try:
            wait_until_memory_available(min_percent=10.0)
            run_wrapper = timeit_logger(self.run)
            res = run_wrapper(*args, **kwds)
        except Exception as e:
            error_message = str(e)
            for r in self.requests:
                if self.info_reportor is not None:
                    self.info_reportor.update_state(
                        hash_id=r[info_report.HASH_ID],
                        state=State.RUNTIME_ERROR,
                    )
                    self.info_reportor.update_error_message(
                        hash_id=r[info_report.HASH_ID], error_msg=error_message
                    )
            traceback.print_exception(*sys.exc_info())
            return False

        self.run_time += time.time() - t_start
        self.on_run_end()
        return res


class SearchRunner(BaseRunner):
    """
    SearchRunner is a runner for search pipeline.
    TODO: here we could make the runner more flexible by make it compatible with parameters.
    """

    def __init__(
        self,
        requests: List[Dict[str, Any]],
        search_requests: List[Dict[str, Any]],
        db_path: Union[str, Path] = None,
        GroupCommandRunners=[],
        SingleCommandRunners=[],
    ) -> None:
        super().__init__(requests, db_path)
        self.search_requests = search_requests
        self.Groupc = GroupCommandRunners
        self.Singlec = SingleCommandRunners
        for runner in self.Groupc:
            if isinstance(runner, JackhmmerRunner):
                self.add_runner(runner=runner, name="jkrunner")
            if isinstance(runner, HHblitsRunner):
                self.add_runner(runner=runner, name="hhrunner")
        for runner in self.Singlec:
            if isinstance(runner, MMseqRunner):
                self.add_runner(runner=runner, name="mmrunner")
            if isinstance(runner, BlastRunner):
                self.add_runner(runner=runner, name="blastrunner")

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
            for G_c in self.Groupc:
                G_c.run(dry=dry)
            for S_c in self.Singlec:
                S_c.run(dry=dry)

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
