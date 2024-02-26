import os
import sys
import tempfile
import time
import traceback
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Union

from loguru import logger

from lib.monitor import info_report
from lib.state import State
from lib.utils import datatool as dtool
from lib.utils.execute import execute
from lib.utils.logging import timeit_logger
from lib.utils.systool import wait_until_memory_available


class BaseRunner:
    def __init__(
        self, requests: List[Dict[str, Any]], db_path: Union[str, Path] = None
    ) -> None:

        self.requests = requests
        self.db_path = db_path
        self._runners = OrderedDict()
        self.run_time = 0.0

        self._info_reportor = None

    @property
    def info_reportor(self) -> info_report.InfoReport:
        if self._info_reportor is None and self.db_path is not None:
            self._info_reportor = info_report.InfoReport(db_path=self.db_path)
        return self._info_reportor

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

    @property
    def start_stage(self) -> State:
        raise NotImplementedError

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
            wait_until_memory_available(min_percent=10.)
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


class BaseCommandRunner(BaseRunner):
    """Process requests one by one."""

    def _decorate_command(self, command):
        # Add PYTHONPATH to execute scripts
        command = f"export PYTHONPATH={os.getcwd()}:$PYTHONPATH && {command}"
        return command

    def build_command(self, request: Dict[str, Any]) -> str:
        """Build a bash command according to the request."""
        raise NotImplementedError

    def run(self, dry=False):
        for i, request in enumerate(self.requests):
            try:
                command = self._decorate_command(self.build_command(request))
                logger.info(f"[{i}] {command}")
                if not dry:
                    execute(command)
            except:
                pass


class BaseGroupCommandRunner(BaseCommandRunner):
    """Process requests group by group.

    The requirement comes from that, some scripts need to process a batch of
    requests in one run. The reasons are two folds:

    1. History code.
    2. To be more efficient, such as loading a model to process requests.

    Basic design idea:

    1. Group requests, so that requests in each group have same input dir
    and output dir.
    2. Process requests group by group:
    - write group of requests to a temp file,
    - build command according to requests and the temp file.
    3. Remove temp files.

    """

    def __init__(
        self,
        requests: List[Dict[str, Any]],
        db_path: Union[str, Path] = None,
        tmpdir: Union[str, Path] = "/tmp",
    ):
        super().__init__(requests, db_path)
        self.groups = self.group_requests(self.requests)
        self.tmpdir = Path(tmpdir)

    def group_requests(
        self, requests: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """Implement this method to group requests.

        Requests in each group should have the same input and out directory.
        """
        raise NotImplementedError

    def requests2file(
        self, requests: List[Dict[str, Any]], path_requests: Path = None
    ) -> Path:
        """Temporarily save requests into a jsonl file.

        Parameters
        ----------
        requests : List[Dict[str, Any]]
        path_requests : Path, optional

        Returns
        -------
        Path
            Path of saved jsonl file.
        """
        if path_requests is None:
            path_requests = Path(tempfile.mkstemp(dir=self.tmpdir, suffix=".jsonl")[1])
        dtool.write_jsonlines(path_requests, requests)
        return path_requests

    def seq2fasta(self, seq: str, path_fasta: Path = None) -> Path:
        """
        Save a sequence into a fasta file.
        """
        if path_fasta is None:
            path_fasta = Path(tempfile.mkstemp(dir=self.tmpdir, suffix=".fasta")[1])
        dtool.list2fasta(path_fasta, [seq])
        return path_fasta

    def seq2aln(self, seq: str, path_aln: Path = None) -> Path:
        """
        Save a sequence into an aln file.
        """
        if path_aln is None:
            path_aln = Path(tempfile.mkstemp(dir=self.tmpdir, suffix=".aln")[1])
        dtool.list2aln(path_aln, [seq])
        # return path_fasta

    def mk_seqs(self, requests):
        for r_ in requests:
            if r_["sequence"] is not None:
                self.seq2aln(r_["sequence"], self.tmpdir / (r_["name"] + ".aln"))
                self.seq2fasta(r_["sequence"], self.tmpdir / (r_["name"] + ".fasta"))

        return self.tmpdir

    def build_command(self, requests: List[Dict[str, Any]], *args, **kwargs) -> str:
        """Implement this method to return command to be excuted.

        Build a bash command according to the request.
        """
        raise NotImplementedError

    def run(self, *args, dry=False, **kwargs):
        """execute requests group by group

        Parameters
        ----------
        dry : bool, optional
            If true, only ouput the command to be executed, but not run,
            by default False.

        """
        for i, group in enumerate(self.groups):
            try:
                command = self._decorate_command(
                    self.build_command(group, *args, **kwargs)
                )
                logger.info(f"[{i}] {command}")
                if not dry:
                    execute(command)
            except:
                logger.exception(f"command failed: {command}")
