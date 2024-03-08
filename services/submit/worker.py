import os
from pathlib import Path
from typing import Any, Dict, List, Union

from celery import Celery
from loguru import logger
from time import sleep
import numpy as np
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.message import EmailMessage

from lib.base import BaseRunner, PathTreeGroup
from lib.state import State
from lib.pathtree import get_pathtree
from lib.monitor import info_report
from lib.tool import pdb_clustering
from lib.tool.colabfold.alphafold.common import protein
from lib.monitor import post_utils
from lib.utils import misc


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
    "worker.*": {"queue": "queue_submit"},
}

SEQUENCE = "sequence"
EMAIL = "email"
NAME = "name"
TARGET = "target"

DB_PATH = Path("/data/protein/CAMEO/database/cameo_test.db")

@celery.task(name="submit")
def submitTask(requests: List[Dict[str, Any]]):
    UniforSubmitRunner(requests=requests, db_path=DB_PATH).run()


class CAMEOSubmitRunner(BaseRunner):
    def __init__(
        self,
        requests: List[Dict[str, Any]],
        db_path: Union[str, Path] = None,
        loop_forever=True,
    ) -> None:
        super().__init__(requests, db_path)
        self.smtp_ssl_host = "smtp.office365.com"  # smtp.mail.yahoo.com
        self.smtp_ssl_port = 587
        # self.username = "air_psp@outlook.com"
        # self.username = "airfold_2023@outlook.com"
        self.username = "airfold_add_2023@outlook.com"
        # self.password = "xyvgec-6riDdu-tunfaw"
        # self.password = "airfold_reset@2023"
        self.password = "airfold_add@2023"
        # self.sender = "air_psp@outlook.com"
        # self.sender = "airfold_2023@outlook.com"
        self.sender = "airfold_add_2023@outlook.com"
        self.loop_forever = loop_forever

    @property
    def start_stage(self) -> int:
        return State.SUBMIT_START

    @staticmethod
    def calculate_plddt_from_peb(pdb_path):
        with open(pdb_path) as fd:
            prot = protein.from_pdb_string(fd.read())
            plddt = np.mean(prot.b_factors[:, 0])
        return plddt

    def run(self, dry=False, *args, **kwargs):
        for _request in self.requests:
            if not _request.get("submit", True):
                if self.info_reportor is not None:
                    self.info_reportor.update_state(
                        hash_id=_request[info_report.HASH_ID],
                        state=State.SUBMIT_SKIP,
                    )
                continue
            with smtplib.SMTP(self.smtp_ssl_host, self.smtp_ssl_port) as server:
                server.ehlo()
                server.starttls()
                server.login(self.username, self.password)

                pred_target_paths = get_pathtree(request=_request).alphafold.submit_pdbs
                if len(pred_target_paths) > 5:
                    (
                        tm_score_matrix,
                        plddts,
                        pdbfiles,
                    ) = pdb_clustering.get_tm_score_matrix_plddt(
                        pdb_paths=pred_target_paths
                    )
                    (submit_target_path2plddts, _,) = pdb_clustering.model_selection(
                        tm_score_matrix=tm_score_matrix,
                        names=pdbfiles,
                        plddts=plddts,
                    )

                else:
                    submit_target_path2plddts = [
                        (pdb, self.calculate_plddt_from_peb(pdb))
                        for pdb in pred_target_paths
                    ]
                submit_target_path2plddts = sorted(
                    submit_target_path2plddts, key=lambda x: x[-1], reverse=True
                )
                if len(submit_target_path2plddts) > 0:
                    target_addresses = [_request[EMAIL]]
                    msg = MIMEMultipart()
                    msg["Subject"] = (
                        _request[TARGET] if TARGET in _request else _request[NAME]
                    )
                    msg["From"] = self.sender
                    msg["To"] = ", ".join(target_addresses)
                    # todo remove debug
                    logger.info(f"Sender is: {self.sender}")
                    logger.info(f"Receivers: {target_addresses}")
                    logger.info(f"Attatches:")
                    for i, (tgt_path, plddt) in enumerate(submit_target_path2plddts):
                        logger.info(f"  [{i}] {tgt_path} with plddt: {plddt}")
                        attach = MIMEText(
                            open(tgt_path, "rb").read(), "base64", "utf-8"
                        )
                        attach["Content-Type"] = "application/octet-stream"
                        attach[
                            "Content-Disposition"
                        ] = f'attachment; filename="{os.path.basename(tgt_path)}"'
                        msg.attach(attach)

                    if not dry:
                        server.sendmail(self.sender, target_addresses, msg.as_string())
                        if self.info_reportor is not None:
                            self.info_reportor.update_state(
                                hash_id=_request[info_report.HASH_ID],
                                state=State.SUBMIT_SUCCESS,
                            )
                else:
                    if not dry:
                        if self.info_reportor is not None:
                            self.info_reportor.update_state(
                                hash_id=_request[info_report.HASH_ID],
                                state=State.SUBMIT_ERROR,
                            )
                        if self.loop_forever:
                            post_utils.set_visible(hash_id=_request[post_utils.HASH_ID])
                    logger.info("No pdb to submit.")


class CASPSubmitRunner(BaseRunner):
    def __init__(
        self,
        requests: List[Dict[str, Any]],
        db_path: Union[str, Path] = None,
        loop_forever=True,
        group_name="Shennong",
        author_code="9458-3041-1177",
    ) -> None:
        super().__init__(requests, db_path)
        self.smtp_ssl_host = "smtp.office365.com"  # smtp.mail.yahoo.com
        self.smtp_ssl_port = 587
        self.username = "air_psp@outlook.com"
        self.password = "xyvgec-6riDdu-tunfaw"
        self.sender = "air_psp@outlook.com"

        self.loop_forever = loop_forever
        self._group_name = group_name
        self._author_code = author_code
        self.cc_list = ["jingjing.gong@qq.com"]

    @property
    def start_stage(self) -> int:
        return State.SUBMIT_START

    @staticmethod
    def calculate_plddt_from_peb(pdb_path):
        with open(pdb_path) as fd:
            prot = protein.from_pdb_string(fd.read())
            plddt = np.mean(prot.b_factors[:, 0])
        return plddt

    def run(self, dry=False, *args, **kwargs):
        for reqs in misc.chunk_generate(self.requests, chunk_size=3):
            for _request in reqs:
                if not _request.get("submit", True):
                    if self.info_reportor is not None:
                        self.info_reportor.update_state(
                            hash_id=_request[info_report.HASH_ID],
                            state=State.SUBMIT_SKIP,
                        )
                    continue
                with smtplib.SMTP(self.smtp_ssl_host, self.smtp_ssl_port) as server:
                    server.ehlo()
                    server.starttls()
                    server.login(self.username, self.password)
                    # server.login(self.username, self.password)
                    pred_target_paths = get_pathtree(
                        request=_request
                    ).alphafold.submit_pdbs
                    pred_target_paths = sorted(pred_target_paths)
                    if len(pred_target_paths) > 5:
                        (
                            tm_score_matrix,
                            plddts,
                            pdbfiles,
                        ) = pdb_clustering.get_tm_score_matrix_plddt(
                            pdb_paths=pred_target_paths
                        )
                        (
                            submit_target_path2plddts,
                            _,
                        ) = pdb_clustering.model_selection(
                            tm_score_matrix=tm_score_matrix,
                            names=pdbfiles,
                            plddts=plddts,
                        )

                    else:
                        submit_target_path2plddts = [
                            (pdb, self.calculate_plddt_from_peb(pdb))
                            for pdb in pred_target_paths
                        ]
                    submit_target_path2plddts = sorted(
                        submit_target_path2plddts,
                        key=lambda x: x[-1],
                        reverse=True,
                    )

                    if len(submit_target_path2plddts) > 0:
                        target_addresses = [_request[EMAIL]] + self.cc_list
                        msg = EmailMessage()
                        msg["Subject"] = f"{_request[TARGET]}\t{self._group_name}"
                        msg["From"] = self.sender
                        msg["To"] = ", ".join(target_addresses)
                        # todo remove debug
                        logger.info(f"Sender is: {self.sender}")
                        logger.info(f"Receivers: {target_addresses}")
                        contents = (
                            f"PFRMAT TS\n"
                            f"TARGET {_request[TARGET]}\n"
                            f"AUTHOR {self._author_code}\n"
                            f"METHOD Description of methods used\n"
                        )

                        for i, (tgt_path, plddt) in enumerate(
                            submit_target_path2plddts
                        ):
                            logger.info(f"  [{i}] {tgt_path} with plddt: {plddt}")
                            coordinates = "".join(
                                filter(
                                    lambda x: x.startswith("ATOM"),
                                    open(tgt_path, "r").readlines(),
                                )
                            )
                            coordinates = (
                                f"MODEL  {i+1}\nPARENT N/A\n{coordinates}TER\nEND\n"
                            )
                            contents = f"{contents}{coordinates}"
                        if not dry:
                            msg.set_payload(contents)
                            server.sendmail(
                                self.sender, target_addresses, msg.as_string()
                            )
                            if self.info_reportor is not None:
                                self.info_reportor.update_state(
                                    hash_id=_request[info_report.HASH_ID],
                                    state=State.SUBMIT_SUCCESS,
                                )
                    else:
                        if not dry:
                            if self.info_reportor is not None:
                                self.info_reportor.update_state(
                                    hash_id=_request[info_report.HASH_ID],
                                    state=State.SUBMIT_ERROR,
                                )
                            if self.loop_forever:
                                post_utils.set_visible(
                                    hash_id=_request[post_utils.HASH_ID]
                                )
                        logger.info("No pdb to submit.")

            sleep(3)


class UniforSubmitRunner(BaseRunner):
    def __init__(
        self,
        requests: List[Dict[str, Any]],
        db_path: Union[str, Path] = None,
        loop_forever=True,
    ) -> None:
        super().__init__(requests, db_path)
        self.loop_forever = loop_forever

        self.groups = self.group_requests(self.requests)
        self.runners = self.init_runners(self.groups)

    @property
    def start_stage(self) -> State:
        return State.SUBMIT_START

    def group_requests(self, requests):
        trees = [get_pathtree(request=request) for request in requests]
        groups = [
            tree_group.requests
            for tree_group in PathTreeGroup.groups_from_trees(trees, "request.sender")
        ]
        return groups

    def init_runners(self, groups):
        runners = []
        for group in groups:
            sender = group[0]["sender"]
            if sender.startswith("cameo"):
                submit = CAMEOSubmitRunner(group, loop_forever=self.loop_forever)
            elif sender.startswith("casp"):
                submit = CASPSubmitRunner(group, loop_forever=self.loop_forever)
            else:
                submit = CAMEOSubmitRunner(group, loop_forever=self.loop_forever)
            runners.append(submit)
            self.add_runner(f"submit_{sender}", submit)
        return runners

    def run(self, dry=False):
        for runner in self.runners:
            runner.run(dry=dry)
        return "submitted!"
