import os

from pathlib import Path
from typing import Union

from celery import Celery

import smtplib
from email.header import Header
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


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


@celery.task(name="submit")
def submit(pdb_path, plddt, target_addresses):
    SubmitRunner(pdb_path, plddt, target_addresses).run()


class SubmitRunner:
    def __init__(
        self,
        pdb_path: Union[str, Path] = None,
        plddt: float = 0.0,
        target_addresses: str = ""
        # loop_forever=True,
    ) -> None:
        super().__init__()
        self.smtp_ssl_host = "smtp.office365.com"  # smtp.mail.yahoo.com
        self.smtp_ssl_port = 587
        self.username = "airfold_add_2023@outlook.com"
        self.password = "airfold_add@2023"
        self.sender = "airfold_add_2023@outlook.com"
        # self.loop_forever = loop_forever
        self.pdb_path = pdb_path
        self.plddt = plddt
        self.target_addresses = target_addresses
        # self.target_addresses = "3517109690@qq.com"

    def run(self):
        with smtplib.SMTP(self.smtp_ssl_host, self.smtp_ssl_port) as server:
            server.ehlo()
            server.starttls()
            server.login(self.username, self.password)
            msg = MIMEMultipart()
            # msg["Subject"] = (
            #     _request[TARGET] if TARGET in _request else _request[NAME]
            # )
            subject = "Python SMTP emali test!"
            msg["Subject"] = Header(subject, "utf-8")

            msg["From"] = self.sender
            # msg["To"] = ", ".join(target_addresses)
            msg["To"] = self.target_addresses
            # todo remove debug
            # logger.info(f"Sender is: {self.sender}")
            # logger.info(f"Receivers: {target_addresses}")
            # logger.info(f"Attatches:")
            # for i, (tgt_path, plddt) in enumerate(submit_target_path2plddts):
            # logger.info(f"  [{i}] {tgt_path} with plddt: {plddt}")

            # attach = MIMEText(
            #     open(self.pdb_path, "rb").read(), "base64", "utf-8"
            # )
            text = "Hi!"
            attach = MIMEText(text, "base64", "utf-8")

            attach["Content-Type"] = "application/octet-stream"
            attach[
                "Content-Disposition"
            ] = f'attachment; filename="{os.path.basename(self.pdb_path)}"'
            msg.attach(attach)
            print("Email processinng...")
            server.sendmail(self.sender, self.target_addresses, msg.as_string())
            # return "Email success!"
            # if not dry:
            #     server.sendmail(self.sender, target_addresses, msg.as_string())


# if __name__ == "__main__":
#      SubmitRunner('a.pdb', 0.77, "3517109690@qq.com").run()
