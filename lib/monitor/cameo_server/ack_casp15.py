import smtplib, os
from email.message import EmailMessage

smtp_ssl_host = "smtp.office365.com"  # smtp.mail.yahoo.com
smtp_ssl_port = 587
username = "air_psp@outlook.com"
password = "xyvgec-6riDdu-tunfaw"
sender = "air_psp@outlook.com"

casp_ack_email = "casp-meta@predictioncenter.org"
debug_ack_email = "jingjing.gong@qq.com"
MY_SERVER = "http://1.15.181.183:8808/casp"


def casp15_submit_ack(target: str):
    msg = {}
    with smtplib.SMTP(smtp_ssl_host, smtp_ssl_port) as server:
        server.ehlo()
        server.starttls()
        server.login(username, password)
        target_addresses = [casp_ack_email, debug_ack_email]
        msg = EmailMessage()
        msg.set_payload(f"{target} - query received by {MY_SERVER}")
        msg["Subject"] = f"{target} - query received by {MY_SERVER}"
        msg["From"] = sender
        msg["To"] = ", ".join(target_addresses)
        # todo remove debug
        print(f"Sender is: {sender}")
        print(f"Receivers: {target_addresses}")
        server.sendmail(sender, target_addresses, msg.as_string())


if __name__ == "__main__":
    casp15_submit_ack("T1031")
