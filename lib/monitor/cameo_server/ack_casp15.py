import smtplib
from email.message import EmailMessage

casp_ack_email = "casp-meta@predictioncenter.org"
debug_ack_email = "3517109690@qq.com"
MY_SERVER = "http://airfold.yanyanlan.com/casp"


smtp_ssl_host = "smtp.aliyun.com"
smtp_ssl_port = 465
username = "airfold_2024@aliyun.com"
sender = "airfold_2024@aliyun.com"
password="xnkFdpJyh_3E4Ns"

def casp15_submit_ack(target: str):
    msg = {}
    with smtplib.SMTP_SSL(smtp_ssl_host, smtp_ssl_port) as server:
        server.ehlo()
        # server.starttls()
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
