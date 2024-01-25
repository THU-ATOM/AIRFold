import os
from celery import Celery, group
from tasks import hhblist, jackhmmer, blast

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
def msaGen(params):
    g = group(hhblist(), jackhmmer(), blast())
    res = g()
    res.get()
