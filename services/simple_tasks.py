from celery import shared_task


@shared_task(name="add")
def add(x, y):
    return x + y
