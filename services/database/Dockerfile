FROM python:3.9-alpine

ENV CELERY_BROKER_URL pyamqp://guest:guest@localhost:5672/
ENV CELERY_RESULT_BACKEND rpc://
ENV C_FORCE_ROOT true

ENV HOST 0.0.0.0
ENV PORT 8003
ENV DEBUG True

COPY . /api
WORKDIR /api

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install -U setuptools pip
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE ${PORT}

# Specify the command to run on container start
CMD uvicorn api:app --host ${HOST} --port ${PORT}