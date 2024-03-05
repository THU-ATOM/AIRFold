FROM python:3.9

ENV CELERY_BROKER_URL pyamqp://guest:guest@localhost:5672/
ENV CELERY_RESULT_BACKEND rpc://
ENV C_FORCE_ROOT true

COPY requirements.txt /tmp/requirements.txt

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install -U setuptools pip
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install PyTorch
ENV PYTORCH_VERSION=1.8.1+cu111
ENV TORCHVISION_VERSION=0.9.1+cu111
RUN pip install --no-cache-dir \
    torch==${PYTORCH_VERSION} \
    torchvision==${TORCHVISION_VERSION} \
    -f https://mirror.sjtu.edu.cn/pytorch-wheels/torch_stable.html

WORKDIR /worker

# Specify the command to run on container start
CMD ["celery", "-A", "tplfeature", "worker", "--loglevel=info", "-Q", "queue_tplfeature"]