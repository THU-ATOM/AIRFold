FROM python:3.9

ENV CELERY_BROKER_URL pyamqp://guest:guest@localhost:5672/
ENV CELERY_RESULT_BACKEND rpc://
ENV C_FORCE_ROOT true

ARG USER_ID=1011
ARG GROUP_ID=1011
ARG USER_NAME=docker

COPY requirements.txt /tmp/requirements.txt

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install -U setuptools pip
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install HMMER
RUN mkdir /tmp/hmmer \
    && cd /tmp/hmmer \
    && wget -O hmmer-3.3.2.tar.gz http://eddylab.org/software/hmmer/hmmer-3.3.2.tar.gz \
    && tar xvzf hmmer-3.3.2.tar.gz \
    && cd hmmer-3.3.2 \
    && ./configure --prefix /usr/local \
    && make \
    && make install \
    && rm -rf /tmp/hmmer
    
WORKDIR /worker

# Specify the command to run on container start
CMD ["celery", "-A", "jackhmmer", "worker", "--loglevel=info", "-Q", "queue_jackhmmer"]