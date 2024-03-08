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

# install cmake
RUN wget -O cmake-3.22.5-linux-x86_64.tar.gz https://github.com/Kitware/CMake/releases/download/v3.22.5/cmake-3.22.5-linux-x86_64.tar.gz \
    && tar zxvf cmake-3.22.5-linux-x86_64.tar.gz \
    && cd cmake-3.22.5-linux-x86_64 \
    && cp -r bin/ /usr/local \
    && cp -r share/ /usr/local \
    && rm -rf /tmp/cmake-3.22.5-linux-x86_64 \
    && rm -rf /tmp/cmake-3.22.5-linux-x86_64.tar.gz

# Install HH-suite3
RUN git clone --recursive https://github.com/soedinglab/hh-suite.git /tmp/hhsuite \
    && mkdir /tmp/hhsuite/build \
    && cd /tmp/hhsuite/build \
    && cmake -DCMAKE_INSTALL_PREFIX=/usr/local .. \
    && make \
    && make install\
    && rm -rf /tmp/hhsuite

WORKDIR /worker

# Specify the command to run on container start
CMD ["celery", "-A", "hhblits", "worker", "--loglevel=info", "-Q", "queue_hhblits"]