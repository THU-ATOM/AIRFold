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

# Install MMseqs2
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/install-conda.sh \
    && chmod +x /tmp/install-conda.sh \
    && bash /tmp/install-conda.sh -b -f -p /usr/local \
    && rm -f /tmp/install-conda.sh \
    && conda install -c conda-forge -c bioconda mmseqs2

# add user
RUN if [ $USER_NAME != "root" ] ; \
    then addgroup --gid ${GROUP_ID} ${USER_NAME} \
    && adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID ${USER_NAME} \
    && usermod -aG sudo ${USER_NAME} \
    && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers ; fi
    
# Switch to the created user
USER ${USER_NAME}

WORKDIR /worker

# Specify the command to run on container start
CMD ["celery", "-A", "mmseqs", "worker", "--loglevel=info", "-Q", "queue_mmseqs"]