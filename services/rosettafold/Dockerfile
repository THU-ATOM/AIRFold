ARG CUDA=12.1.1
# FROM nvidia/cuda:${CUDA}-cudnn8-runtime-ubuntu18.04 AS base_af2_env
FROM nvidia/cuda:${CUDA}-cudnn8-devel-ubuntu22.04
ARG CUDA

# Use bash to support string substitution.
SHELL ["/bin/bash", "-c"]

# RUN rm /etc/apt/sources.list.d/cuda.list
# #RUN rm /etc/apt/sources.list.d/nvidia-ml.list
# RUN   apt-key del 7fa2af80
# ADD   https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb .
# RUN   dpkg -i cuda-keyring_1.0-1_all.deb

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
      build-essential \
      cmake \
      cuda-command-line-tools-$(cut -f1,2 -d- <<< ${CUDA//./-}) \
      git \
      wget \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda & use Python 3.8
ARG python=3.10
ENV PYTHON_VERSION=${python}
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/install-conda.sh \
    && chmod +x /tmp/install-conda.sh \
    && bash /tmp/install-conda.sh -b -f -p /usr/local \
    && rm -f /tmp/install-conda.sh \
    && conda install -y python=${PYTHON_VERSION} \
    && conda clean -y --all

# Install conda packages.
# RUN conda update -qy conda \
#     && conda install -y -c conda-forge \
#       cudatoolkit==${CUDA_VERSION}

ENV CELERY_BROKER_URL pyamqp://guest:guest@localhost:5672/
ENV CELERY_RESULT_BACKEND rpc://
ENV C_FORCE_ROOT true

ARG USER_ID=1011
ARG GROUP_ID=1011
ARG USER_NAME=docker

# Install PyTorch
RUN pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
RUN pip install torchdata==0.7.1
RUN pip install dgl -f https://data.dgl.ai/wheels/torch-2.2/cu121/repo.html
# ENV PYTORCH_VERSION=2.2.0+cu121
# ENV TORCHVISION_VERSION=0.17.0+cu121
# RUN pip install --no-cache-dir \
#     torch==${PYTORCH_VERSION} \
#     torchvision==${TORCHVISION_VERSION} \
#     -f https://mirror.sjtu.edu.cn/pytorch-wheels/torch_stable.html

COPY requirements.txt /tmp/requirements.txt
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install -U setuptools pip
RUN pip install --no-cache-dir -r /tmp/requirements.txt

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
CMD ["celery", "-A", "rosettafold", "worker", "--loglevel=info", "-Q", "queue_rosettafold"]
