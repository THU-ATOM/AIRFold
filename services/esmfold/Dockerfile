ARG CUDA=11.3.1
FROM nvidia/cuda:${CUDA}-cudnn8-devel-ubuntu20.04
# FROM nvidia/cuda:${CUDA}-cudnn8-runtime-ubuntu18.04 AS base_af2_env
ARG CUDA

# Use bash to support string substitution.
SHELL ["/bin/bash", "-c"]

RUN rm /etc/apt/sources.list.d/cuda.list
#RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN   apt-key del 7fa2af80
ADD   https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb .
RUN   dpkg -i cuda-keyring_1.0-1_all.deb

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
      build-essential \
      cmake \
      cuda-command-line-tools-$(cut -f1,2 -d- <<< ${CUDA//./-}) \
      git \
      wget \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda & use Python 3.8
ARG python=3.8
ENV PYTHON_VERSION=${python}
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/install-conda.sh \
    && chmod +x /tmp/install-conda.sh \
    && bash /tmp/install-conda.sh -b -f -p /usr/local \
    && rm -f /tmp/install-conda.sh \
    && conda install -y python=${PYTHON_VERSION} \
    && conda clean -y --all

# Install conda packages.
RUN conda update -qy conda \
    && conda install -y -c conda-forge \
      cudatoolkit==${CUDA_VERSION}

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

# RUN pip install -q omegaconf pytorch_lightning biopython ml_collections einops py3Dmol modelcif
# Install PyTorch
# RUN pip install torch==1.10.0+rocm4.2 torchvision==0.11.0+rocm4.2 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html

# Install PyTorch
ENV PYTORCH_VERSION=1.10.0+cu111
ENV TORCHVISION_VERSION=0.11.0+cu111
RUN pip install --no-cache-dir \
    torch==${PYTORCH_VERSION} \
    torchvision==${TORCHVISION_VERSION} \
    -f https://mirror.sjtu.edu.cn/pytorch-wheels/torch_stable.html

# Install ESM
# RUN pip install git+https://github.com/facebookresearch/esm.git
RUN pip install "fair-esm[esmfold]"
# OpenFold and its remaining dependency
# RUN pip install 'dllogger @ git+https://github.com/NVIDIA/dllogger.git'
RUN pip install -q git+https://github.com/NVIDIA/dllogger.git
RUN pip install 'openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307'
# RUN pip install -q git+https://github.com/sokrypton/openfold.git


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
CMD ["celery", "-A", "esmfold", "worker", "--loglevel=info", "-Q", "queue_esmfold"]