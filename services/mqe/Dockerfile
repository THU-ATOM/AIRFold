ARG CUDA=11.1.1
FROM nvidia/cuda:${CUDA}-cudnn8-runtime-ubuntu18.04 AS base_af2_env
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
      hmmer \
      kalign \
      tzdata \
      wget \
    && rm -rf /var/lib/apt/lists/*

# Compile HHsuite from source.
RUN git clone --branch v3.3.0 https://github.com/soedinglab/hh-suite.git /tmp/hh-suite \
    && mkdir /tmp/hh-suite/build \
    && pushd /tmp/hh-suite/build \
    && cmake -DCMAKE_INSTALL_PREFIX=/opt/hhsuite .. \
    && make -j 4 && make install \
    && ln -s /opt/hhsuite/bin/* /usr/bin \
    && popd \
    && rm -rf /tmp/hh-suite

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

# Install PyTorch
# RUN pip install torch==1.10.0+rocm4.2 torchvision==0.11.0+rocm4.2 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html

# Install PyTorch
ENV PYTORCH_VERSION=1.10.0+cu111
ENV TORCHVISION_VERSION=0.11.0+cu111
RUN pip install --no-cache-dir \
    torch==${PYTORCH_VERSION} \
    torchvision==${TORCHVISION_VERSION} \
    -f https://mirror.sjtu.edu.cn/pytorch-wheels/torch_stable.html

# Install torch-geometric
RUN pip install torch-geometric==2.2.0
RUN pip install https://data.pyg.org/whl/torch-1.10.0%2Bcu113/torch_cluster-1.6.0-cp38-cp38-linux_x86_64.whl
RUN pip install https://data.pyg.org/whl/torch-1.10.0%2Bcu113/torch_scatter-2.0.9-cp38-cp38-linux_x86_64.whl
RUN pip install https://data.pyg.org/whl/torch-1.10.0%2Bcu113/torch_sparse-0.6.12-cp38-cp38-linux_x86_64.whl

# Install pytorch3d
RUN pip install "git+https://github.com/facebookresearch/pytorch3d.git"

# Install ESM
RUN pip install git+https://github.com/facebookresearch/esm.git

# Install PyRosetta4
RUN pip install https://graylab.jhu.edu/download/PyRosetta4/archive/release/PyRosetta4.Release.python38.linux.wheel/pyrosetta-2024.15+release.d972b59-cp38-cp38-linux_x86_64.whl
    
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
CMD ["celery", "-A", "mqe", "worker", "--loglevel=info", "-Q", "queue_mqe"]
