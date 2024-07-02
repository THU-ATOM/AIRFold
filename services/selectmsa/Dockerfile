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

# Install PyTorch
ENV PYTORCH_VERSION=1.8.1+cu111
ENV TORCHVISION_VERSION=0.9.1+cu111
RUN pip install --no-cache-dir \
    torch==${PYTORCH_VERSION} \
    torchvision==${TORCHVISION_VERSION} \
    -f https://mirror.sjtu.edu.cn/pytorch-wheels/torch_stable.html

# Install ESM
RUN pip install git+https://github.com/facebookresearch/esm.git

# Install cmake
RUN wget -O cmake-3.22.5-linux-x86_64.tar.gz https://github.com/Kitware/CMake/releases/download/v3.22.5/cmake-3.22.5-linux-x86_64.tar.gz \
    && tar zxvf cmake-3.22.5-linux-x86_64.tar.gz \
    && cd cmake-3.22.5-linux-x86_64 \
    && cp -r bin/ /usr/local \
    && cp -r share/ /usr/local \
    && rm -rf /tmp/cmake-3.22.5-linux-x86_64 \
    && rm -rf /tmp/cmake-3.22.5-linux-x86_64.tar.gz

# Install CCMpred
RUN git clone --recursive https://github.com/soedinglab/CCMpred.git /tmp/CCMpred \
&& mkdir /tmp/CCMpred/build \
&& cd /tmp/CCMpred/build \
&& cmake -DCMAKE_INSTALL_PREFIX=/usr/local .. \
&& make \
&& make install\
&& rm -rf /tmp/CCMpred

# Install MMseqs2
#RUN git clone https://github.com/soedinglab/MMseqs2.git /tmp/MMseqs2 \
#    && mkdir /tmp/MMseqs2/build \
#    && cd /tmp/MMseqs2/build \
#    && cmake -DHAVE_AVX2=1 -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=/usr/local .. \
#    && make \
#    && make install\
#    && rm -rf /tmp/MMseqs2
RUN cd /tmp \
    && wget https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz \
    && tar xvfz mmseqs-linux-avx2.tar.gz \
    && cp -r mmseqs/bin/* /usr/local/bin/ \
    && rm -rf /tmp/mmseqs

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
CMD ["celery", "-A", "selectmsa", "worker", "--loglevel=info", "-Q", "queue_selectmsa"]