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

# RUN wget https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz \
#     && tar xvfz mmseqs-linux-avx2.tar.gz \
#     && export PATH=$(pwd)/mmseqs/bin/:$PATH

# RUN mkdir /tmp/mmseqs \
#     && cd /tmp/mmseqs \
#     &&  wget -O mmseqs-linux-avx2.tar.gz https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz \
#     && tar zxvf mmseqs-linux-avx2.tar.gz \
#     && cp mmseqs/bin/* /usr/local/bin/ \
#     && rm -rf /tmp/mmseqs

# Install cmake
RUN wget -O cmake-3.22.5-linux-x86_64.tar.gz https://github.com/Kitware/CMake/releases/download/v3.22.5/cmake-3.22.5-linux-x86_64.tar.gz \
    && tar zxvf cmake-3.22.5-linux-x86_64.tar.gz \
    && cd cmake-3.22.5-linux-x86_64 \
    && cp -r bin/ /usr/local \
    && cp -r share/ /usr/local \
    && rm -rf /tmp/cmake-3.22.5-linux-x86_64 \
    && rm -rf /tmp/cmake-3.22.5-linux-x86_64.tar.gz

# Install vmtouch
RUN git clone https://github.com/hoytech/vmtouch.git /tmp/vmtouch \
    && cd /tmp/vmtouch \
    && make \
    && make install \
    && rm -rf /tmp/vmtouch

RUN vmtouch -l -d -t /data/protein/datasets_2024/database/uniref30_2302_db.idx.0
RUN vmtouch -l -d -t /data/protein/datasets_2024/database/uniref30_2302_db.idx.1
RUN vmtouch -l -d -t /data/protein/datasets_2024/database/uniref30_2302_db.idx.2
RUN vmtouch -l -d -t /data/protein/datasets_2024/database/uniref30_2302_db.idx.3
RUN vmtouch -l -d -t  /data/protein/datasets_2024/database/colabfold_envdb_202108_db.idx.0
RUN vmtouch -l -d -t  /data/protein/datasets_2024/database/colabfold_envdb_202108_db.idx.1
RUN vmtouch -l -d -t  /data/protein/datasets_2024/database/colabfold_envdb_202108_db.idx.2
RUN vmtouch -l -d -t  /data/protein/datasets_2024/database/colabfold_envdb_202108_db.idx.3
RUN vmtouch -l -d -t  /data/protein/datasets_2024/database/colabfold_envdb_202108_db.idx.4
RUN vmtouch -l -d -t  /data/protein/datasets_2024/database/colabfold_envdb_202108_db.idx.5
RUN vmtouch -l -d -t  /data/protein/datasets_2024/database/colabfold_envdb_202108_db.idx.6

# Install mmseqs2
RUN git clone --recursive https://github.com/soedinglab/MMseqs2.git /tmp/mmseqs2 \
    && mkdir /tmp/mmseqs2/build \
    && cd /tmp/mmseqs2/build \
    && cmake -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=/usr/local .. \
    && make \
    && make install \
    && rm -rf /tmp/mmseqs2

# RUN git clone https://github.com/soedinglab/MMseqs2.git \
#     && cd MMseqs2 \
#     && mkdir build \
#     && cd build \
#     && cmake -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=. .. \
#     && make \
#     && make install \
#     && export PATH=$(pwd)/bin/:$PATH

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