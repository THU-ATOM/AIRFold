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

WORKDIR /worker

# Specify the command to run on container start
CMD ["celery", "-A", "mmseqs", "worker", "--loglevel=info", "-Q", "queue_mmseqs"]