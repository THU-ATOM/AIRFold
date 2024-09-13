FROM python:3.9

ENV CELERY_BROKER_URL pyamqp://guest:guest@localhost:5672/
ENV CELERY_RESULT_BACKEND rpc://
ENV C_FORCE_ROOT true

ENV HOST 0.0.0.0
ENV PORT 8000
ENV DEBUG True

ARG USER_ID=1011
ARG GROUP_ID=1011
ARG USER_NAME=docker

COPY requirements.txt /tmp/requirements.txt

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install -U setuptools pip
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install TM-align
RUN mkdir /tmp/align \
    && cd /tmp/align \
    && wget https://zhanggroup.org/TM-align/TMalign.cpp \
    && g++ -static -O3 -ffast-math -lm -o TMalign TMalign.cpp \
    && cp TMalign /usr/local/bin/ \
    && wget https://zhanggroup.org/US-align/bin/module/USalign.cpp \
    && g++ -static -O3 -ffast-math -o USalign USalign.cpp \
    && cp USalign /usr/local/bin/ \
    && rm -rf /tmp/align

# Install kalign 3
# RUN git clone https://github.com/TimoLassmann/kalign.git /tmp/kalign \
# && cd /tmp/kalign \
# && mkdir build \
# && cd build \
# && cmake .. \
# && make \
# && make test \
# && make install \
# && rm -rf /tmp/kalign

# add user
RUN if [ $USER_NAME != "root" ] ; \
    then addgroup --gid ${GROUP_ID} ${USER_NAME} \
    && adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID ${USER_NAME} \
    && usermod -aG sudo ${USER_NAME} \
    && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers ; fi
    
# Switch to the created user
USER ${USER_NAME}

WORKDIR /app

EXPOSE ${PORT}

# Specify the command to run on container start
CMD uvicorn api:app --host ${HOST} --port ${PORT}
