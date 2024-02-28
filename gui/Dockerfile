FROM node:lts

RUN apt-get update \
   && apt-get install -qqy --no-install-recommends \
   libvips-dev \
   rsync \
   sudo \
   && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN yarn global add yrm

WORKDIR /app

CMD ["yarn"]