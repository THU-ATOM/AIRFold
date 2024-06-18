# AIRFold

**Features:**

- launch all with one `docker-compose up`
- services run in isolated docker container
- submit tasks with RESTful API (FastAPI)
- separated task queues
- concurrence control
- tasks monitor

## Quick Start

**Launch the demo:**

``` sh
docker-compose up
```

**Check the page:**

- FastAPI page: http://127.0.0.1:8081/docs
- tasks monitor page (powered by [flower](https://github.com/mher/flower)): http://127.0.0.1:5555

*Note: please change IP address and ports accordingly, they are specified in `docker-compose.yml`*
