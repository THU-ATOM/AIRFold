import urllib.parse, requests
from functools import partial
from collections import OrderedDict

HASH_ID = 'hash_id'
VISIBLE='visible'
PUSH_PATH = '/cgi-bin/cameo.py'
PULL_PATH = '/query'
PULL_BY_HASH_PATH = f'/query/hash_id'
UPDATE_PATH = '/update'
UPDATE_BY_HASH_PATH = f'/update/hash_id'
VISIBILITY_SCRIPT = 'update_record.py'
URL_ROOT = 'http://1.15.181.183:8808'


def send_requests(key2val: dict, path: str) -> dict:
    url = urllib.parse.urljoin(URL_ROOT, path)
    ret = requests.post(url, data=key2val).json()
    return ret


# push_request will fail if `key2val` are identical to some previous submitted request
push_request = partial(send_requests, path=PUSH_PATH)
pull_request = partial(send_requests, path=PULL_PATH)
update_record = partial(send_requests, path=UPDATE_PATH)


def pull_all():
    return pull_request({})


def pull_visible():
    return pull_request({VISIBLE: 1})


def pull_invisible():
    return pull_request({VISIBLE: 0})


def set_visible(hash_id: str) -> dict:
    return update_record({HASH_ID: hash_id, VISIBLE: 1})


def set_invisible(hash_id: str) -> dict:
    return update_record({HASH_ID: hash_id, VISIBLE: 0})


def dict_keyrename(input_dict, keymap):
    return {keymap[k] if k in keymap else k: input_dict[k] for k in input_dict}


def sorted_dict(d: dict):
    return OrderedDict(sorted(d.items()))