import os.path
import json
from collections import namedtuple, OrderedDict
from typing import List

from lib.utils.database import Database
from lib.constant import DB_PATH

from pymongo import MongoClient

client = MongoClient('mongodb://localhost:27017/')
db = client.cameo_2024



StateRecord = namedtuple(
    "StateRecord",
    [
        "hash_id",
        "name",
        "sender",
        "receive_time",
        "request_json",
        "state",
        "state_msg",
        "event_time_json",
        "path_tree",
        "plddt",
        "lddt",
        "reserved",
        "error",
        "visible",
    ],
    defaults=[""] * 14,
)


def sorted_dict(d: dict):
    return OrderedDict(sorted(d.items()))


class DBManager:
    def __init__(self):
        self.transactions = db.transactions

    def _insert(self, state_item: StateRecord):
        _fields = StateRecord._fields
        insert_data = {}
        for k in _fields:
            insert_value = state_item.__getattribute__(k)
            insert_data[k] = insert_value
        self.transactions.insert_one(insert_data)

    def _query(self, query_dict: dict):
        resulsts = self.transactions.find(query_dict)
        return resulsts

    def _update(self, hash_id: str, update_dict: dict):
        # transactions.update( {'account_id': 'sns_03821023'}, {'$set': {'purchase_method': 'account'}})
        self.transactions.update( {'hash_id': hash_id}, {'$set': update_dict})

    def _record_in_db(self, hash_id: str, db: Database) -> bool:
        res = self._query(query_dict={self.stcolnames.hash_id: hash_id}, db=db)
        if len(res) > 0:
            return True
        else:
            return False

    def insert(self, insert_dict: dict) -> None:
        def dump(item):
            if (
                isinstance(item, dict)
                or isinstance(item, tuple)
                or isinstance(item, list)
            ):
                return json.dumps(item)
            elif (
                isinstance(item, str)
                or isinstance(item, int)
                or isinstance(item, float)
            ):
                return item
            else:
                raise ValueError(f"{item} with unknown value type: {type(item)}")

        _state_dict = {
            k: insert_dict[k] for k in insert_dict if k in StateRecord._fields
        }
        _extra_fields = {
            k: insert_dict[k] for k in insert_dict if k not in StateRecord._fields
        }
        if _extra_fields:
            _state_dict[self.stcolnames.reserved] = _extra_fields
        state_dict = {k: dump(_state_dict[k]) for k in _state_dict}

        with Database(self._db_path) as db:
            self._insert(StateRecord(**state_dict), db)

    def query(self, query_dict: dict) -> List[StateRecord]:
        with Database(self._db_path) as db:
            resulsts = self._query(query_dict=query_dict, db=db)
        return [item for item in map(StateRecord._make, resulsts)]

    def update(self, hash_id: str, update_dict: dict) -> None:
        def dump(item):
            if (
                isinstance(item, dict)
                or isinstance(item, tuple)
                or isinstance(item, list)
            ):
                return json.dumps(item)
            elif (
                isinstance(item, str)
                or isinstance(item, int)
                or isinstance(item, float)
            ):
                return item
            else:
                raise ValueError(f"{item} with unknown value type: {type(item)}")

        update_dict = {k: dump(update_dict[k]) for k in update_dict}
        with Database(self._db_path) as db:
            self._update(hash_id, update_dict=update_dict, db=db)

    def record_in_db(self, hash_id: str) -> bool:
        with Database(self._db) as db:
            return self._record_in_db(hash_id=hash_id, db=db)
