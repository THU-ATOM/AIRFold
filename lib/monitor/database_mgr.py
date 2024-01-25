import os.path
import json
from collections import namedtuple, OrderedDict
from typing import List

from lib.utils.database import Database
from lib.constant import DB_PATH


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
    def __init__(self, db_path=DB_PATH):
        self._db_path = db_path
        self.state_table_name = "state_tbl"
        # state colum names
        self.stcolnames = StateRecord._make(StateRecord._fields)
        if not os.path.exists(self._db_path):
            self.create_tables()

    def create_tables(self):
        """
        :param db_path: path to location where the database file resides
        """

        _state_sql = f""" CREATE TABLE IF NOT EXISTS {self.state_table_name} (
                                                        {self.stcolnames.hash_id} text PRIMARY KEY,
                                                        {self.stcolnames.name} text NOT NULL,
                                                        {self.stcolnames.sender} text NOT NULL,
                                                        {self.stcolnames.receive_time} text NOT NULL,
                                                        {self.stcolnames.request_json} text NOT NULL,
                                                        {self.stcolnames.state} text NOT NULL,
                                                        {self.stcolnames.state_msg} text NOT NULL,
                                                        {self.stcolnames.event_time_json} text NOT NULL,
                                                        {self.stcolnames.path_tree} text,
                                                        {self.stcolnames.plddt} text,
                                                        {self.stcolnames.lddt} text,
                                                        {self.stcolnames.reserved} text,
                                                        {self.stcolnames.error} text,
                                                        {self.stcolnames.visible} integer NOT NULL
                                                        ); """

        with Database(name=self._db_path) as db:
            db.execute(_state_sql)

    def _insert(self, state_item: StateRecord, db: Database):
        _fields = StateRecord._fields
        insert_values = ", ".join(
            [f"'{state_item.__getattribute__(k)}'" for k in _fields]
        )
        _fields_q = ", ".join(_fields)
        _sql_insert = f""" INSERT INTO {self.state_table_name} ({_fields_q}) VALUES ({insert_values}); """
        db.execute(_sql_insert)

    def _query(self, query_dict: dict, db: Database) -> List[StateRecord]:
        _fields = StateRecord._fields
        _fields_q = ", ".join(_fields)
        like_list = [
            k
            for k in query_dict
            if (not k.endswith("_lt") and not k.endswith("_gt") and not k == "limit")
        ]
        less_than_list = [k for k in query_dict if k.endswith("_lt")]
        greater_than_list = [k for k in query_dict if k.endswith("_gt")]
        if "limit" in query_dict:
            limit = f"limit {query_dict['limit']}"
            query_dict.pop("limit")
        else:
            limit = ""

        if len(query_dict) > 0:
            _query_cond = " AND ".join(
                [f"{k} LIKE '{query_dict[k]}'" for k in like_list]
            )
            _query_cond = [_query_cond] if _query_cond else []
            _less_than_cond = " AND ".join(
                [f"{k.rstrip('_lt')} < '{query_dict[k]}'" for k in less_than_list]
            )
            _less_than_cond = [_less_than_cond] if _less_than_cond else []

            _greater_than_cond = " AND ".join(
                [f"{k.rstrip('_gt')} > '{query_dict[k]}'" for k in greater_than_list]
            )
            _greater_than_cond = [_greater_than_cond] if _greater_than_cond else []
            _query_cond = " AND ".join(
                _query_cond + _less_than_cond + _greater_than_cond
            )
            _query_cond = f"WHERE {_query_cond}"
        else:
            _query_cond = ""

        _sql_query = f""" SELECT {_fields_q} FROM {self.state_table_name} {_query_cond} {limit}; """
        resulsts = db.query(_sql_query)
        return resulsts

    def _update(self, hash_id: str, update_dict: dict, db: Database) -> None:
        _update_str = ", ".join([f"{k} = '{update_dict[k]}'" for k in update_dict])
        _sql_update = f""" UPDATE {self.state_table_name} SET {_update_str} WHERE {self.stcolnames.hash_id}='{hash_id}' """
        db.execute(_sql_update)

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
