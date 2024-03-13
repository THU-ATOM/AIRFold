from lib.utils.database import Database
from collections import namedtuple
from typing import List
import os

RequestRecord = namedtuple(
    "RequestRecord",
    "hash_id, name, sender, request_json, time_stamp, reserved, visible",
)


class RequestRecordDBMGR:
    def __init__(self, db_path):
        self._db_path = db_path
        self.table_name = "request_tbl"
        # state colum names
        self.stcolnames = RequestRecord._make(RequestRecord._fields)
        if not os.path.exists(self._db_path):
            self.create_table()

    def create_table(self):
        """
        :param db_path: path to location where the database file resides
        """

        _state_sql = f""" CREATE TABLE IF NOT EXISTS {self.table_name} (
                                                         {self.stcolnames.hash_id} text PRIMARY KEY,
                                                         {self.stcolnames.name} text NOT NULL,
                                                         {self.stcolnames.sender} text NOT NULL,
                                                         {self.stcolnames.request_json} integer NOT NULL,
                                                         {self.stcolnames.time_stamp} text NOT NULL,
                                                         {self.stcolnames.reserved} text,
                                                         {self.stcolnames.visible} integer NOT NULL
                                                         ); """

        with Database(name=self._db_path) as db:
            db.execute(_state_sql)

    def insert(self, state_item: RequestRecord) -> None:
        _fields = RequestRecord._fields
        insert_values = ", ".join(
            [f"'{state_item.__getattribute__(k)}'" for k in _fields]
        )
        _fields_q = ", ".join(_fields)
        _sql_insert = f""" INSERT INTO {self.table_name} ({_fields_q}) VALUES ({insert_values}); """
        with Database(self._db_path) as db:
            db.execute(_sql_insert)

    def query(self, query_dict: dict) -> List[RequestRecord]:
        _fields = RequestRecord._fields
        _fields_q = ", ".join(_fields)
        if len(query_dict) > 0:
            _query_cond = " AND ".join(
                [f"{k} like '{query_dict[k]}'" for k in query_dict]
            )
            _query_cond = f"WHERE {_query_cond}"
        else:
            _query_cond = ""
        _sql_query = (
            f""" SELECT {_fields_q} FROM {self.table_name} {_query_cond}; """
        )
        with Database(self._db_path) as db:
            resulsts = db.query(_sql_query)
        return [item for item in map(RequestRecord._make, resulsts)]

    def update(self, hash_id: str, update_dict: dict) -> None:
        _fields_q = ", ".join([k for k in update_dict])
        _update_str = ", ".join(
            [f"{k} = '{update_dict[k]}'" for k in update_dict]
        )
        _sql_update = f""" UPDATE {self.table_name} SET {_update_str} WHERE {self.stcolnames.hash_id}='{hash_id}' """
        with Database(self._db_path) as db:
            db.execute(_sql_update)

    def make_invisible(self, name):
        self.update(name, update_dict={self.stcolnames.visible: 0})

    def make_visible(self, name):
        self.update(name, update_dict={self.stcolnames.visible: 1})
