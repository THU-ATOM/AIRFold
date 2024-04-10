from collections import namedtuple
from typing import List
from pymongo import MongoClient

client = MongoClient('mongodb://10.0.0.12:27017/', username='admin', password='admin123')
db = client.cameo

RequestRecord = namedtuple(
    "RequestRecord",
    "hash_id, name, sender, request_json, time_stamp, reserved, visible",
)

class RequestRecordDBMGR:
    def __init__(self):
        self.collection = db.request_2024
        # state colum names
        self.stcolnames = RequestRecord._make(RequestRecord._fields)

    def insert(self, state_item: RequestRecord) -> None:
        _fields = RequestRecord._fields
        insert_data = {}
        for k in _fields:
            insert_value = state_item.__getattribute__(k)
            insert_data[k] = insert_value
        self.collection.insert_one(insert_data)

    def query(self, query_dict: dict) -> List[RequestRecord]:
        resulsts = self.collection.find(query_dict)
        return [item for item in map(RequestRecord._make, resulsts)]

    def update(self, hash_id: str, update_dict: dict) -> None:
        self.collection.update( {'hash_id': hash_id}, {'$set': update_dict})

    def make_invisible(self, name):
        self.update(name, update_dict={self.stcolnames.visible: 0})

    def make_visible(self, name):
        self.update(name, update_dict={self.stcolnames.visible: 1})
