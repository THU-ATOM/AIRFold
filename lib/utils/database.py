import pymongo
import logging
import time
class Database:
    def __init__(self, name=None):

        self.conn = None
        self.cursor = None

        if name:
            self.open(name)

    def open(self, name):

        try:
            self.conn = pymongo.connect(name, timeout=10)
            self.cursor = self.conn.cursor()

        except pymongo.errors.PyMongoError as e:
            raise ValueError("Error connecting to database!")

    def close(self):

        if self.conn:
            self.conn.commit()
            self.cursor.close()
            self.conn.close()

    def __enter__(self):

        return self

    def __exit__(self, exc_type, exc_value, traceback):

        self.close()

    def execute(self, query, retry_num=10):
        
        try:
            self.cursor.execute(query)
        except pymongo.errors.PyMongoError as e:
            if retry_num > 0:
                logging.log(level=logging.WARNING, msg=f'{str(e)}\n retrying execution of the query ...')
                time.sleep(3.)
                self.execute(query=query, retry_num=retry_num-1)
            else:
                raise e


    def query(self, sql):
        self.cursor.execute(sql)
        rows = self.cursor.fetchall()
        return rows
