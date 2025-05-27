import os
import time
import psycopg
from langgraph.checkpoint.postgres import PostgresSaver
from dotenv import load_dotenv
load_dotenv()
def make_connection():
    conn = psycopg.connect(
        os.getenv("POSTGRES_SUPABASE"),
        prepare_threshold=None,
    )
    conn.autocommit = True
    return conn

class ResilientPostgresSaver:
    def __init__(self):
        self._connect()

    def _connect(self):
        self.conn = make_connection()
        self.memory = PostgresSaver(self.conn)

    def _ensure_connected(self):
        try:
            # a cheap check that will hit the server
            with self.conn.cursor() as cur:
                cur.execute("SELECT 1")
        except psycopg.OperationalError:
            # connection has died, re-connect and rebuild the saver
            self._connect()

    def __getattr__(self, attr):
        # intercept all calls and make sure we're alive first
        self._ensure_connected()
        return getattr(self.memory, attr)