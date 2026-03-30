"""PostgreSQL checkpointer setup for LangGraph."""

import os
from contextlib import contextmanager

import psycopg
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.memory import MemorySaver


def get_checkpointer(db_url: str | None = None):
    """
    Return a LangGraph checkpointer.

    Uses PostgresSaver if a DB URL is available, otherwise falls back to
    MemorySaver (in-process only; state lost on restart — for testing).
    """
    url = db_url or os.getenv("ITERARE_DB_URL")

    if url:
        return _postgres_checkpointer(url)

    print("[iterare] No ITERARE_DB_URL set — using in-memory checkpointer (state will not persist).")
    return MemorySaver()


@contextmanager
def _postgres_checkpointer(url: str):
    """Context manager that yields a PostgresSaver and closes the connection on exit."""
    with psycopg.connect(url, autocommit=True) as conn:
        checkpointer = PostgresSaver(conn)
        checkpointer.setup()  # creates tables if they don't exist
        yield checkpointer
