"""SQLite connection and schema helpers."""

import sqlite3

from app.config import DATABASE_PATH


def connect_db() -> sqlite3.Connection:
    """Open a connection with rows accessible by column name."""
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database() -> None:
    """Create the ingestion tables if they do not already exist."""
    with connect_db() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                file_size_bytes INTEGER NOT NULL,
                page_count INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                page_number INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                UNIQUE(document_id, page_number, chunk_index)
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_document_page
                ON chunks(document_id, page_number);
            """
        )
