"""Database module for Zepto Data Pipeline.
Defines normalized SQLite schema with primary and foreign key constraints,
and provides helper functions for connection and table creation.
"""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "zepto_books.db"


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Create and return a SQLite connection with foreign key constraints enabled."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path = DB_PATH) -> None:
    """Initialize database tables with normalized schema and PK/FK relationships."""
    conn = get_connection(db_path)
    cursor = conn.cursor()

    # Drop existing tables to allow clean re-runs
    cursor.execute("DROP TABLE IF EXISTS books;")
    cursor.execute("DROP TABLE IF EXISTS categories;")

    # Table 1: categories (Parent table)
    cursor.execute("""
    CREATE TABLE categories (
        category_id INTEGER PRIMARY KEY AUTOINCREMENT,
        category_name TEXT UNIQUE NOT NULL
    );
    """)

    # Table 2: books (Child table with Foreign Key reference to categories)
    cursor.execute("""
    CREATE TABLE books (
        book_id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        price_gbp REAL NOT NULL,
        price_inr REAL NOT NULL,
        rating INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 5),
        in_stock INTEGER NOT NULL CHECK(in_stock IN (0, 1)),
        category_id INTEGER NOT NULL,
        FOREIGN KEY (category_id) REFERENCES categories(category_id) ON DELETE RESTRICT
    );
    """)

    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
    print(f"Initialized normalized SQLite database at {DB_PATH}")
