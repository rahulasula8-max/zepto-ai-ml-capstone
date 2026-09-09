"""Automated Test Suite for Module 1 — Data Pipeline.
Validates all graded criteria for data ingestion, normalization, SQL, and Pandas integration.
"""
import sqlite3
from pathlib import Path
import pandas as pd
import pytest

from database import DB_PATH, get_connection
from scrape_pipeline import CLEAN_CSV_PATH, RAW_CSV_PATH, GBP_TO_INR


def test_files_exist():
    """Verify raw, cleaned CSVs and SQLite database exist."""
    assert RAW_CSV_PATH.exists(), f"Missing {RAW_CSV_PATH}"
    assert CLEAN_CSV_PATH.exists(), f"Missing {CLEAN_CSV_PATH}"
    assert DB_PATH.exists(), f"Missing {DB_PATH}"


def test_row_count_and_categories():
    """Verify >= 60 books and >= 3 categories scraped and stored."""
    df = pd.read_csv(CLEAN_CSV_PATH)
    assert len(df) >= 60, f"Expected >= 60 books, got {len(df)}"
    num_cats = df["category"].nunique()
    assert num_cats >= 3, f"Expected >= 3 categories, got {num_cats}"


def test_data_types_and_inr_conversion():
    """Verify correct data types and exact 1 GBP = 105.50 INR conversion."""
    df = pd.read_csv(CLEAN_CSV_PATH)

    assert pd.api.types.is_float_dtype(df["price_gbp"]), "price_gbp must be float"
    assert pd.api.types.is_float_dtype(df["price_inr"]), "price_inr must be float"
    assert pd.api.types.is_integer_dtype(df["rating"]), "rating must be integer"
    assert set(df["rating"].unique()).issubset({1, 2, 3, 4, 5}), "rating must be in 1..5"
    assert set(df["in_stock"].unique()).issubset({0, 1}), "in_stock must be 0 or 1"

    # Verify 1 GBP = 105.50 INR exactly
    expected_inr = (df["price_gbp"] * 105.50).round(2)
    diff = (df["price_inr"] - expected_inr).abs().max()
    assert diff < 0.01, f"Mismatch in GBP->INR calculation: max diff={diff}"


def test_sqlite_normalization_and_fk():
    """Verify normalized schema with 2 related tables and PK/FK enforcement."""
    conn = get_connection(DB_PATH)
    cur = conn.cursor()

    # Check categories table exists and has PK
    cur.execute("PRAGMA table_info(categories);")
    cat_cols = {row["name"]: row for row in cur.fetchall()}
    assert "category_id" in cat_cols and cat_cols["category_id"]["pk"] == 1
    assert "category_name" in cat_cols

    # Check books table exists and has PK + FK
    cur.execute("PRAGMA table_info(books);")
    book_cols = {row["name"]: row for row in cur.fetchall()}
    assert "book_id" in book_cols and book_cols["book_id"]["pk"] == 1
    assert "category_id" in book_cols

    cur.execute("PRAGMA foreign_key_list(books);")
    fks = cur.fetchall()
    assert len(fks) >= 1, "Expected foreign key constraint on books table"
    fk = fks[0]
    assert fk["table"] == "categories", f"Expected FK referencing categories, got {fk['table']}"
    assert fk["from"] == "category_id"
    assert fk["to"] == "category_id"

    # Test FK integrity violation is properly rejected
    try:
        cur.execute("INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id) VALUES ('Invalid', 10.0, 1055.0, 3, 1, 99999);")
        conn.commit()
        fk_rejected = False
    except sqlite3.IntegrityError:
        fk_rejected = True
    assert fk_rejected, "Foreign key constraint failed to reject non-existent category_id"

    conn.close()


def test_sql_join_matches_pandas_merge():
    """Verify SQL JOIN output and pd.merge output are completely identical."""
    conn = get_connection(DB_PATH)
    sql_join = """
    SELECT 
        b.book_id,
        b.title,
        c.category_name,
        b.price_gbp,
        b.price_inr,
        b.rating,
        b.in_stock
    FROM books b
    INNER JOIN categories c ON b.category_id = c.category_id
    ORDER BY b.book_id ASC;
    """
    df_sql = pd.read_sql(sql_join, conn)

    df_b = pd.read_sql("SELECT book_id, title, price_gbp, price_inr, rating, in_stock, category_id FROM books;", conn)
    df_c = pd.read_sql("SELECT category_id, category_name FROM categories;", conn)
    conn.close()

    df_pandas = pd.merge(df_b, df_c, on="category_id", how="inner")
    cols = ["book_id", "title", "category_name", "price_gbp", "price_inr", "rating", "in_stock"]
    df_pandas = df_pandas[cols].sort_values("book_id").reset_index(drop=True)
    df_sql = df_sql[cols].sort_values("book_id").reset_index(drop=True)

    pd.testing.assert_frame_equal(df_sql, df_pandas)


if __name__ == "__main__":
    print("Running test_files_exist...")
    test_files_exist()
    print("Running test_row_count_and_categories...")
    test_row_count_and_categories()
    print("Running test_data_types_and_inr_conversion...")
    test_data_types_and_inr_conversion()
    print("Running test_sqlite_normalization_and_fk...")
    test_sqlite_normalization_and_fk()
    print("Running test_sql_join_matches_pandas_merge...")
    test_sql_join_matches_pandas_merge()
    print("\nALL DATA PIPELINE TESTS PASSED!")
