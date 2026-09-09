"""SQL Query Execution & Pandas Validation Module.
Executes normalized SQLite queries, records outputs, reads query results via pd.read_sql,
and reproduces the SQL JOIN using pd.merge, proving dataframe equivalence.
"""
from pathlib import Path
import sqlite3
import pandas as pd
from database import DB_PATH, get_connection

QUERIES_SQL_PATH = Path(__file__).resolve().parent / "queries.sql"
OUTPUT_TXT_PATH = Path(__file__).resolve().parent / "query_outputs.txt"


def execute_and_log_queries() -> None:
    """Read queries from queries.sql, execute each, and log outputs to text file."""
    conn = get_connection(DB_PATH)
    cursor = conn.cursor()

    with open(QUERIES_SQL_PATH, "r", encoding="utf-8") as f:
        sql_content = f.read()

    # Split queries by semicolon
    statements = [stmt.strip() for stmt in sql_content.split(";") if stmt.strip() and not stmt.strip().startswith("-- ==")]

    output_lines = []
    output_lines.append("====================================================================")
    output_lines.append("ZEPTO DATA PIPELINE: SQL QUERY EXECUTION RESULTS")
    output_lines.append(f"Database: {DB_PATH.name}")
    output_lines.append("====================================================================\n")

    for idx, stmt in enumerate(statements, 1):
        clean_stmt = "\n".join([line for line in stmt.splitlines() if not line.strip().startswith("--")]).strip()
        if not clean_stmt:
            continue
        output_lines.append(f"--- QUERY {idx} ---")
        output_lines.append(f"SQL:\n{clean_stmt}\n")

        cursor.execute(clean_stmt)
        rows = cursor.fetchall()
        col_names = [desc[0] for desc in cursor.description]
        output_lines.append(f"Row count: {len(rows)}")
        output_lines.append(f"Columns: {', '.join(col_names)}")

        # Print first up to 10 rows for display
        df_preview = pd.DataFrame([dict(r) for r in rows])
        output_lines.append(df_preview.head(10).to_string(index=False))
        if len(rows) > 10:
            output_lines.append(f"... ({len(rows) - 10} more rows)")
        output_lines.append("\n" + "=" * 68 + "\n")

    conn.close()

    with open(OUTPUT_TXT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    print(f"Executed all SQL queries and saved outputs to {OUTPUT_TXT_PATH}")


def demonstrate_pandas_integration() -> None:
    """Demonstrate pd.read_sql on at least 2 queries and prove pd.merge equivalence to SQL JOIN."""
    conn = get_connection(DB_PATH)

    print("\n--- PANDAS READ_SQL DEMONSTRATION ---")

    # 1. pd.read_sql on Query 1 (High-rated in-stock books)
    q1 = "SELECT book_id, title, price_inr, rating FROM books WHERE in_stock = 1 AND rating >= 4;"
    df_q1 = pd.read_sql(q1, conn)
    print(f"pd.read_sql (Query 1 - in_stock & rating>=4): {len(df_q1)} rows returned.")
    print(df_q1.head(3))

    # 2. pd.read_sql on Query 2 (Top 5 expensive books)
    q2 = "SELECT book_id, title, price_inr, rating FROM books ORDER BY price_inr DESC LIMIT 5;"
    df_q2 = pd.read_sql(q2, conn)
    print(f"\npd.read_sql (Query 2 - top 5 expensive): {len(df_q2)} rows returned.")
    print(df_q2)

    # 3. SQL JOIN Query vs pd.merge Reproducibility
    print("\n--- SQL JOIN vs PD.MERGE EQUIVALENCE VERIFICATION ---")
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
    df_sql_join = pd.read_sql(sql_join, conn)

    # Fetch raw books and categories tables independently
    df_books_raw = pd.read_sql("SELECT book_id, title, price_gbp, price_inr, rating, in_stock, category_id FROM books;", conn)
    df_categories_raw = pd.read_sql("SELECT category_id, category_name FROM categories;", conn)

    # Perform pandas merge equivalent
    df_pandas_merged = pd.merge(
        df_books_raw,
        df_categories_raw,
        on="category_id",
        how="inner"
    )

    # Select and reorder columns to match SQL JOIN projection exactly
    cols_order = ["book_id", "title", "category_name", "price_gbp", "price_inr", "rating", "in_stock"]
    df_pandas_merged = df_pandas_merged[cols_order].sort_values("book_id").reset_index(drop=True)
    df_sql_join = df_sql_join[cols_order].sort_values("book_id").reset_index(drop=True)

    # Verify equivalence
    pd.testing.assert_frame_equal(df_sql_join, df_pandas_merged, check_dtype=True)
    print("SUCCESS: pd.testing.assert_frame_equal confirmed that SQL JOIN and pd.merge produce identical datasets!")
    print(f"Total merged rows matched: {len(df_pandas_merged)}")
    print("Sample matched row:")
    print(df_pandas_merged.head(1).to_dict(orient="records")[0])

    conn.close()


if __name__ == "__main__":
    execute_and_log_queries()
    demonstrate_pandas_integration()
