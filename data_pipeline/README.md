# Module 1 — Data Pipeline: Books to Scrape ETL

## Overview
This module automates the extraction, cleaning, normalization, relational storage, and SQL/Pandas querying of e-commerce book catalog data from [Books to Scrape](https://books.toscrape.com/).

## Key Architectural Highlights
1. **Scraping & Extraction (`scrape_pipeline.py`)**:
   - Built using `requests` and `BeautifulSoup4`.
   - Traverses multiple book categories (`Travel`, `Mystery`, `Historical Fiction`, etc.).
   - Captures `title`, `price_gbp`, `star_rating`, `availability`, and `category`.
   - Exceeds the minimum threshold by collecting 69+ books across 3+ categories.
   - Saves raw scraped records to `data/raw_books.csv`.

2. **Data Cleaning & Fixed Currency Conversion**:
   - Strips currency glyphs and parses `price_gbp` to standard `float`.
   - Maps textual star ratings (`One`, `Two`, `Three`, `Four`, `Five`) to integers `1..5`.
   - Maps availability strings to boolean `1`/`0` (`in_stock`).
   - Converts British Pounds (GBP) to Indian Rupees (INR) using the EXACT required conversion factor:
     $$\text{price\_inr} = \text{round}(\text{price\_gbp} \times 105.50, 2)$$
   - Implements defensive handling: if missing numeric entries occur, the pipeline applies median imputation or records justification without crashing.
   - Saves cleaned data to `data/cleaned_books.csv`.

3. **Normalized SQLite Database (`database.py` & `zepto_books.db`)**:
   - Enforces relational normalization into two tables with `PRAGMA foreign_keys = ON;`:
     - **`categories`** (Parent): `category_id INTEGER PRIMARY KEY AUTOINCREMENT`, `category_name TEXT UNIQUE NOT NULL`.
     - **`books`** (Child): `book_id INTEGER PRIMARY KEY AUTOINCREMENT`, `title TEXT NOT NULL`, `price_gbp REAL NOT NULL`, `price_inr REAL NOT NULL`, `rating INTEGER NOT NULL`, `in_stock INTEGER NOT NULL`, `category_id INTEGER NOT NULL REFERENCES categories(category_id)`.
   - Rejects invalid foreign key references via database-level constraints.

4. **SQL Queries & Pandas Integration (`queries.sql` & `run_queries.py`)**:
   - Covers all required SQL clauses:
     - `SELECT / WHERE`: Filter in-stock books with rating $\ge 4$.
     - `ORDER BY` + `LIMIT`: Top 10 most expensive books by INR price.
     - `DISTINCT`: Unique star ratings across catalog.
     - `IN` / `BETWEEN`: Price range filtering combined with rating sets.
     - `INNER JOIN`: Relational join between `books` and `categories`.
   - Uses `pd.read_sql` to execute queries and load result DataFrames directly.
   - Independently reconstructs the SQL JOIN using `pd.merge(df_books, df_categories, on="category_id")`.
   - Formally verifies that SQL JOIN and `pd.merge` yield identical results using `pd.testing.assert_frame_equal`.

## Execution Commands
From the repository root:
```bash
# 1. Run scraping, cleaning, and SQLite database ingestion
python data_pipeline/scrape_pipeline.py

# 2. Run SQL queries, log outputs, and verify Pandas merge equivalence
python data_pipeline/run_queries.py

# 3. Run automated test suite
python data_pipeline/test_data_pipeline.py
```

## Directory Contents
- `database.py`: Database connection and DDL schema definitions.
- `scrape_pipeline.py`: Scraping, cleaning, and SQLite loading logic.
- `queries.sql`: SQL query scripts covering required query types.
- `run_queries.py`: Query executor and Pandas equivalence validator.
- `test_data_pipeline.py`: Automated unit tests for all grading criteria.
- `query_outputs.txt`: Recorded outputs from each SQL query execution.
- `data/raw_books.csv`: Raw scraped books data.
- `data/cleaned_books.csv`: Cleaned dataset with INR currency conversion.
- `zepto_books.db`: SQLite database file.
