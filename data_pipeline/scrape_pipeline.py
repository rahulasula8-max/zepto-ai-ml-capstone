"""Scraping and ETL Pipeline for Books to Scrape (books.toscrape.com).
Extracts book records across multiple categories, cleans and converts currency,
and loads data into normalized SQLite tables with primary/foreign keys.
"""
import logging
import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

from database import DB_PATH, get_connection, init_db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

BASE_URL = "https://books.toscrape.com/"
DATA_DIR = Path(__file__).resolve().parent / "data"
RAW_CSV_PATH = DATA_DIR / "raw_books.csv"
CLEAN_CSV_PATH = DATA_DIR / "cleaned_books.csv"

# Fixed exchange rate specified by assignment requirements
GBP_TO_INR = 105.50

RATING_MAP = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5
}


def fetch_soup(url: str, timeout: int = 15) -> BeautifulSoup:
    """Fetch HTML from URL and return parsed BeautifulSoup object."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ZeptoDataPipeline/1.0"
    }
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def get_categories(soup: BeautifulSoup) -> Dict[str, str]:
    """Extract category names and full URLs from the home page."""
    categories: Dict[str, str] = {}
    category_links = soup.select("ul.nav-list > li > ul > li > a")
    for link in category_links:
        name = link.text.strip()
        full_url = urljoin(BASE_URL, link["href"])
        categories[name] = full_url
    return categories


def scrape_books(target_categories: Optional[List[str]] = None, min_books: int = 60) -> List[Dict[str, str]]:
    """Scrape books across categories until at least min_books and >=3 categories are collected."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Fetching categories from books.toscrape.com...")
    home_soup = fetch_soup(BASE_URL)
    all_categories = get_categories(home_soup)

    if not target_categories:
        # Select categories with good book volume
        target_categories = ["Travel", "Mystery", "Historical Fiction", "Sequential Art", "Classics"]

    scraped_data: List[Dict[str, str]] = []

    for cat_name in target_categories:
        if cat_name not in all_categories:
            continue
        cat_url = all_categories[cat_name]
        current_url = cat_url
        cat_books_count = 0

        logger.info("Scraping category: %s from %s", cat_name, current_url)
        while current_url:
            soup = fetch_soup(current_url)
            articles = soup.find_all("article", class_="product_pod")

            for art in articles:
                # 1. Title
                title_elem = art.h3.find("a")
                title = title_elem.get("title") or title_elem.text.strip()

                # 2. Price GBP raw
                price_elem = art.find("p", class_="price_color")
                price_raw = price_elem.text.strip() if price_elem else ""

                # 3. Star rating text
                rating_p = art.find("p", class_="star-rating")
                rating_classes = rating_p.get("class", []) if rating_p else []
                rating_text = "None"
                for cls in rating_classes:
                    if cls.lower() in RATING_MAP:
                        rating_text = cls
                        break

                # 4. Availability text
                avail_elem = art.find("p", class_="instock availability")
                availability_raw = avail_elem.text.strip() if avail_elem else ""

                scraped_data.append({
                    "title": title,
                    "price_gbp": price_raw,
                    "star_rating": rating_text,
                    "availability": availability_raw,
                    "category": cat_name
                })
                cat_books_count += 1

            # Check next page within category
            next_btn = soup.select_one("li.next > a")
            if next_btn:
                current_url = urljoin(current_url, next_btn["href"])
            else:
                current_url = None

        logger.info("Category '%s' finished: %d books scraped.", cat_name, cat_books_count)
        if len(scraped_data) >= min_books and len(set(b["category"] for b in scraped_data)) >= 3:
            logger.info("Reached threshold of %d books across %d categories.", len(scraped_data), len(set(b["category"] for b in scraped_data)))
            break

    raw_df = pd.DataFrame(scraped_data)
    raw_df.to_csv(RAW_CSV_PATH, index=False)
    logger.info("Saved raw scraped data (%d records) to %s", len(raw_df), RAW_CSV_PATH)
    return scraped_data


def clean_book_record(record: Dict[str, str]) -> Optional[Dict]:
    """Clean individual book dictionary with defensive error handling."""
    title = str(record.get("title", "")).strip()
    if not title:
        return None  # Drop rows without title

    # Clean Price GBP
    raw_price = str(record.get("price_gbp", ""))
    price_match = re.search(r"(\d+(?:\.\d+)?)", raw_price)
    if price_match:
        try:
            price_gbp = float(price_match.group(1))
        except ValueError:
            price_gbp = None
    else:
        price_gbp = None

    # Clean Rating
    rating_raw = str(record.get("star_rating", "")).strip().lower()
    rating = RATING_MAP.get(rating_raw, None)

    # Clean Availability
    avail_raw = str(record.get("availability", "")).strip().lower()
    in_stock = 1 if "in stock" in avail_raw else 0

    category = str(record.get("category", "General")).strip()

    return {
        "title": title,
        "price_gbp": price_gbp,
        "rating": rating,
        "in_stock": in_stock,
        "category": category
    }


def clean_scraped_data(raw_records: List[Dict[str, str]]) -> pd.DataFrame:
    """Clean records, handle missing numeric values via median imputation or drop, and compute price_inr."""
    cleaned_list = []
    for rec in raw_records:
        cleaned = clean_book_record(rec)
        if cleaned is not None:
            cleaned_list.append(cleaned)

    df = pd.DataFrame(cleaned_list)

    # Defensive handling for missing numeric values:
    # If price_gbp has missing values, impute with median of existing valid prices
    if df["price_gbp"].isnull().any():
        median_price = df["price_gbp"].median()
        logger.warning("Imputing %d missing price_gbp values with median: £%.2f", df["price_gbp"].isnull().sum(), median_price)
        df["price_gbp"].fillna(median_price, inplace=True)

    # If rating is missing, default to median rating rounded to int (3)
    if df["rating"].isnull().any():
        median_rating = int(round(df["rating"].median()))
        logger.warning("Imputing %d missing ratings with median: %d", df["rating"].isnull().sum(), median_rating)
        df["rating"].fillna(median_rating, inplace=True)
    df["rating"] = df["rating"].astype(int)

    # Calculate price_inr using EXACT fixed rate 105.50
    df["price_inr"] = (df["price_gbp"] * GBP_TO_INR).round(2)

    df.to_csv(CLEAN_CSV_PATH, index=False)
    logger.info("Saved cleaned data (%d records) to %s", len(df), CLEAN_CSV_PATH)
    return df


def load_to_sqlite(df: pd.DataFrame, db_path: Path = DB_PATH) -> None:
    """Load cleaned DataFrame into normalized SQLite database with PK/FK constraints."""
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()

    # 1. Populate categories table
    unique_categories = sorted(df["category"].unique())
    cat_to_id: Dict[str, int] = {}
    for cat in unique_categories:
        cursor.execute("INSERT INTO categories (category_name) VALUES (?);", (cat,))
        cat_to_id[cat] = cursor.lastrowid
    logger.info("Inserted %d categories into SQLite.", len(cat_to_id))

    # 2. Populate books table referencing category_id
    books_rows = []
    for _, row in df.iterrows():
        cat_id = cat_to_id[row["category"]]
        books_rows.append((
            row["title"],
            float(row["price_gbp"]),
            float(row["price_inr"]),
            int(row["rating"]),
            int(row["in_stock"]),
            cat_id
        ))

    cursor.executemany("""
    INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id)
    VALUES (?, ?, ?, ?, ?, ?);
    """, books_rows)

    conn.commit()
    logger.info("Inserted %d books into SQLite table 'books'.", len(books_rows))
    conn.close()


def run_pipeline() -> Tuple[pd.DataFrame, int, int]:
    """Execute complete end-to-end data pipeline."""
    raw_records = scrape_books(min_books=60)
    cleaned_df = clean_scraped_data(raw_records)
    load_to_sqlite(cleaned_df)

    conn = get_connection(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM books;")
    total_books = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM categories;")
    total_cats = cur.fetchone()[0]
    conn.close()

    logger.info("Data Pipeline Complete! Books: %d, Categories: %d", total_books, total_cats)
    return cleaned_df, total_books, total_cats


if __name__ == "__main__":
    run_pipeline()
