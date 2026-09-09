-- ====================================================================
-- ZEPTO DATA PIPELINE: SQL QUERIES DEMONSTRATION
-- Normalized SQLite schema: categories (1) -> books (M)
-- ====================================================================

-- --------------------------------------------------------------------
-- QUERY 1: SELECT / WHERE
-- Filter books that are in stock and have a star rating of 4 or 5
-- --------------------------------------------------------------------
SELECT 
    book_id,
    title,
    price_gbp,
    price_inr,
    rating,
    in_stock
FROM books
WHERE in_stock = 1 AND rating >= 4;

-- --------------------------------------------------------------------
-- QUERY 2: ORDER BY + LIMIT
-- Retrieve top 10 most expensive books ordered by price_inr descending
-- --------------------------------------------------------------------
SELECT 
    book_id,
    title,
    price_inr,
    rating
FROM books
ORDER BY price_inr DESC
LIMIT 10;

-- --------------------------------------------------------------------
-- QUERY 3: DISTINCT
-- Find all distinct star rating values present across all books
-- --------------------------------------------------------------------
SELECT DISTINCT 
    rating
FROM books
ORDER BY rating ASC;

-- --------------------------------------------------------------------
-- QUERY 4: BETWEEN (and IN)
-- Find books with price_inr BETWEEN 2000.00 and 4500.00 having rating IN (3, 4, 5)
-- --------------------------------------------------------------------
SELECT 
    book_id,
    title,
    price_inr,
    rating
FROM books
WHERE price_inr BETWEEN 2000.00 AND 4500.00
  AND rating IN (3, 4, 5)
ORDER BY price_inr ASC;

-- --------------------------------------------------------------------
-- QUERY 5: INNER JOIN (with Aggregation & Filtering)
-- Join books and categories on category_id, showing category name and pricing
-- --------------------------------------------------------------------
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

-- --------------------------------------------------------------------
-- BONUS QUERY 6: Category Summary (GROUP BY + JOIN)
-- Aggregate summary statistics by book category
-- --------------------------------------------------------------------
SELECT 
    c.category_name,
    COUNT(b.book_id) AS total_books,
    ROUND(AVG(b.price_inr), 2) AS avg_price_inr,
    ROUND(MIN(b.price_inr), 2) AS min_price_inr,
    ROUND(MAX(b.price_inr), 2) AS max_price_inr,
    ROUND(AVG(b.rating), 2) AS avg_rating
FROM categories c
INNER JOIN books b ON c.category_id = b.category_id
GROUP BY c.category_id, c.category_name
ORDER BY total_books DESC;
