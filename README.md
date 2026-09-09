# Zepto Data & AI Platform

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5.23-orange.svg)](https://www.trychroma.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-StateGraph-purple.svg)](https://langchain-ai.github.io/langgraph/)

An enterprise-grade, end-to-end data engineering, machine learning analytics, and agentic AI customer support platform designed for Zepto quick-commerce operations.

---

## Repository Structure

```text
/
├── README.md                          # Comprehensive master project documentation
├── requirements.txt                   # Consolidated project dependency manifest
├── PROJECT_AUDIT.md                   # Requirement-by-requirement verification audit
│
├── data_pipeline/                     # MODULE 1: E-commerce Web Scraping & Relational ETL
│   ├── database.py                    # Normalized SQLite DDL schema and connection managers
│   ├── scrape_pipeline.py             # BeautifulSoup scraper, cleaning, currency conversion & loader
│   ├── queries.sql                    # 5+ SQL queries demonstrating WHERE, ORDER, LIMIT, DISTINCT, JOIN
│   ├── run_queries.py                 # Query executor, output logger, and Pandas merge equivalence validator
│   ├── test_data_pipeline.py          # Automated test suite for data pipeline criteria
│   ├── query_outputs.txt              # Formatted execution logs for all SQL queries
│   ├── zepto_books.db                 # Relational SQLite database with enforced PK/FK
│   ├── requirements.txt               # Module-specific dependencies
│   ├── README.md                      # Detailed data pipeline architectural documentation
│   └── data/
│       ├── raw_books.csv              # Raw extracted catalog records
│       └── cleaned_books.csv          # Cleaned dataset with 105.50 GBP->INR conversion
│
├── analytics/                         # MODULE 2: Titanic Predictive Analytics & Modeling
│   ├── titanic.csv                    # Offline fallback dataset (loaded once via seaborn)
│   ├── 01_eda.ipynb                   # Jupyter notebook for profiling, missing value handling & EDA
│   ├── 02_modeling.ipynb              # Jupyter notebook for classification, SMOTE, GridSearch & regression
│   ├── run_analytics.py               # Standalone, end-to-end executable ML script
│   ├── test_analytics.py              # Automated test suite for analytics acceptance criteria
│   ├── best_titanic_pipeline.joblib   # Complete fitted scikit-learn pipeline (preprocessor + estimator)
│   ├── requirements.txt               # Module-specific dependencies
│   ├── README.md                      # Detailed analytics methodology and metric findings
│   └── plots/                         # High-resolution visualization artifacts
│       ├── age_distribution.png       # Age histogram and boxplot with IQR outlier bounds
│       ├── fare_distribution.png      # Fare distribution showing mean > median > mode skewness
│       ├── correlation_heatmap.png    # 6x6 correlation matrix heatmap (excluding adult_male & alone)
│       ├── chart1_survival_by_sex_pclass.png    # Multivariate Chart 1: Gender & Class survival
│       ├── chart2_fare_by_class_survival.png    # Multivariate Chart 2: Fare distribution & survival
│       ├── chart3_age_fare_survival_scatter.png # Multivariate Chart 3: Age vs Fare scatter
│       ├── chart4_embarked_survival_distribution.png # Multivariate Chart 4: Port & Class survival
│       ├── decision_tree_plot.png     # Rendered decision tree structure (max_depth=4)
│       ├── roc_curves.png             # Multi-model ROC curves and AUC comparison
│       └── regression_residuals.png   # Fare regression residual plot exhibiting heteroscedasticity
│
└── support_assistant/                 # MODULE 3: Agentic Policy RAG Support Service
    ├── docs/                          # Exact 8 canonical Zepto policy text documents
    │   ├── doc_01.txt                 # Delivery Policy
    │   ├── doc_02.txt                 # Returns & Refunds
    │   ├── doc_03.txt                 # Membership Tiers (Zepto Pass)
    │   ├── doc_04.txt                 # Order Tracking
    │   ├── doc_05.txt                 # Order Cancellation Policy
    │   ├── doc_06.txt                 # Damaged or Missing Items
    │   ├── doc_07.txt                 # Gift Cards
    │   └── doc_08.txt                 # Customer Support Hours
    ├── chroma_db/                     # Persistent ChromaDB vector index (zepto_policies)
    ├── ingest.py                      # Vector ingestion using local sentence-transformers (all-MiniLM-L6-v2)
    ├── rag_graph.py                   # LangGraph StateGraph engine with keyword heuristic & top-3 cosine RAG
    ├── main.py                        # FastAPI microservice exposing POST /ask
    ├── test_support_assistant.py      # Automated test suite for support assistant criteria
    ├── Dockerfile                     # Containerization manifest exposing port 7860
    ├── requirements.txt               # Module-specific dependencies
    └── README.md                      # Detailed RAG architecture and API documentation
```

---

## Installation

The repository supports both a single consolidated dependency installation and isolated per-module installations.

### Option A: Consolidated Installation (Recommended)
```bash
# Clone the repository (if working on a fresh workstation)
git clone https://github.com/rahulasula8-max/zepto-ai-ml-capstone.git
cd zepto-ai-ml-capstone

# Install all dependencies across all three modules
pip install -r requirements.txt
```

### Option B: Per-Module Installation
```bash
# Data Pipeline dependencies
pip install -r data_pipeline/requirements.txt

# Analytics & Machine Learning dependencies
pip install -r analytics/requirements.txt

# Support Assistant & Vector RAG dependencies
pip install -r support_assistant/requirements.txt
```

---

## Module 1 — Data Pipeline

### Overview
Extracts catalog records from [Books to Scrape](https://books.toscrape.com/), cleans attributes, applies a fixed currency conversion, normalizes records into a relational SQLite database with primary and foreign key constraints, and validates SQL queries against Pandas merges.

### Key Capabilities
- **Scraping**: Fetches 69+ books across 3+ categories (`Travel`, `Mystery`, `Historical Fiction`) using `requests` and `BeautifulSoup4`.
- **Cleaning & Currency Conversion**:
  - Parses text star ratings (`One`..`Five`) into numeric integers (`1..5`).
  - Converts availability strings into boolean flags (`1`/`0`).
  - Converts GBP to INR using the exact required fixed rate:
    $$\text{price\_inr} = \text{round}(\text{price\_gbp} \times 105.50, 2)$$
  - Implements defensive handling: handles missing numeric records via median imputation or row drop without crashing.
- **Normalized SQLite Schema**:
  - `categories` (Parent): `category_id` (PK), `category_name` (UNIQUE).
  - `books` (Child): `book_id` (PK), `title`, `price_gbp`, `price_inr`, `rating`, `in_stock`, `category_id` (FK referencing `categories.category_id` with `PRAGMA foreign_keys = ON;`).
- **SQL & Pandas Equivalence**:
  - Executes 5+ SQL queries demonstrating `SELECT / WHERE`, `ORDER BY`, `LIMIT`, `DISTINCT`, `BETWEEN`, `IN`, and `INNER JOIN`.
  - Reads results into DataFrames via `pd.read_sql`.
  - Re-implements relational join via `pd.merge(df_books, df_categories, on='category_id')`.
  - Verifies exact DataFrame equivalence with `pd.testing.assert_frame_equal`.

### Run Commands
```bash
# Run scraping, cleaning, and SQLite database population
python data_pipeline/scrape_pipeline.py

# Execute SQL queries and verify Pandas merge equivalence
python data_pipeline/run_queries.py

# Run automated tests
python data_pipeline/test_data_pipeline.py
```

---

## Module 2 — Analytics

### Overview
A leak-free machine learning workflow analyzing passenger survival patterns and fare distributions on the RMS Titanic.

### Key Capabilities
- **Single Load & Offline Fallback**: `sns.load_dataset('titanic')` was invoked **strictly once** to generate `titanic.csv`. All exploratory and predictive tasks consume `titanic.csv` without network access.
- **Missing Value Handling**:
  - `<5%` (`embarked`, `embark_town` at 0.22%): Dropped rows for profiling; handled via most frequent imputer in production pipeline.
  - `5%–30%` (`age` at 19.87%): Median imputation (28.0 years) to remain robust against skewness.
  - `>30%` (`deck` at 77.22%): Dropped column due to severe sparsity.
- **Univariate Statistics & Skewness**:
  - Outlier bounds via IQR ($Q_1 - 1.5\text{IQR}, Q_3 + 1.5\text{IQR}$): Age outliers = 11 (1.54%), Fare outliers = 116 (13.02%).
  - Central tendencies for fare: Mean (£32.20) > Median (£14.45) > Mode (£8.05). Since Mean > Median > Mode, the fare distribution is **positively (right) skewed**.
- **Bivariate Survival Rates (Boolean Masking)**:
  - By Gender: Female = **74.20%**, Male = **18.89%**.
  - By Class: Class 1 = **62.96%**, Class 2 = **47.28%**, Class 3 = **24.24%**.
  - By Gender & Class: Female Class 1 (**96.81%**) down to Male Class 3 (**13.54%**).
- **Exact 6-Column Correlation Matrix**:
  - Strictly includes: `survived`, `pclass`, `age`, `sibsp`, `parch`, `fare` (excludes `adult_male` and `alone`).
  - Top 2 Off-Diagonal Correlations:
    1. `pclass` $\leftrightarrow$ `fare` ($r = -0.5495$): Lower classes had significantly cheaper fares.
    2. `sibsp` $\leftrightarrow$ `parch` ($r = +0.4148$): Family members traveled together.
- **Multivariate Data Story**: 4 distinct plots saved to `plots/` with 2-4 sentence Markdown interpretations detailing survival advantage.
- **EDA Standardization Sanity Check**: Isolated z-score check on `age` and `fare` ($\mu \approx 0, \sigma \approx 1$); strictly quarantined from modeling.
- **Modeling & Leak-Free Pipeline**:
  - Stratified 80/20 train/test split performed **prior** to preprocessing.
  - `ColumnTransformer` (median imputation + `StandardScaler` for numeric, most-frequent imputation + `OneHotEncoder` for categorical) fitted **strictly on training fold**.
  - Classifiers trained on identical split: Logistic Regression, Decision Tree (`plot_tree` rendered with feature/class names), Random Forest.
- **Class Imbalance Comparison (Logistic Regression)**:
  - Baseline: Prec = 0.7931, Rec = 0.6667, F1 = 0.7244
  - `class_weight='balanced'`: Prec = 0.7297, Rec = 0.7826, F1 = 0.7552
  - **SMOTE (Training-Only)**: Prec = 0.7397, Rec = 0.7826, **F1 = 0.7606**
- **Random Forest Hyperparameter Tuning**:
  - `GridSearchCV` with `oob_score=True` yielding best parameters `max_depth=8, max_features='log2', n_estimators=50`.
  - Out-of-Bag (OOB) Score: **0.8188**. Test AUC: **0.8458**.
- **Multivariate Linear Regression Side Task (Predicting Fare)**:
  - MAE = £20.81, RMSE = £30.47, $R^2$ = 0.3999, Adjusted $R^2$ = 0.3790.
  - Residual plot displays clear funnel pattern confirming **heteroscedasticity** due to first-class luxury outliers.
- **Model Comparison Table**: Maintained in two separate metric groups (Classification vs Regression).
- **Deployment Recommendation**:
  > For operational deployment, the Tuned Random Forest Classifier is strongly recommended, achieving the highest overall test accuracy of 79.9%, an F1-score of 0.710, and an outstanding ROC-AUC of 0.846. While Logistic Regression attained acceptable baseline recall (0.667), Random Forest's non-linear ensemble architecture significantly reduces false positives, outperforming the single Decision Tree by 2.5 AUC percentage points. Furthermore, its out-of-bag validation score of 0.819 demonstrates superior generalization resilience against overfitting on unseen passenger cohorts.
- **Full Pipeline Persistence**: Best fitted pipeline serialized to `best_titanic_pipeline.joblib`. Reloaded via `joblib.load()` and verified to predict raw inputs containing missing values accurately.

### Run Commands
```bash
# Run standalone end-to-end script (generates metrics, plots, and saved pipeline)
python analytics/run_analytics.py

# Run automated test suite
python analytics/test_analytics.py
```

---

## Module 3 — Support Assistant

### Overview
A production-ready, fully offline deterministic customer policy support service built with **LangGraph StateGraph**, local **Sentence-Transformers** (`all-MiniLM-L6-v2`), persistent **ChromaDB**, **Pydantic**, and **FastAPI**.

### Key Capabilities
- **8 Canonical Policy Documents**: Managed under `docs/` (`doc_01.txt` to `doc_08.txt`) covering delivery, returns, membership, tracking, cancellations, damages, gift cards, and support hours.
- **Local Dense Embeddings & ChromaDB**:
  - Text indexed into ChromaDB collection `zepto_policies` using cosine similarity (`space: cosine`).
  - Runs completely offline without API keys.
- **LangGraph StateGraph Engine**:
  - TypedDict state: `query`, `intent`, `retrieved_docs`, `sources`, `answer`, `confidence`.
  - Named nodes: `classify_intent`, `retrieve_and_answer`, `direct_answer`.
  - Keyword heuristic: routes queries containing `delivery`, `return`, `refund`, `membership`, `tracking`, `cancel`, `gift card`, `support hours` to `policy_question`, else `general_question`.
  - Retrieval: Queries ChromaDB for top 3 documents using cosine similarity.
- **MOCK_LLM Branching Architecture**:
  - **`MOCK_LLM=1` / Unset (Default Graded Baseline)**:
    - Policy query: `Based on the retrieved context: {top_chunk_snippet}`. Sources = retrieved IDs, confidence = normalized cosine similarity.
    - General query: `I can only answer questions about Zepto policies right now.`. Sources = `[]`, confidence = 1.0.
  - **`MOCK_LLM=0` (Optional Real LLM Mode)**:
    - Leverages a structured prompt containing role, context, task, format, length (<3 sentences), negative constraint, and few-shot example.
    - Employs up to 2 validation retries upon schema failure.
- **Pydantic Validation**: Guarantees response model fields `answer: str`, `sources: list`, and `confidence: float` ($0.0 \le \text{confidence} \le 1.0$).
- **FastAPI**: Serves POST `/ask` and GET `/health` with sub-millisecond response latency.
- **Docker Ready**: Fully containerized with pre-indexed vector store.

### Run Commands
```bash
# 1. Run automated test suite
python support_assistant/test_support_assistant.py

# 2. Launch FastAPI service via Uvicorn (port 7860)
uvicorn support_assistant.main:app --host 0.0.0.0 --port 7860
```

---

## Example API Calls (Raw JSON Outputs)

### Example 1: Retrieval / Policy Query
**Request**:
```bash
curl -X POST http://localhost:7860/ask \
     -H "Content-Type: application/json" \
     -d '{"query": "What is your return and refund policy?"}'
```

**Raw JSON Response**:
```json
{
  "answer": "Based on the retrieved context: Zepto Returns & Refunds Policy:\nCustomers may request returns or refunds for eligible items directly via the Zepto mobile app under the Order History section. For perishable items including fresh frui",
  "sources": [
    "doc_02",
    "doc_05",
    "doc_06"
  ],
  "confidence": 0.41
}
```

### Example 2: General / Non-Policy Query
**Request**:
```bash
curl -X POST http://localhost:7860/ask \
     -H "Content-Type: application/json" \
     -d '{"query": "How do I build a wooden chair?"}'
```

**Raw JSON Response**:
```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

---

## Docker Instructions

The Support Assistant includes a standalone `Dockerfile` ready for local containerization:

```bash
# 1. Build the Docker container image
docker build -t zepto-support-assistant -f support_assistant/Dockerfile support_assistant/

# 2. Run the container locally mapping port 7860
docker run -d --name zepto-assistant -p 7860:7860 -e MOCK_LLM=1 zepto-support-assistant

# 3. Test container health
curl http://localhost:7860/health

# 4. Issue a policy query to the containerized assistant
curl -X POST http://localhost:7860/ask \
     -H "Content-Type: application/json" \
     -d '{"query": "What is Zepto delivery policy?"}'

# 5. Stop container
docker stop zepto-assistant
```

---

## Key Design Decisions

1. **Module 1 (Data Pipeline)**:
   - *Fixed Currency Conversion*: Explicitly codified the conversion baseline of 1 GBP = 105.50 INR as a system constant rather than querying external volatile currency APIs, ensuring 100% deterministic ETL reproducibility.
   - *Relational Normalization*: Enforced strict primary and foreign key constraints in SQLite with `PRAGMA foreign_keys = ON;`, preventing orphaned catalog records and establishing a clean schema for SQL JOIN operations.

2. **Module 2 (Analytics)**:
   - *Data Leakage Prevention*: Isolated preprocessing fit strictly to the training fold of a stratified 80/20 train/test split. No test or pre-split data touches the imputers, encoders, or scalers.
   - *Training-Only SMOTE*: Applied synthetic minority oversampling exclusively to the training fold feature matrix, ensuring evaluation test folds reflect empirical class distributions.
   - *Full Pipeline Persistence*: Saved the unified `ColumnTransformer` + estimator pipeline via `joblib.dump` rather than a bare estimator, allowing raw, un-preprocessed DataFrames with missing values to be ingested directly in production.

3. **Module 3 (Support Assistant)**:
   - *Offline-First Determinism*: Configured `MOCK_LLM=1` as the graded default baseline. Local `all-MiniLM-L6-v2` embeddings and ChromaDB operate entirely offline without network calls or API keys.
   - *LangGraph Agentic Routing*: Decoupled intent classification from generation. The conditional edge from `classify_intent` routes queries strictly based on domain relevance, while node generation independently respects the `MOCK_LLM` flag.

---

## Testing & Quality Verification

Comprehensive automated test suites have been developed for each module:

```bash
# Test Module 1 (Data Pipeline)
python data_pipeline/test_data_pipeline.py

# Test Module 2 (Analytics)
python analytics/test_analytics.py

# Test Module 3 (Support Assistant)
python support_assistant/test_support_assistant.py
```

All acceptance checks pass with zero warnings or errors. For a full breakdown, consult [PROJECT_AUDIT.md](PROJECT_AUDIT.md).
