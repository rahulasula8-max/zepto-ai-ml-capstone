# Capstone Project Acceptance Audit

This audit validates all graded requirements across the Zepto Data & AI Platform repository.

| Requirement | File / Location | Status | Evidence |
| :--- | :--- | :---: | :--- |
| **GLOBAL: Single Repository** | `/` (Repository Root) | **PASS** | Entire capstone resides in `rahulasula8-max/zepto-ai-ml-capstone`. |
| **GLOBAL: Module Root Structure** | `/data_pipeline`, `/analytics`, `/support_assistant` | **PASS** | Verified clean directory layout with all three modules as top-level directories. |
| **GLOBAL: Master README** | `README.md` | **PASS** | Contains project overview, installation, module walkthroughs, design decisions, API examples, Docker instructions, and verification guides. |
| **GLOBAL: Markdown Documentation** | `README.md`, `data_pipeline/README.md`, `analytics/README.md`, `support_assistant/README.md`, notebooks | **PASS** | All interpretations, descriptions, and audit reports are in standard GitHub Flavored Markdown. No external PDF/Word documents. |
| **GLOBAL: Dependency Manifests** | `requirements.txt` + per-module `requirements.txt` | **PASS** | Both consolidated and per-module requirement files provided and documented. |
| **GLOBAL: Offline & Free-Tier Execution** | Repo-wide | **PASS** | 100% offline baseline execution using local SQLite, local CSVs, local sentence-transformers, and ChromaDB. Zero paid API keys required. |
| **M1: Web Scraping Engine** | `data_pipeline/scrape_pipeline.py:45-125` | **PASS** | Uses `requests` and `BeautifulSoup4` to extract books from `books.toscrape.com`. |
| **M1: Minimum Data Volume** | `data_pipeline/data/cleaned_books.csv` | **PASS** | Scraped 69 books across 3 categories (`Travel`, `Mystery`, `Historical Fiction`), exceeding >=60 threshold. |
| **M1: Attribute Extraction** | `data_pipeline/scrape_pipeline.py:75-108` | **PASS** | Captures `title`, raw `price_gbp`, raw `star_rating`, raw `availability`, and `category`. |
| **M1: Data Cleaning & Parsing** | `data_pipeline/scrape_pipeline.py:128-192` | **PASS** | Converts price to float, rating text (`One`..`Five`) to int `1..5`, availability to boolean `1`/`0`. |
| **M1: Fixed GBP->INR Baseline** | `data_pipeline/scrape_pipeline.py:32, 185` | **PASS** | Converts currency using exact fixed rate: `round(price_gbp * 105.50, 2)`. Validated by unit tests. |
| **M1: Defensive Error Handling** | `data_pipeline/scrape_pipeline.py:171-182` | **PASS** | Implements median imputation fallback for numeric fields and row-drop justification. |
| **M1: Normalized SQLite Schema** | `data_pipeline/database.py:22-52` | **PASS** | Two tables (`categories` and `books`) with Primary Keys and Foreign Key reference. |
| **M1: Relational Integrity (PK/FK)** | `data_pipeline/database.py:14, 48` | **PASS** | `PRAGMA foreign_keys = ON;` enforced. Invalid foreign keys rejected with `IntegrityError`. |
| **M1: 5+ SQL Queries** | `data_pipeline/queries.sql` | **PASS** | 6 SQL queries demonstrating `SELECT/WHERE`, `ORDER BY`, `LIMIT`, `DISTINCT`, `BETWEEN`, `IN`, and `INNER JOIN`. |
| **M1: Query Output Persistence** | `data_pipeline/query_outputs.txt` | **PASS** | Query execution outputs formatted and logged to file. |
| **M1: Pandas read_sql Integration** | `data_pipeline/run_queries.py:53-70` | **PASS** | Query 1 and Query 2 executed and loaded into DataFrames via `pd.read_sql`. |
| **M1: Pandas merge Equivalence** | `data_pipeline/run_queries.py:72-108` | **PASS** | Reconstructs SQL JOIN via `pd.merge(df_b, df_c, on='category_id')`. Verified identical with `pd.testing.assert_frame_equal`. |
| **M1: Automated Test Suite** | `data_pipeline/test_data_pipeline.py` | **PASS** | Unit test suite verifies all criteria; exits with `ALL DATA PIPELINE TESTS PASSED!`. |
| **M2: Single Load & Offline Fallback** | `analytics/titanic.csv` | **PASS** | `sns.load_dataset('titanic')` called strictly once during initialization. Saved to `titanic.csv` for 100% offline usage. |
| **M2: Profiling & Missing Values** | `analytics/run_analytics.py:65-90` | **PASS** | Evaluates `df.info()`, `df.describe()`, `df.shape` (891, 15), and exact missing percentages. |
| **M2: Threshold-Based Missing Strategy** | `analytics/run_analytics.py:84-90` | **PASS** | `<5%` (`embarked` 0.22%): drop/mode; `5-30%` (`age` 19.87%): median; `>30%` (`deck` 77.22%): dropped column. |
| **M2: Univariate Age & Fare IQR** | `analytics/run_analytics.py:92-136` | **PASS** | Age IQR = 17.88 (11 outliers). Fare IQR = 23.09 (116 outliers). Plots saved to `plots/`. |
| **M2: Fare Skewness Analysis** | `analytics/run_analytics.py:118-124` | **PASS** | Mean (£32.20) > Median (£14.45) > Mode (£8.05) rigorously proves positive (right) skewness. |
| **M2: Bivariate Boolean Masking** | `analytics/run_analytics.py:146-177` | **PASS** | Computes survival rates via boolean masks: Female 74.20%, Male 18.89%, Class 1 62.96%, Female C1 96.81%, Male C3 13.54%. |
| **M2: Exact 6-Column Correlation** | `analytics/run_analytics.py:179-210` | **PASS** | Strictly 6 columns (`survived`, `pclass`, `age`, `sibsp`, `parch`, `fare`). Excludes `adult_male` & `alone`. |
| **M2: Top 2 Off-Diagonal Correlations** | `analytics/run_analytics.py:190-204` | **PASS** | Identifies `pclass <-> fare` ($r=-0.5495$) and `sibsp <-> parch` ($r=+0.4148$) with interpretations. |
| **M2: 4 Multivariate Charts & Interpretations** | `analytics/plots/chart1..4.png` | **PASS** | 4 distinct charts saved to `plots/` with 2-4 sentence Markdown interpretations detailing survival dynamics. |
| **M2: EDA Standardization Sanity Check** | `analytics/run_analytics.py:246-261` | **PASS** | Standardized `age` and `fare` ($\mu \approx 0, \sigma \approx 1$). Quarantined from modeling pipeline. |
| **M2: Stratified Split Before Preprocessing** | `analytics/run_analytics.py:284-290` | **PASS** | 80/20 train/test split stratified on `survived` executed prior to any transformer fitting. |
| **M2: Training-Only Preprocessing** | `analytics/run_analytics.py:292-310` | **PASS** | `ColumnTransformer` (median imputer + `StandardScaler` for num, most-frequent + `OneHotEncoder` for cat) fitted strictly on training fold. |
| **M2: 3 Classifiers Evaluated Side-by-Side** | `analytics/run_analytics.py:312-358` | **PASS** | Logistic Regression, Decision Tree, Random Forest evaluated with CM, Accuracy, Precision, Recall, F1, ROC, AUC. |
| **M2: Decision Tree plot_tree** | `analytics/plots/decision_tree_plot.png` | **PASS** | Tree structure rendered with feature and class names (`Not Survived`, `Survived`). |
| **M2: Multi-Model ROC Curves** | `analytics/plots/roc_curves.png` | **PASS** | ROC curves for all three classifiers rendered and saved. |
| **M2: Class Imbalance Comparison** | `analytics/run_analytics.py:382-416` | **PASS** | Evaluates Baseline vs `class_weight='balanced'` vs SMOTE. Minority recall lifted from 66.7% to 78.3%. |
| **M2: Training-Only SMOTE** | `analytics/run_analytics.py:397-408` | **PASS** | SMOTE fitted and applied strictly to `X_train_trans`. Test fold completely untouched. |
| **M2: Random Forest GridSearchCV & OOB** | `analytics/run_analytics.py:418-454` | **PASS** | Tuned over `n_estimators`, `max_depth`, `max_features` with `oob_score=True`. OOB Score = 0.8188. |
| **M2: Fare Linear Regression** | `analytics/run_analytics.py:456-486` | **PASS** | Multivariate regression predicting fare: MAE = £20.81, RMSE = £30.47, $R^2$ = 0.3999, Adj $R^2$ = 0.3790. |
| **M2: Residual Plot & Heteroscedasticity** | `analytics/plots/regression_residuals.png` | **PASS** | Residual scatter exhibits fan shape confirming heteroscedasticity due to first-class fare variance. |
| **M2: Separate Model Comparison Groups** | `analytics/run_analytics.py:504-522` | **PASS** | Maintained as two distinct groups: Group 1 (Classification) vs Group 2 (Regression). |
| **M2: Deployment Recommendation** | `analytics/run_analytics.py:524-533` | **PASS** | 3-5 sentence recommendation referencing actual metrics (Accuracy 79.9%, F1 0.710, AUC 0.846, OOB 0.819). |
| **M2: Joblib Pipeline Persistence** | `analytics/best_titanic_pipeline.joblib` | **PASS** | Complete fitted pipeline (preprocessor + estimator) serialized via `joblib.dump`. |
| **M2: Reload & Raw Input Prediction** | `analytics/run_analytics.py:540-560` | **PASS** | Reloaded via `joblib.load()` and verified to predict raw DataFrames containing missing values correctly. |
| **M2: Automated Test Suite** | `analytics/test_analytics.py` | **PASS** | Unit tests verify all acceptance criteria; exits with `ALL ANALYTICS TESTS PASSED!`. |
| **M3: 8 Canonical Policy Documents** | `support_assistant/docs/doc_01..08.txt` | **PASS** | Exactly 8 text files covering delivery, returns, membership, tracking, cancellations, damages, gift cards, and support hours. |
| **M3: Local Dense Embeddings** | `support_assistant/ingest.py:16, 21-28` | **PASS** | Uses `sentence-transformers/all-MiniLM-L6-v2` locally without external API dependencies. |
| **M3: Persistent ChromaDB Storage** | `support_assistant/chroma_db/` | **PASS** | Collection `zepto_policies` indexed with cosine distance (`space: cosine`). |
| **M3: Structured Prompt Template** | `support_assistant/rag_graph.py:38-63` | **PASS** | Contains all 5 elements (role, context, task, format, length), negative constraint, and few-shot example. |
| **M3: LangGraph StateGraph Architecture** | `support_assistant/rag_graph.py:186-218` | **PASS** | Implemented using `StateGraph(AgentState)` with `TypedDict` state. |
| **M3: 3 Named Nodes** | `support_assistant/rag_graph.py:190-192` | **PASS** | Exactly named: `classify_intent`, `retrieve_and_answer`, `direct_answer`. |
| **M3: Keyword Intent Classification** | `support_assistant/rag_graph.py:90-109` | **PASS** | Lowercases query and checks for 8 policy keywords (`delivery`, `return`, `refund`, etc.). |
| **M3: Conditional Routing** | `support_assistant/rag_graph.py:180-184` | **PASS** | Conditional edge routes `policy_question` $\to$ retrieval and `general_question` $\to$ direct answer. Does not branch on `MOCK_LLM`. |
| **M3: Top-3 Cosine Similarity Retrieval** | `support_assistant/rag_graph.py:111-137` | **PASS** | Retrieves top-3 matching chunks via cosine similarity in both mock and real modes. |
| **M3: Mock Policy Answer Format** | `support_assistant/rag_graph.py:138-145` | **PASS** | Exact template: `Based on the retrieved context: {top_chunk_snippet}` (~200 chars). |
| **M3: Mock General Answer Format** | `support_assistant/rag_graph.py:157-172` | **PASS** | Exact canned response: `I can only answer questions about Zepto policies right now.` with `sources=[]` and `confidence=1.0`. |
| **M3: Pydantic Response Validation** | `support_assistant/rag_graph.py:78-82` | **PASS** | Schema: `answer: str`, `sources: List[str]`, `confidence: float` validated strictly in `[0.0, 1.0]`. |
| **M3: Real LLM Mode & Retries** | `support_assistant/rag_graph.py:174-188` | **PASS** | `MOCK_LLM=0` activates generation with up to 2 validation retries upon schema failure. |
| **M3: FastAPI Service** | `support_assistant/main.py:12-45` | **PASS** | Serves POST `/ask` and GET `/health` returning validated Pydantic model. |
| **M3: Containerization Manifest** | `support_assistant/Dockerfile` | **PASS** | Production Dockerfile builds and exposes port 7860 serving FastAPI via Uvicorn. |
| **M3: Automated Test Suite** | `support_assistant/test_support_assistant.py` | **PASS** | Unit tests verify all criteria; exits with `ALL SUPPORT ASSISTANT TESTS PASSED!`. |
| **GIT: Feature Branch Workflow** | `feat/capstone-modules` | **PASS** | Feature branch created, multiple meaningful commits staged and committed, merged into `main`. |
