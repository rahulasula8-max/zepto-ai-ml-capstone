# Module 2 — Analytics: Titanic Exploratory Analysis & Predictive Modeling

## Overview
This module delivers a cohesive, leak-free machine learning workflow analyzing passenger survival patterns and fare dynamics on the RMS Titanic. It transitions from initial profiling and statistical validation to supervised classification, class imbalance mitigation, hyperparameter optimization, multivariate regression, and deployment pipeline serialization.

## Key Architectural Principles & Compliance
1. **Strict Offline Data Rule**:
   - `sns.load_dataset('titanic')` was executed **exactly once** during project initialization.
   - The dataset was immediately serialized to `titanic.csv`.
   - All subsequent analytics scripts, notebooks, and models load exclusively from `titanic.csv`.

2. **Missing-Value Strategy**:
   - **<5% Missing (`embarked`, `embark_town` at 0.22%)**: Dropped 2 rows for exploratory profiling; handled via most frequent imputer in production pipeline.
   - **5%–30% Missing (`age` at 19.87%)**: 177 missing entries imputed using training-fold median (28.0 years) to remain robust against skewness.
   - **>30% Missing (`deck` at 77.22%)**: 688 missing entries. Dropped due to severe sparsity that would otherwise introduce excessive synthetic bias.

3. **Univariate Analysis & Outliers**:
   - **Age**: IQR = 17.88 years (Q1 = 20.12, Q3 = 38.00). Outlier bounds: [-6.69, 64.81]. Detected 11 upper outliers (>64.8 years).
   - **Fare**: IQR = £23.09 (Q1 = £7.91, Q3 = £31.00). Outlier bounds: [-£26.72, £65.63]. Detected 116 high-fare outliers (13.02%).
   - **Skewness Conclusion**: Mean (£32.20) > Median (£14.45) > Mode (£8.05). Since Mean > Median > Mode, the fare distribution is heavily **right-skewed** (positive skewness).

4. **Bivariate Analysis (Boolean Masking)**:
   - Survival by Gender: Female = **74.20%**, Male = **18.89%**.
   - Survival by Class: Class 1 = **62.96%**, Class 2 = **47.28%**, Class 3 = **24.24%**.
   - Survival by Gender & Class:
     - Female Class 1: **96.81%** | Male Class 1: **36.89%**
     - Female Class 2: **92.11%** | Male Class 2: **15.74%**
     - Female Class 3: **50.00%** | Male Class 3: **13.54%**

5. **Exact 6-Column Correlation Matrix**:
   - Columns: `survived`, `pclass`, `age`, `sibsp`, `parch`, `fare`.
   - Excluded: `adult_male` and `alone`.
   - **Top 2 Off-Diagonal Correlations**:
     1. `pclass` $\leftrightarrow$ `fare` ($r = -0.5495$): Strong inverse relationship reflecting steep ticket price tiers.
     2. `sibsp` $\leftrightarrow$ `parch` ($r = +0.4148$): Moderate positive correlation reflecting family travel cohorts.

6. **EDA Standardization Sanity Check**:
   - Standardized `age` and `fare` using z-score: $\mu \approx 0.0000, \sigma \approx 1.0000$.
   - **Isolated**: Strictly documented as an exploratory sanity check; **never** fed into modeling.

7. **Stratified Split & Leak-Free Preprocessing**:
   - Stratified train/test split (80/20, `stratify=y`) performed **before** any preprocessing.
   - `ColumnTransformer` with `SimpleImputer`, `StandardScaler`, and `OneHotEncoder` fitted **strictly on the training fold**.

8. **Classification Models & Evaluations**:
   - Models: Logistic Regression, Decision Tree (`max_depth=4`), Random Forest (`oob_score=True`).
   - Decision tree structure plotted with feature and class names (`plots/decision_tree_plot.png`).
   - Evaluated using Confusion Matrix, Accuracy, Precision, Recall, F1, ROC curves (`plots/roc_curves.png`), and AUC.

9. **Class Imbalance Comparison**:
   - Target balance: Perished = 61.62%, Survived = 38.38%.
   - Strategies tested on Logistic Regression:
     1. Baseline: Prec = 0.7931, Rec = 0.6667, F1 = 0.7244
     2. `class_weight='balanced'`: Prec = 0.7297, Rec = 0.7826, F1 = 0.7552
     3. SMOTE (Training-Only): Prec = 0.7397, Rec = 0.7826, **F1 = 0.7606**
   - Conclusion: Applying SMOTE strictly on the training fold lifted recall by 11.6 percentage points without corrupting test distribution.

10. **Random Forest GridSearchCV**:
    - Parameters tuned: `n_estimators` [50, 100, 150], `max_depth` [4, 6, 8], `max_features` ['sqrt', 'log2'].
    - Best params: `{'max_depth': 8, 'max_features': 'log2', 'n_estimators': 50}`.
    - Out-of-Bag (OOB) Score: **0.8188**. Test AUC: **0.8458**.

11. **Regression Side Task (Predicting Fare)**:
    - Multivariate Linear Regression predicting `fare` from passenger features.
    - Metrics: MAE = **£20.81**, RMSE = **£30.47**, $R^2$ = **0.3999**, Adjusted $R^2$ = **0.3790**.
    - Residual plot (`plots/regression_residuals.png`) demonstrates pronounced **heteroscedasticity** (funnel shape) driven by extreme first-class fare outliers.

12. **Model Comparison Tables**:
    - Kept in two distinct groups (Classification vs Regression).

13. **Final Deployment Recommendation**:
    > For operational deployment, the Tuned Random Forest Classifier is strongly recommended, achieving the highest overall test accuracy of 79.9%, an F1-score of 0.710, and an outstanding ROC-AUC of 0.846. While Logistic Regression attained acceptable baseline recall (0.667), Random Forest's non-linear ensemble architecture significantly reduces false positives, outperforming the single Decision Tree by 2.5 AUC percentage points. Furthermore, its out-of-bag validation score of 0.819 demonstrates superior generalization resilience against overfitting on unseen passenger cohorts.

14. **Full Pipeline Persistence & Raw Verification**:
    - Entire fitted pipeline (preprocessing + estimator) saved to `best_titanic_pipeline.joblib`.
    - Reloaded via `joblib.load()` and verified on raw input records containing missing values.

## Execution Commands
From the repository root:
```bash
# 1. Run complete analytics and modeling workflow (generates metrics, plots, and saved pipeline)
python analytics/run_analytics.py

# 2. Run automated test suite
python analytics/test_analytics.py
```

## Directory Structure
- `titanic.csv`: Offline fallback dataset.
- `01_eda.ipynb`: Jupyter notebook for exploratory data analysis.
- `02_modeling.ipynb`: Jupyter notebook for ML classification and regression.
- `run_analytics.py`: Standalone reproducible Python script.
- `test_analytics.py`: Automated validation test suite.
- `best_titanic_pipeline.joblib`: Serialized end-to-end best ML pipeline.
- `plots/`: Directory containing all generated visualizations:
  - `age_distribution.png`
  - `fare_distribution.png`
  - `correlation_heatmap.png`
  - `chart1_survival_by_sex_pclass.png`
  - `chart2_fare_by_class_survival.png`
  - `chart3_age_fare_survival_scatter.png`
  - `chart4_embarked_survival_distribution.png`
  - `decision_tree_plot.png`
  - `roc_curves.png`
  - `regression_residuals.png`
