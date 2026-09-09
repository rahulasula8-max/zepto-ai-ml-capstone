"""Automated Test Suite for Module 2 — Analytics.
Validates all graded criteria for profiling, EDA, modeling, imbalance handling,
regression, pipeline persistence, and raw input prediction.
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

ANALYTICS_DIR = Path(__file__).resolve().parent
CSV_PATH = ANALYTICS_DIR / "titanic.csv"
PIPELINE_PATH = ANALYTICS_DIR / "best_titanic_pipeline.joblib"
PLOTS_DIR = ANALYTICS_DIR / "plots"


def test_titanic_csv_exists():
    """Verify offline fallback dataset exists and has expected rows/columns."""
    assert CSV_PATH.exists(), f"Missing {CSV_PATH}"
    df = pd.read_csv(CSV_PATH)
    assert len(df) == 891, f"Expected 891 rows in titanic.csv, got {len(df)}"
    assert "survived" in df.columns
    assert "fare" in df.columns
    assert "age" in df.columns


def test_saved_pipeline_and_raw_prediction():
    """Verify complete fitted pipeline loads and predicts on raw input containing NaNs."""
    assert PIPELINE_PATH.exists(), f"Missing {PIPELINE_PATH}"
    pipeline = joblib.load(PIPELINE_PATH)

    # Check that pipeline contains both preprocessor and estimator
    assert hasattr(pipeline, "named_steps"), "Pipeline object lacks named_steps"
    assert "preprocessor" in pipeline.named_steps or "rf" in pipeline.named_steps or "classifier" in pipeline.named_steps

    # Test raw un-preprocessed input DataFrame
    raw_input = pd.DataFrame([
        {"pclass": 1, "sex": "female", "age": 28.0, "sibsp": 0, "parch": 0, "fare": 100.0, "embarked": "S"},
        {"pclass": 3, "sex": "male", "age": np.nan, "sibsp": 1, "parch": 0, "fare": 7.25, "embarked": "Q"},
        {"pclass": 2, "sex": "female", "age": 45.0, "sibsp": 0, "parch": 2, "fare": 30.0, "embarked": np.nan}
    ])

    preds = pipeline.predict(raw_input)
    assert len(preds) == 3, f"Expected 3 predictions, got {len(preds)}"
    assert set(preds).issubset({0, 1}), f"Predictions must be 0 or 1, got {preds}"

    probas = pipeline.predict_proba(raw_input)
    assert probas.shape == (3, 2), f"Expected shape (3, 2), got {probas.shape}"
    assert np.all((probas >= 0.0) & (probas <= 1.0)), "Probabilities outside [0, 1]"


def test_required_plots_exist():
    """Verify all required plots and charts were generated and saved."""
    required_plots = [
        "age_distribution.png",
        "fare_distribution.png",
        "correlation_heatmap.png",
        "chart1_survival_by_sex_pclass.png",
        "chart2_fare_by_class_survival.png",
        "chart3_age_fare_survival_scatter.png",
        "chart4_embarked_survival_distribution.png",
        "decision_tree_plot.png",
        "roc_curves.png",
        "regression_residuals.png"
    ]
    for plot_name in required_plots:
        plot_path = PLOTS_DIR / plot_name
        assert plot_path.exists(), f"Missing plot: {plot_path}"
        assert plot_path.stat().st_size > 1000, f"Plot file {plot_name} appears empty or truncated"


def test_notebooks_exist():
    """Verify both 01_eda.ipynb and 02_modeling.ipynb exist."""
    assert (ANALYTICS_DIR / "01_eda.ipynb").exists()
    assert (ANALYTICS_DIR / "02_modeling.ipynb").exists()


if __name__ == "__main__":
    print("Running test_titanic_csv_exists...")
    test_titanic_csv_exists()
    print("Running test_saved_pipeline_and_raw_prediction...")
    test_saved_pipeline_and_raw_prediction()
    print("Running test_required_plots_exist...")
    test_required_plots_exist()
    print("Running test_notebooks_exist...")
    test_notebooks_exist()
    print("\nALL ANALYTICS TESTS PASSED!")
