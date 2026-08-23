"""
Model evaluation and explainability for the Real Estate ML project.

Generates classification reports, confusion matrices,
regression metrics, and SHAP explanations with plots.
"""

import logging

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    classification_report,
    mean_absolute_error,
    r2_score,
)

from src.config import MODELS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PLOTS_DIR = MODELS_DIR.parent / "plots"


def get_feature_names(preprocessor):
    """Extract feature names from a fitted ColumnTransformer."""
    names = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == "remainder":
            continue
        if hasattr(transformer, "get_feature_names_out"):
            names.extend(transformer.get_feature_names_out(columns))
        elif hasattr(transformer[-1], "get_feature_names_out"):
            names.extend(transformer[-1].get_feature_names_out(columns))
        else:
            names.extend(columns)
    return names


def evaluate_classifier(model, X_test, y_test, label_encoder):
    """Print classification report and confusion matrix."""
    y_pred = model.predict(X_test)
    target_names = label_encoder.classes_

    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(y_test, y_pred, target_names=target_names))

    print("=== CONFUSION MATRIX ===")
    cm = pd.crosstab(
        pd.Series(label_encoder.inverse_transform(y_test), name="Actual"),
        pd.Series(label_encoder.inverse_transform(y_pred), name="Predicted"),
    )
    print(cm)
    print()


def evaluate_regressor(model, X_test, y_test):
    """Print regression metrics and error analysis."""
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    errors = y_test - y_pred

    print("\n=== REGRESSION METRICS ===")
    print(f"R² Score:           {r2:.3f}")
    print(f"MAE:                ${mae:,.0f}")
    print(f"Mean Error:         ${errors.mean():,.0f}")
    print(f"Median Error:       ${errors.median():,.0f}")
    print(f"Max Overprediction: ${errors.min():,.0f}")
    print(f"Max Underprediction: ${errors.max():,.0f}")
    print()


def explain_classifier(model, X_test, feature_names, label_encoder, top_n=10):
    """SHAP analysis for the classification model with plots."""
    print("\n=== SHAP — CLASSIFICATION ===")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    if shap_values.ndim == 3:
        mean_shap = np.abs(shap_values).mean(axis=(0, 2))
    else:
        mean_shap = np.abs(shap_values).mean(axis=0)

    importance = pd.Series(mean_shap, index=feature_names)
    importance = importance.sort_values(ascending=False).head(top_n)

    print(f"\nTop {top_n} most important features:")
    for i, (feat, val) in enumerate(importance.items(), 1):
        print(f"  {i}. {feat}: {val:.4f}")

    # Bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    importance.sort_values().plot.barh(ax=ax)
    ax.set_title("SHAP — Top Classification Features")
    ax.set_xlabel("Mean |SHAP value|")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "shap_classification.png", dpi=150)
    plt.close(fig)
    print(f"\nPlot saved: plots/shap_classification.png")


def explain_regressor(model, X_test, feature_names, top_n=10):
    """SHAP analysis for the regression model with plots."""
    print("\n=== SHAP — REGRESSION ===")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    mean_shap = np.abs(shap_values).mean(axis=0)
    importance = pd.Series(mean_shap, index=feature_names)
    importance = importance.sort_values(ascending=False).head(top_n)

    print(f"\nTop {top_n} most important features:")
    for i, (feat, val) in enumerate(importance.items(), 1):
        print(f"  {i}. {feat}: {val:.4f}")

    # Bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    importance.sort_values().plot.barh(ax=ax)
    ax.set_title("SHAP — Top Regression Features")
    ax.set_xlabel("Mean |SHAP value| ($)")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "shap_regression.png", dpi=150)
    plt.close(fig)
    print(f"\nPlot saved: plots/shap_regression.png")


def main():
    """Load saved models and run full evaluation."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load models and artifacts
    classifier = joblib.load(MODELS_DIR / "best_classifier.joblib")
    regressor = joblib.load(MODELS_DIR / "best_regressor.joblib")
    label_encoder = joblib.load(MODELS_DIR / "label_encoder.joblib")
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")

    # Get feature names from preprocessor
    feature_names = get_feature_names(preprocessor)

    # Prepare test data (same pipeline as training)
    from src.config import (
        CLASSIFICATION_TARGET,
        RANDOM_STATE,
        RAW_DATA_FILE,
        REGRESSION_TARGET,
        TEST_SIZE,
    )
    from src.feature_engineering import run_feature_engineering
    from src.preprocessing import cap_outliers, drop_correlated_features
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    df = pd.read_csv(RAW_DATA_FILE)
    df_eng, _ = run_feature_engineering(df, fit=True)

    X = df_eng.drop(columns=[CLASSIFICATION_TARGET, REGRESSION_TARGET])
    le = LabelEncoder()
    y_class = pd.Series(le.fit_transform(df_eng[CLASSIFICATION_TARGET]), index=df_eng.index)
    y_reg = df_eng[REGRESSION_TARGET]

    X_train, X_test, y_train_clf, y_test_clf = train_test_split(
        X, y_class, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_class,
    )
    y_test_reg = y_reg.loc[X_test.index]

    outlier_bounds = joblib.load(MODELS_DIR / "outlier_bounds.joblib")
    X_test, _ = cap_outliers(X_test, fit=False, bounds=outlier_bounds)

    corr_cols = joblib.load(MODELS_DIR / "corr_cols.joblib")
    X_test, _ = drop_correlated_features(X_test, fit=False, cols_to_drop=corr_cols)

    X_test_processed = preprocessor.transform(X_test)

    # Run evaluations
    evaluate_classifier(classifier, X_test_processed, y_test_clf, label_encoder)
    evaluate_regressor(regressor, X_test_processed, y_test_reg)

    # SHAP explanations
    explain_classifier(classifier, X_test_processed, feature_names, label_encoder)
    explain_regressor(regressor, X_test_processed, feature_names)

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()