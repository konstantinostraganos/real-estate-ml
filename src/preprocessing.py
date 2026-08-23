"""
Preprocessing pipeline for the Real Estate ML project.

Builds a scikit-learn ColumnTransformer that handles imputation,
encoding, and scaling in a single reproducible object.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_FEATURES,
    CORRELATION_THRESHOLD,
    NUMERICAL_FEATURES,
    OUTLIER_FEATURES,
    OUTLIER_IQR_MULTIPLIER,
)


def _get_available_features(
    df: pd.DataFrame,
    feature_list: list[str],
) -> list[str]:
    """Return only features that actually exist in the dataframe."""
    return [f for f in feature_list if f in df.columns]


def build_preprocessing_pipeline(
    df: pd.DataFrame,
) -> tuple[ColumnTransformer, list[str], list[str]]:
    """
    Build a ColumnTransformer that handles numerical and categorical
    features in parallel.

    Returns
    -------
    preprocessor : Unfitted ColumnTransformer.
    num_features : List of numerical feature names used.
    cat_features : List of categorical feature names used.
    """
    num_features = _get_available_features(df, NUMERICAL_FEATURES)
    cat_features = _get_available_features(df, CATEGORICAL_FEATURES)

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, num_features),
            ("cat", categorical_pipeline, cat_features),
        ],
        remainder="drop",
    )

    return preprocessor, num_features, cat_features


def cap_outliers(
    df: pd.DataFrame,
    fit: bool = True,
    bounds: dict[str, tuple[float, float]] | None = None,
) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    """
    Cap outliers using the IQR method on selected features.
    """
    df = df.copy()
    outlier_cols = [c for c in OUTLIER_FEATURES if c in df.columns]

    if fit:
        bounds = {}
        for col in outlier_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - OUTLIER_IQR_MULTIPLIER * iqr
            upper = q3 + OUTLIER_IQR_MULTIPLIER * iqr
            bounds[col] = (lower, upper)

    for col in outlier_cols:
        if col in bounds:
            lower, upper = bounds[col]
            df[col] = df[col].clip(lower, upper)

    return df, bounds


def drop_correlated_features(
    df: pd.DataFrame,
    fit: bool = True,
    cols_to_drop: list[str] | None = None,
    threshold: float = CORRELATION_THRESHOLD,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Remove one of each pair of highly correlated features.
    """
    if fit:
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        cols_to_drop = [
            col for col in upper.columns if any(upper[col] > threshold)
        ]

    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    return df, cols_to_drop