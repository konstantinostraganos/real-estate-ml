"""
Data validation for the Real Estate ML project.

Performs basic schema and range checks on incoming data before
it enters the ML pipeline.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Columns that must be present in the raw data
REQUIRED_COLUMNS = [
    "SalePrice", "GrLivArea", "OverallQual", "TotalBsmtSF",
    "1stFlrSF", "2ndFlrSF", "FullBath", "HalfBath",
    "GarageCars", "GarageArea", "YearBuilt", "YrSold",
    "Neighborhood", "BldgType", "OverallCond",
]

# Sensible ranges for numerical features
RANGE_CHECKS = {
    "SalePrice": (1_000, 2_000_000),
    "GrLivArea": (100, 10_000),
    "OverallQual": (1, 10),
    "OverallCond": (1, 10),
    "YearBuilt": (1800, 2025),
    "YrSold": (1900, 2025),
    "GarageCars": (0, 10),
}


class DataValidationError(Exception):
    """Raised when data fails a critical validation check."""


def validate_columns(df: pd.DataFrame) -> list[str]:
    """Check that all required columns are present."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(
            f"Missing required columns: {missing}"
        )
    logger.info(
        "Column check passed — all %d required columns present.",
        len(REQUIRED_COLUMNS),
    )
    return missing


def validate_no_duplicate_rows(df: pd.DataFrame) -> int:
    """Check for and report duplicate rows."""
    n_dupes = df.duplicated().sum()
    if n_dupes > 0:
        logger.warning("Found %d duplicate rows.", n_dupes)
    else:
        logger.info("No duplicate rows found.")
    return n_dupes


def validate_ranges(df: pd.DataFrame) -> dict[str, int]:
    """Check that numerical features fall within sensible ranges."""
    violations = {}
    for col, (lo, hi) in RANGE_CHECKS.items():
        if col not in df.columns:
            continue
        out_of_range = ((df[col] < lo) | (df[col] > hi)).sum()
        if out_of_range > 0:
            violations[col] = out_of_range
            logger.warning(
                "Column '%s': %d values outside [%s, %s].",
                col, out_of_range, lo, hi,
            )
    if not violations:
        logger.info("Range checks passed.")
    return violations


def validate_missing_values(
    df: pd.DataFrame, threshold: float = 0.40,
) -> list[str]:
    """Report columns with missing values above a threshold."""
    missing_pct = df.isnull().mean()
    high_missing = missing_pct[missing_pct > threshold].sort_values(
        ascending=False,
    )
    if not high_missing.empty:
        for col, pct in high_missing.items():
            logger.warning(
                "Column '%s' has %.1f%% missing values.",
                col, pct * 100,
            )
    else:
        logger.info("No columns exceed %.0f%% missing threshold.", threshold * 100)
    return high_missing.index.tolist()


def run_validation(df: pd.DataFrame) -> dict:
    """Run all validation checks and return a summary report."""
    report = {}
    validate_columns(df)
    report["n_rows"] = len(df)
    report["n_columns"] = len(df.columns)
    report["duplicate_rows"] = validate_no_duplicate_rows(df)
    report["range_violations"] = validate_ranges(df)
    report["high_missing_columns"] = validate_missing_values(df)
    logger.info(
        "Validation complete — %d rows, %d columns.",
        report["n_rows"], report["n_columns"],
    )
    return report