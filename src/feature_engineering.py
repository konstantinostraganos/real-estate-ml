"""
Feature engineering for the Real Estate ML project.

Creates domain-driven features from raw Ames Housing data.
Each feature has a business justification — an investor or appraiser
would recognise these as meaningful property characteristics.

"""

import pandas as pd

from src.config import (
    CLASSIFICATION_TARGET,
    COLUMNS_TO_DROP,
    PROPERTY_TIERS,
    REGRESSION_TARGET,
)


def drop_low_value_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns with excessive missing values or near-zero information."""
    cols_to_drop = [c for c in COLUMNS_TO_DROP if c in df.columns]
    return df.drop(columns=cols_to_drop)


def create_house_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    HouseAge = YrSold - YearBuilt

    Older houses generally sell for less unless they have historical value.
    """
    df = df.copy()
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    return df


def create_remodel_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    HasRemodel = 1 if the house was remodelled after construction, else 0.

    Remodelled properties command higher prices because buyers
    perceive them as updated and move-in ready.
    """
    df = df.copy()
    df["HasRemodel"] = (df["YearRemodAdd"] != df["YearBuilt"]).astype(int)
    return df


def create_total_sf(df: pd.DataFrame) -> pd.DataFrame:
    """
    TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF

    Total livable square footage is the single strongest driver
    of property value.
    """
    df = df.copy()
    df["TotalSF"] = (
        df["TotalBsmtSF"].fillna(0)
        + df["1stFlrSF"].fillna(0)
        + df["2ndFlrSF"].fillna(0)
    )
    return df


def create_total_bathrooms(df: pd.DataFrame) -> pd.DataFrame:
    """
    TotalBathrooms = FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath

    Industry convention counts half-baths as 0.5. More bathrooms
    increase both comfort and resale value.
    """
    df = df.copy()
    df["TotalBathrooms"] = (
        df["FullBath"].fillna(0)
        + 0.5 * df["HalfBath"].fillna(0)
        + df["BsmtFullBath"].fillna(0)
        + 0.5 * df["BsmtHalfBath"].fillna(0)
    )
    return df


def create_total_porch_sf(df: pd.DataFrame) -> pd.DataFrame:
    """
    TotalPorchSF = sum of all porch areas.

    Outdoor living space adds value. Combining individual porch
    measurements simplifies the feature space.
    """
    df = df.copy()
    porch_cols = [
        "WoodDeckSF", "OpenPorchSF", "EnclosedPorch",
        "3SsnPorch", "ScreenPorch",
    ]
    df["TotalPorchSF"] = df[porch_cols].fillna(0).sum(axis=1)
    return df


def create_quality_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    QualityScore = OverallQual * OverallCond

    Interaction feature: a high-quality house in poor condition
    scores lower than one in good condition.
    """
    df = df.copy()
    df["QualityScore"] = df["OverallQual"] * df["OverallCond"]
    return df




def create_classification_target(
    df: pd.DataFrame,
    fit: bool = True,
    quantile_boundaries: list[float] | None = None,
) -> tuple[pd.DataFrame, list[float] | None]:
    """
    Create 3-tier PropertyTier label from SalePrice.

    Parameters
    ----------
    df : DataFrame with SalePrice column.
    fit : If True, compute boundaries from data (training).
          If False, use provided boundaries (inference).
    quantile_boundaries : Pre-computed bin edges (used when fit=False).

    Returns
    -------
    df : DataFrame with new PropertyTier column.
    boundaries : The bin edges, for reuse on test data.
    """
    df = df.copy()

    if fit:
        quantile_boundaries = (
            df[REGRESSION_TARGET]
            .quantile(PROPERTY_TIERS["quantiles"])
            .tolist()
        )
        quantile_boundaries[0] = 0
        quantile_boundaries[-1] = float("inf")

    df[CLASSIFICATION_TARGET] = pd.cut(
        df[REGRESSION_TARGET],
        bins=quantile_boundaries,
        labels=PROPERTY_TIERS["labels"],
        include_lowest=True,
    )
    return df, quantile_boundaries


def drop_raw_columns_after_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop raw columns replaced by engineered features.
    Avoids redundancy in the model.
    """
    cols_to_drop = ["YearBuilt", "YrSold", "YearRemodAdd", "MoSold"]
    cols_present = [c for c in cols_to_drop if c in df.columns]
    return df.drop(columns=cols_present)


def run_feature_engineering(
    df: pd.DataFrame,
    fit: bool = True,
    quantile_boundaries: list[float] | None = None,
) -> tuple[pd.DataFrame, list[float] | None]:
    """
    Full feature engineering pipeline — applies all steps in sequence.
    """
    df = drop_low_value_columns(df)
    df = create_house_age(df)
    df = create_remodel_flag(df)
    df = create_total_sf(df)
    df = create_total_bathrooms(df)
    df = create_total_porch_sf(df)
    df = create_quality_score(df)
    df, quantile_boundaries = create_classification_target(
        df, fit=fit, quantile_boundaries=quantile_boundaries,
    )
    df = drop_raw_columns_after_engineering(df)
    return df, quantile_boundaries