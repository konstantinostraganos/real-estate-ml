"""
Tests for feature engineering module.
"""

import pandas as pd
import pytest

from src.feature_engineering import (
    create_house_age,
    create_quality_score,
    create_remodel_flag,
    create_total_bathrooms,
    create_total_porch_sf,
    create_total_sf,
    run_feature_engineering,
)


def make_sample_row():
    """Create a minimal sample row for testing."""
    return pd.DataFrame([{
        "Id": 1,
        "SalePrice": 200000,
        "YrSold": 2010,
        "YearBuilt": 2000,
        "YearRemodAdd": 2005,
        "MoSold": 6,
        "TotalBsmtSF": 800,
        "1stFlrSF": 1000,
        "2ndFlrSF": 500,
        "FullBath": 2,
        "HalfBath": 1,
        "BsmtFullBath": 1,
        "BsmtHalfBath": 0,
        "WoodDeckSF": 100,
        "OpenPorchSF": 50,
        "EnclosedPorch": 0,
        "3SsnPorch": 0,
        "ScreenPorch": 0,
        "OverallQual": 7,
        "OverallCond": 5,
        "GrLivArea": 1500,
        "LotArea": 8000,
        "LotFrontage": 60,
        "GarageCars": 2,
        "GarageArea": 500,
        "PoolQC": None,
        "MiscFeature": None,
        "Alley": None,
        "Fence": None,
        "MasVnrArea": 0,
        "BsmtFinSF1": 0,
        "BsmtFinSF2": 0,
        "BsmtUnfSF": 800,
        "LowQualFinSF": 0,
        "BedroomAbvGr": 3,
        "KitchenAbvGr": 1,
        "TotRmsAbvGrd": 7,
        "Fireplaces": 1,
        "MiscVal": 0,
        "MSZoning": "RL",
        "Street": "Pave",
        "LotShape": "Reg",
        "LandContour": "Lvl",
        "Utilities": "AllPub",
        "LotConfig": "Inside",
        "LandSlope": "Gtl",
        "Neighborhood": "NAmes",
        "Condition1": "Norm",
        "Condition2": "Norm",
        "BldgType": "1Fam",
        "HouseStyle": "1Story",
        "RoofStyle": "Gable",
        "RoofMatl": "CompShg",
        "Exterior1st": "VinylSd",
        "Exterior2nd": "VinylSd",
        "MasVnrType": "None",
        "ExterQual": "TA",
        "ExterCond": "TA",
        "Foundation": "PConc",
        "BsmtQual": "TA",
        "BsmtCond": "TA",
        "BsmtExposure": "No",
        "BsmtFinType1": "Unf",
        "BsmtFinType2": "Unf",
        "Heating": "GasA",
        "HeatingQC": "Ex",
        "CentralAir": "Y",
        "Electrical": "SBrkr",
        "KitchenQual": "TA",
        "Functional": "Typ",
        "FireplaceQu": "Gd",
        "GarageType": "Attchd",
        "GarageFinish": "RFn",
        "GarageQual": "TA",
        "GarageCond": "TA",
        "PavedDrive": "Y",
        "SaleType": "WD",
        "SaleCondition": "Normal",
    }])


def test_house_age():
    df = make_sample_row()
    result = create_house_age(df)
    assert result["HouseAge"].iloc[0] == 10  # 2010 - 2000


def test_remodel_flag():
    df = make_sample_row()
    result = create_remodel_flag(df)
    assert result["HasRemodel"].iloc[0] == 1  # 2005 != 2000


def test_remodel_flag_no_remodel():
    df = make_sample_row()
    df["YearRemodAdd"] = 2000  # same as YearBuilt
    result = create_remodel_flag(df)
    assert result["HasRemodel"].iloc[0] == 0


def test_total_sf():
    df = make_sample_row()
    result = create_total_sf(df)
    assert result["TotalSF"].iloc[0] == 2300  # 800 + 1000 + 500


def test_total_bathrooms():
    df = make_sample_row()
    result = create_total_bathrooms(df)
    assert result["TotalBathrooms"].iloc[0] == 3.5  # 2 + 0.5 + 1 + 0


def test_total_porch_sf():
    df = make_sample_row()
    result = create_total_porch_sf(df)
    assert result["TotalPorchSF"].iloc[0] == 150  # 100 + 50


def test_quality_score():
    df = make_sample_row()
    result = create_quality_score(df)
    assert result["QualityScore"].iloc[0] == 35  # 7 * 5


def test_run_feature_engineering():
    # Need 3+ rows for 3-tier classification
    df = pd.concat([make_sample_row()] * 3, ignore_index=True)
    df["SalePrice"] = [100000, 200000, 350000]  # one per tier
    result, boundaries = run_feature_engineering(df)
    # Check new features exist
    assert "HouseAge" in result.columns
    assert "TotalSF" in result.columns
    assert "PropertyTier" in result.columns
    # Check dropped columns are gone
    assert "PoolQC" not in result.columns
    assert "Id" not in result.columns
    assert "YearBuilt" not in result.columns