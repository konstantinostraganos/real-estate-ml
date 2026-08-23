"""
Central configuration for the Real Estate ML project.
All paths, constants, and hyperparameter grids in one place.
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

RAW_DATA_FILE = DATA_RAW_DIR / "train.csv"

# ── Random state ──────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2

# ── Target configuration ─────────────────────────────────────
REGRESSION_TARGET = "SalePrice"

# 3-tier classification instead of binary (group project had Standard/Premium)
CLASSIFICATION_TARGET = "PropertyTier"
PROPERTY_TIERS = {
    "labels": ["Budget", "Mid-Range", "Premium"],
    "quantiles": [0.0, 1 / 3, 2 / 3, 1.0],
}

# ── Columns to drop (>40% missing or no value) ───────────────
COLUMNS_TO_DROP = ["PoolQC", "MiscFeature", "Alley", "Fence", "Id"]

# ── Numerical features ───────────────────────────────────────
NUMERICAL_FEATURES = [
    "LotFrontage", "LotArea", "MasVnrArea",
    "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
    "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea",
    "BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath",
    "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd",
    "Fireplaces", "GarageCars", "GarageArea",
    "WoodDeckSF", "OpenPorchSF", "EnclosedPorch",
    "3SsnPorch", "ScreenPorch", "MiscVal",
    "OverallQual", "OverallCond",
]

# ── Categorical features ─────────────────────────────────────
CATEGORICAL_FEATURES = [
    "MSZoning", "Street", "LotShape", "LandContour", "Utilities",
    "LotConfig", "LandSlope", "Neighborhood", "Condition1", "Condition2",
    "BldgType", "HouseStyle", "RoofStyle", "RoofMatl",
    "Exterior1st", "Exterior2nd", "MasVnrType", "ExterQual", "ExterCond",
    "Foundation", "BsmtQual", "BsmtCond", "BsmtExposure",
    "BsmtFinType1", "BsmtFinType2", "Heating", "HeatingQC",
    "CentralAir", "Electrical", "KitchenQual",
    "Functional", "FireplaceQu", "GarageType", "GarageFinish",
    "GarageQual", "GarageCond", "PavedDrive",
    "SaleType", "SaleCondition",
]

# ── Outlier handling ──────────────────────────────────────────
OUTLIER_FEATURES = ["GrLivArea", "GarageArea", "TotalBsmtSF", "LotArea"]
OUTLIER_IQR_MULTIPLIER = 1.5

# ── Correlation filtering ────────────────────────────────────
CORRELATION_THRESHOLD = 0.90

# ── Hyperparameter grids ─────────────────────────────────────
RF_CLASSIFIER_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
}

XGBOOST_CLASSIFIER_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}

ELASTICNET_GRID = {
    "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
    "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
}

XGBOOST_REGRESSOR_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}

# ── MLflow ────────────────────────────────────────────────────
MLFLOW_EXPERIMENT_NAME = "real-estate-ml"
MLFLOW_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"