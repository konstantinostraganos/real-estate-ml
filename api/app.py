"""
FastAPI prediction service for the Real Estate ML project.

Loads the trained models and preprocessor, accepts property
features, and returns tier classification + price prediction.
"""

import joblib
import pandas as pd
from fastapi import FastAPI

from api.schemas import PredictionOutput, PropertyInput
from src.config import MODELS_DIR
from src.feature_engineering import (
    create_house_age,
    create_quality_score,
    create_remodel_flag,
    create_total_bathrooms,
    create_total_porch_sf,
    create_total_sf,
    drop_low_value_columns,
    drop_raw_columns_after_engineering,
)
from src.preprocessing import cap_outliers, drop_correlated_features

# Load models once at startup (not on every request)
classifier = joblib.load(MODELS_DIR / "best_classifier.joblib")
regressor = joblib.load(MODELS_DIR / "best_regressor.joblib")
preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
label_encoder = joblib.load(MODELS_DIR / "label_encoder.joblib")
outlier_bounds = joblib.load(MODELS_DIR / "outlier_bounds.joblib")
corr_cols = joblib.load(MODELS_DIR / "corr_cols.joblib")

app = FastAPI(
    title="Real Estate ML API",
    description="Predicts property tier and sale price for Ames, Iowa properties.",
    version="1.0.0",
)


@app.get("/health")
def health_check():
    """Simple health check — confirms the API is running."""
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionOutput)
def predict(property_data: PropertyInput):
    """
    Accept property features and return:
    - property_tier: Budget, Mid-Range, or Premium
    - predicted_price: estimated sale price in USD
    """
    # Convert input to DataFrame (models expect DataFrame format)
    data = property_data.model_dump(by_alias=True)
    df = pd.DataFrame([data])

    # Apply same feature engineering as training
    df = drop_low_value_columns(df)
    df = create_house_age(df)
    df = create_remodel_flag(df)
    df = create_total_sf(df)
    df = create_total_bathrooms(df)
    df = create_total_porch_sf(df)
    df = create_quality_score(df)
    df = drop_raw_columns_after_engineering(df)

    # Apply same preprocessing as training
    df, _ = cap_outliers(df, fit=False, bounds=outlier_bounds)
    df, _ = drop_correlated_features(df, fit=False, cols_to_drop=corr_cols)
    X = preprocessor.transform(df)

    # Predict
    tier_code = classifier.predict(X)[0]
    tier_name = label_encoder.inverse_transform([tier_code])[0]
    price = regressor.predict(X)[0]

    return PredictionOutput(
        property_tier=tier_name,
        predicted_price=round(float(price), 2),
    )