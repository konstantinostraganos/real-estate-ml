"""
Tests for the FastAPI prediction endpoint.
"""

from fastapi.testclient import TestClient

from api.app import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_predict_mid_range():
    data = {
        "LotArea": 8450,
        "OverallQual": 7,
        "OverallCond": 5,
        "YearBuilt": 2003,
        "YearRemodAdd": 2003,
        "1stFlrSF": 856,
        "2ndFlrSF": 854,
        "GrLivArea": 1710,
        "FullBath": 2,
        "HalfBath": 1,
        "TotalBsmtSF": 856,
        "GarageCars": 2,
        "GarageArea": 548,
    }
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    result = response.json()
    assert result["property_tier"] in ["Budget", "Mid-Range", "Premium"]
    assert result["predicted_price"] > 0


def test_predict_premium():
    data = {
        "LotArea": 12000,
        "OverallQual": 10,
        "OverallCond": 5,
        "YearBuilt": 2020,
        "YearRemodAdd": 2020,
        "1stFlrSF": 1500,
        "2ndFlrSF": 1200,
        "GrLivArea": 2700,
        "FullBath": 3,
        "HalfBath": 1,
        "TotalBsmtSF": 1500,
        "GarageCars": 3,
        "GarageArea": 900,
    }
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    result = response.json()
    assert result["property_tier"] == "Premium"
    assert result["predicted_price"] > 250000


def test_predict_budget():
    data = {
        "LotArea": 6000,
        "OverallQual": 4,
        "OverallCond": 6,
        "YearBuilt": 1965,
        "YearRemodAdd": 1965,
        "1stFlrSF": 800,
        "2ndFlrSF": 0,
        "GrLivArea": 800,
        "FullBath": 1,
        "HalfBath": 0,
        "TotalBsmtSF": 800,
        "GarageCars": 1,
        "GarageArea": 300,
    }
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    result = response.json()
    assert result["property_tier"] == "Budget"
    assert result["predicted_price"] < 150000