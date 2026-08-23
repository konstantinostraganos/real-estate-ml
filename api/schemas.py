"""
Request and response schemas for the prediction API.

Pydantic models validate incoming data automatically —
if someone sends a string where we expect a number,
FastAPI returns a clear error without our code crashing.
"""

from pydantic import BaseModel, Field


class PropertyInput(BaseModel):
    """What the API expects — property characteristics."""
    LotFrontage: float = Field(default=0, description="Street frontage in feet")
    LotArea: float = Field(description="Lot size in square feet")
    OverallQual: int = Field(description="Overall quality (1-10)")
    OverallCond: int = Field(description="Overall condition (1-10)")
    YearBuilt: int = Field(description="Year built")
    YearRemodAdd: int = Field(description="Year remodelled")
    YrSold: int = Field(default=2010, description="Year sold")
    MasVnrArea: float = Field(default=0)
    BsmtFinSF1: float = Field(default=0)
    BsmtFinSF2: float = Field(default=0)
    BsmtUnfSF: float = Field(default=0)
    TotalBsmtSF: float = Field(default=0)
    FirstFlrSF: float = Field(alias="1stFlrSF", description="First floor sqft")
    SecondFlrSF: float = Field(default=0, alias="2ndFlrSF")
    LowQualFinSF: float = Field(default=0)
    GrLivArea: float = Field(description="Above ground living area")
    BsmtFullBath: float = Field(default=0)
    BsmtHalfBath: float = Field(default=0)
    FullBath: int = Field(description="Full bathrooms")
    HalfBath: int = Field(default=0)
    BedroomAbvGr: int = Field(default=3)
    KitchenAbvGr: int = Field(default=1)
    TotRmsAbvGrd: int = Field(default=6)
    Fireplaces: int = Field(default=0)
    GarageCars: int = Field(default=0)
    GarageArea: float = Field(default=0)
    WoodDeckSF: float = Field(default=0)
    OpenPorchSF: float = Field(default=0)
    EnclosedPorch: float = Field(default=0)
    ThreeSsnPorch: float = Field(default=0, alias="3SsnPorch")
    ScreenPorch: float = Field(default=0)
    MiscVal: float = Field(default=0)
    MSZoning: str = Field(default="RL")
    Street: str = Field(default="Pave")
    LotShape: str = Field(default="Reg")
    LandContour: str = Field(default="Lvl")
    Utilities: str = Field(default="AllPub")
    LotConfig: str = Field(default="Inside")
    LandSlope: str = Field(default="Gtl")
    Neighborhood: str = Field(default="NAmes")
    Condition1: str = Field(default="Norm")
    Condition2: str = Field(default="Norm")
    BldgType: str = Field(default="1Fam")
    HouseStyle: str = Field(default="1Story")
    RoofStyle: str = Field(default="Gable")
    RoofMatl: str = Field(default="CompShg")
    Exterior1st: str = Field(default="VinylSd")
    Exterior2nd: str = Field(default="VinylSd")
    MasVnrType: str = Field(default="None")
    ExterQual: str = Field(default="TA")
    ExterCond: str = Field(default="TA")
    Foundation: str = Field(default="PConc")
    BsmtQual: str = Field(default="TA")
    BsmtCond: str = Field(default="TA")
    BsmtExposure: str = Field(default="No")
    BsmtFinType1: str = Field(default="Unf")
    BsmtFinType2: str = Field(default="Unf")
    Heating: str = Field(default="GasA")
    HeatingQC: str = Field(default="Ex")
    CentralAir: str = Field(default="Y")
    Electrical: str = Field(default="SBrkr")
    KitchenQual: str = Field(default="TA")
    Functional: str = Field(default="Typ")
    FireplaceQu: str = Field(default="Gd")
    GarageType: str = Field(default="Attchd")
    GarageFinish: str = Field(default="RFn")
    GarageQual: str = Field(default="TA")
    GarageCond: str = Field(default="TA")
    PavedDrive: str = Field(default="Y")
    SaleType: str = Field(default="WD")
    SaleCondition: str = Field(default="Normal")

    model_config = {"populate_by_name": True}


class PredictionOutput(BaseModel):
    """What the API returns."""
    property_tier: str = Field(description="Budget, Mid-Range, or Premium")
    predicted_price: float = Field(description="Estimated sale price in USD")