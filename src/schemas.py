from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    hour: int = Field(ge=0, le=23)
    day_of_week: int = Field(ge=0, le=6)
    month: int = Field(ge=1, le=12)
    demand_lag_1: float = Field(gt=0)
    demand_lag_48: float = Field(gt=0)
    demand_rolling_mean_48: float = Field(gt=0)
    temp_proxy: float = 0.0


class PredictionResponse(BaseModel):
    predicted_demand_mw: float
    model_version: str
