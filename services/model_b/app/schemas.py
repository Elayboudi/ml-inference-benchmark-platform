from pydantic import BaseModel, Field, ConfigDict
from typing import List


class PredictionRequest(BaseModel):
    features: List[float] = Field(..., example=[0.1, 0.5, 1.2])


class PredictionResponse(BaseModel):
    prediction: float
    model_version: str
    inference_time_ms: float

    model_config = ConfigDict(protected_namespaces=())