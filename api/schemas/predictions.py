"""Response of the prediction endpoint."""

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    category: str = Field(examples=["bottle"])
    filename: str | None = Field(examples=["image.png"])
    anomaly_score: float = Field(examples=[12.5])
    threshold: float | None = Field(
        description="Null when calibration is absent", examples=[15.0]
    )
    is_anomaly: bool | None = Field(
        description="score > threshold; null without a threshold"
    )
