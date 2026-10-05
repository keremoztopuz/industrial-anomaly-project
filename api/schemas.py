"""Public response contracts; existing JSON field names stay compatible."""

from pydantic import BaseModel, Field


class StatusResponse(BaseModel):
    status: str = Field(examples=["healthy"])


class CategoriesResponse(BaseModel):
    categories: list[str] = Field(examples=[["bottle", "cable"]])


class ModelResponse(BaseModel):
    name: str = Field(examples=["patchcore-mvtec"])
    version: str = Field(examples=["2"])
    model_dir: str = Field(examples=["/models/patchcore-mvtec/v2"])
    categories: int = Field(description="Number of loaded product categories", examples=[15])


class PredictionResponse(BaseModel):
    category: str = Field(examples=["bottle"])
    filename: str | None = Field(examples=["image.png"])
    anomaly_score: float = Field(examples=[12.5])
    threshold: float | None = Field(description="Null when calibration is absent", examples=[15.0])
    is_anomaly: bool | None = Field(description="score > threshold; null without a threshold")


class ErrorResponse(BaseModel):
    detail: str


class ValidationIssue(BaseModel):
    loc: list[str | int]
    msg: str
    type: str


class ValidationErrorResponse(BaseModel):
    detail: list[ValidationIssue]
