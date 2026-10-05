"""Responses of the model information endpoints."""

from pydantic import BaseModel, Field


class CategoriesResponse(BaseModel):
    categories: list[str] = Field(examples=[["bottle", "cable"]])


class ModelResponse(BaseModel):
    name: str = Field(examples=["patchcore-mvtec"])
    version: str = Field(examples=["2"])
    model_dir: str = Field(examples=["/models/patchcore-mvtec/v2"])
    categories: int = Field(description="Number of loaded product categories", examples=[15])
