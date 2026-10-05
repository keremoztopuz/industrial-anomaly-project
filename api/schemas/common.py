"""Response shapes shared by several endpoints."""

from pydantic import BaseModel, Field


class StatusResponse(BaseModel):
    status: str = Field(examples=["healthy"])


class ErrorResponse(BaseModel):
    detail: str


class ValidationIssue(BaseModel):
    loc: list[str | int]
    msg: str
    type: str


class ValidationErrorResponse(BaseModel):
    detail: list[ValidationIssue]
