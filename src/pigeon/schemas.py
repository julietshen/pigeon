from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel

# ---- Discover (GET /v1/modelfiles) ----


class LabelSpec(BaseModel):
    id: str
    display: str
    valueType: Literal["score"] = "score"


class ModelfileSummary(BaseModel):
    id: str
    version: str
    kind: Literal["classifier", "byop", "completion"]
    base: Optional[str] = None
    inputTypes: list[str]
    labels: list[LabelSpec]


class ModelfilesResponse(BaseModel):
    modelfiles: list[ModelfileSummary]


# ---- Classify (POST /v1/classify) ----


class ClassifyInput(BaseModel):
    text: Optional[str] = None
    mediaUrl: Optional[str] = None


class ClassifyRequest(BaseModel):
    model: str
    input: ClassifyInput
    policy: Optional[str] = None


class ClassifyResult(BaseModel):
    label: str
    score: float


class ClassifyResponse(BaseModel):
    model: str
    results: list[ClassifyResult]


# ---- Policy management (BYOP custom models) ----


class PolicyCreate(BaseModel):
    name: str
    base: str  # base modelfile name; must be policy-steerable (BYOP)
    policyText: str
    display: Optional[str] = None


class PolicyResponse(BaseModel):
    id: str  # the custom model id (== name)
    version: str
    base: str
    display: str
