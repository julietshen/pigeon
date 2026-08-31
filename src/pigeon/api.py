from __future__ import annotations

from typing import Awaitable, Callable

from fastapi import APIRouter, Depends, HTTPException

from .classify import classify
from .providers.base import ProviderClient
from .registry import Registry
from .schemas import (
    ClassifyRequest,
    ClassifyResponse,
    ModelSpecsResponse,
    PolicyCreate,
    PolicyResponse,
)

GetOrg = Callable[..., Awaitable[str]]


def build_router(
    registry: Registry, provider: ProviderClient, get_org: GetOrg
) -> APIRouter:
    router = APIRouter()

    @router.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @router.get("/v1/modelspecs", response_model=ModelSpecsResponse)
    async def list_modelspecs(org: str = Depends(get_org)) -> ModelSpecsResponse:
        return ModelSpecsResponse(modelspecs=registry.list_signals(org))

    @router.post("/v1/classify", response_model=ClassifyResponse)
    async def classify_endpoint(
        req: ClassifyRequest, org: str = Depends(get_org)
    ) -> ClassifyResponse:
        try:
            version, results = await classify(
                org_id=org,
                model_ref=req.model,
                text=req.input.text,
                media_url=req.input.mediaUrl,
                policy=req.policy,
                registry=registry,
                provider=provider,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        name = req.model.split("@", 1)[0]
        return ClassifyResponse(model=f"{name}@{version}", results=results)

    @router.get("/v1/policies")
    async def list_policies(org: str = Depends(get_org)) -> dict:
        policies = [
            {
                "id": p["name"],
                "version": str(p["version"]),
                "base": p["base"],
                "display": p["display"],
            }
            for p in registry.store.latest_policies(org)
        ]
        return {"policies": policies}

    @router.post("/v1/policies", response_model=PolicyResponse)
    async def create_policy(
        body: PolicyCreate, org: str = Depends(get_org)
    ) -> PolicyResponse:
        base = registry.modelspecs.get(body.base)
        if base is None or not base.policy_argument:
            raise HTTPException(
                status_code=422,
                detail=f"base '{body.base}' is not a policy-steerable (BYOP) model spec",
            )
        display = body.display or body.name
        version = registry.store.upsert_policy(
            org, body.name, body.base, body.policyText, display
        )
        return PolicyResponse(
            id=body.name, version=str(version), base=body.base, display=display
        )

    return router
