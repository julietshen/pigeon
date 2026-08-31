from __future__ import annotations

from typing import Awaitable, Callable

from fastapi import Header, HTTPException

from .settings import Settings


def make_auth(settings: Settings) -> Callable[[str], Awaitable[str]]:
    """Dependency that maps a bearer token to an org id, or 401s."""

    async def get_org(authorization: str = Header(default="")) -> str:
        token = authorization.removeprefix("Bearer ").strip()
        org = settings.tokens.get(token)
        if not org:
            raise HTTPException(status_code=401, detail="invalid or missing bearer token")
        return org

    return get_org
