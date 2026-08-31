from __future__ import annotations

from typing import Optional

from fastapi import FastAPI

from .api import build_router
from .auth import make_auth
from .modelfiles import load_modelfiles
from .providers.base import ProviderClient
from .providers.mock import MockProvider
from .registry import Registry
from .settings import Settings
from .store import Store


def _make_provider(settings: Settings) -> ProviderClient:
    if settings.provider == "live":
        from .providers.litellm_provider import LiteLLMProvider

        return LiteLLMProvider()
    return MockProvider()


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or Settings.from_env()
    modelfiles = load_modelfiles(settings.modelfiles_dir)
    store = Store(settings.db_path)
    registry = Registry(modelfiles, store)
    provider = _make_provider(settings)
    get_org = make_auth(settings)

    app = FastAPI(title="Pigeon", version="0.1.0")
    app.include_router(build_router(registry, provider, get_org))
    return app
