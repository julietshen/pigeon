from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pigeon.main import create_app
from pigeon.settings import Settings

MODELFILES_DIR = Path(__file__).resolve().parent.parent / "modelfiles"


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        provider="mock",
        modelfiles_dir=MODELFILES_DIR,
        db_path=tmp_path / "test.db",
        tokens={"dev-token": "dev-org"},
    )


@pytest.fixture
def client(settings) -> TestClient:
    return TestClient(create_app(settings))


@pytest.fixture
def auth() -> dict:
    return {"Authorization": "Bearer dev-token"}
