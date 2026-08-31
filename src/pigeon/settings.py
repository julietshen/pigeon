from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DEV_TOKEN = "dev-token"
DEFAULT_DEV_ORG = "dev-org"


@dataclass(frozen=True)
class Settings:
    provider: str
    modelspecs_dir: Path
    db_path: Path
    tokens: dict[str, str]  # bearer token -> org id

    @staticmethod
    def from_env() -> "Settings":
        raw_tokens = os.environ.get("PIGEON_TOKENS")
        tokens = json.loads(raw_tokens) if raw_tokens else {DEFAULT_DEV_TOKEN: DEFAULT_DEV_ORG}
        return Settings(
            provider=os.environ.get("PIGEON_PROVIDER", "mock"),
            modelspecs_dir=Path(os.environ.get("PIGEON_MODELSPECS_DIR", "modelspecs")),
            db_path=Path(os.environ.get("PIGEON_DB_PATH", "pigeon.db")),
            tokens=tokens,
        )
