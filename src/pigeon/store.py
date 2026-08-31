from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Optional

"""Per-org state: versioned BYOP policies and which base modelspecs an org has enabled.
SQLite keeps the prototype self-contained; swap for the eng team's datastore later."""


class Store:
    def __init__(self, path: Path):
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init()

    def _init(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS policies (
                org_id TEXT NOT NULL,
                name TEXT NOT NULL,
                version INTEGER NOT NULL,
                base TEXT NOT NULL,
                policy_text TEXT NOT NULL,
                display TEXT NOT NULL,
                created_at REAL NOT NULL,
                PRIMARY KEY (org_id, name, version)
            );
            CREATE TABLE IF NOT EXISTS enabled (
                org_id TEXT NOT NULL,
                modelspec TEXT NOT NULL,
                PRIMARY KEY (org_id, modelspec)
            );
            """
        )
        self._conn.commit()

    # ---- policies (each edit mints a new version) ----

    def upsert_policy(
        self, org_id: str, name: str, base: str, policy_text: str, display: str
    ) -> int:
        row = self._conn.execute(
            "SELECT COALESCE(MAX(version), 0) AS v FROM policies WHERE org_id = ? AND name = ?",
            (org_id, name),
        ).fetchone()
        version = row["v"] + 1
        self._conn.execute(
            "INSERT INTO policies (org_id, name, version, base, policy_text, display, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (org_id, name, version, base, policy_text, display, time.time()),
        )
        self._conn.commit()
        return version

    def latest_policies(self, org_id: str) -> list[dict]:
        rows = self._conn.execute(
            """
            SELECT p.* FROM policies p
            JOIN (
                SELECT name, MAX(version) AS v FROM policies WHERE org_id = ? GROUP BY name
            ) m ON p.name = m.name AND p.version = m.v
            WHERE p.org_id = ?
            ORDER BY p.name
            """,
            (org_id, org_id),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_policy(
        self, org_id: str, name: str, version: Optional[int] = None
    ) -> Optional[dict]:
        if version is None:
            row = self._conn.execute(
                "SELECT * FROM policies WHERE org_id = ? AND name = ?"
                " ORDER BY version DESC LIMIT 1",
                (org_id, name),
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT * FROM policies WHERE org_id = ? AND name = ? AND version = ?",
                (org_id, name, version),
            ).fetchone()
        return dict(row) if row else None

    # ---- modelspec enablement ----

    def enable(self, org_id: str, modelspec: str) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO enabled (org_id, modelspec) VALUES (?, ?)",
            (org_id, modelspec),
        )
        self._conn.commit()

    def enabled_modelspecs(self, org_id: str) -> set[str]:
        rows = self._conn.execute(
            "SELECT modelspec FROM enabled WHERE org_id = ?", (org_id,)
        ).fetchall()
        return {r["modelspec"] for r in rows}
