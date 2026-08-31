from __future__ import annotations


def main() -> None:
    import uvicorn

    # Default bind is loopback; do not expose without an explicit host.
    uvicorn.run(
        "pigeon.main:create_app",
        factory=True,
        host="127.0.0.1",
        port=8900,
    )
