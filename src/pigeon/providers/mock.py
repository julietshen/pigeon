from __future__ import annotations

from typing import Any, Optional

from ..modelfiles import Modelfile

_VIOLATING = ("harass", "kill", "hate", "abuse", "threat", "violat")


def _looks_violating(text: Optional[str]) -> bool:
    lowered = (text or "").lower()
    return any(word in lowered for word in _VIOLATING)


class MockProvider:
    """Deterministic provider for local dev and tests. No network, no model."""

    async def run_chat(self, mf: Modelfile, messages: list[dict]) -> str:
        content = messages[-1]["content"] if messages else ""
        return "0.9" if _looks_violating(content) else "0.05"

    async def run_chat_top_logprobs(
        self, mf: Modelfile, messages: list[dict]
    ) -> list[tuple[str, float]]:
        content = messages[-1]["content"] if messages else ""
        # Skew the yes/no logprobs so softmax lands near 0.9 / 0.1.
        if _looks_violating(content):
            return [("yes", -0.05), ("no", -2.3)]
        return [("yes", -2.3), ("no", -0.05)]

    async def run_classifier(
        self, mf: Modelfile, *, text: Optional[str], media_url: Optional[str]
    ) -> Any:
        hit = _looks_violating(text)
        primary = mf.labels[0] if mf.labels else "verdict"
        # HF batched form: [[{label, score}, ...]]
        return [
            [
                {"label": lbl, "score": 0.9 if (hit and lbl == primary) else 0.05}
                for lbl in (mf.labels or [primary])
            ]
        ]
