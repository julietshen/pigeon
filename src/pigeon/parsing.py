from __future__ import annotations

import json
import math
from typing import Any, Optional

"""Response normalization. Ported from Coop's server/integrations/openmodel/modelClient.ts
so this model-specific glue lives in Pigeon instead of every consumer."""


def clamp01(n: float) -> float:
    return min(1.0, max(0.0, n))


def resolve_response_path(root: Any, path: Optional[str]) -> Any:
    """Walk a dot-path ("0.results" -> root[0]["results"]) into a parsed response."""
    if not path:
        return root
    current: Any = root
    for segment in path.split("."):
        if isinstance(current, list):
            try:
                current = current[int(segment)]
            except (ValueError, IndexError):
                return None
        elif isinstance(current, dict):
            current = current.get(segment)
        else:
            return None
    return current


def parse_classifier_scores(payload: Any) -> dict[str, float]:
    """Normalize a classifier response into label -> score. Accepts the HF conventions:
    [{label, score}, ...], [[{label, score}, ...]] (batched), and {label: score, ...}."""
    container = payload
    if isinstance(container, list) and container and isinstance(container[0], list):
        container = container[0]

    scores: dict[str, float] = {}
    if isinstance(container, list):
        for entry in container:
            if isinstance(entry, dict):
                label, score = entry.get("label"), entry.get("score")
                if isinstance(label, str) and isinstance(score, (int, float)):
                    scores[label] = clamp01(float(score))
    elif isinstance(container, dict):
        for label, score in container.items():
            if isinstance(score, (int, float)):
                scores[label] = clamp01(float(score))

    if not scores:
        raise ValueError(
            "could not extract any label scores from the response; check parser.response_path"
        )
    return scores


_YES = ("yes", "true", "violat", "unsafe", "flag")
_NO = ("no", "false", "safe", "clean", "allow")
_MAYBE = ("unsure", "uncertain", "unclear", "maybe")


def parse_chat_verdict(content: str) -> float:
    """Map a chat completion's text to a score (Zentropi convention): numeric content is used
    directly; JSON may carry {score}; else violating -> 1, safe -> 0, uncertain -> 0.5."""
    text = content.strip()

    try:
        n = float(text)
        if math.isfinite(n):
            return clamp01(n)
    except ValueError:
        pass

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and isinstance(parsed.get("score"), (int, float)):
            return clamp01(float(parsed["score"]))
    except (ValueError, TypeError):
        pass

    lowered = text.lower()
    if any(w in lowered for w in _YES):
        return 1.0
    if any(w in lowered for w in _NO):
        return 0.0
    if any(w in lowered for w in _MAYBE):
        return 0.5
    raise ValueError(f'could not interpret chat verdict "{text[:80]}"')
