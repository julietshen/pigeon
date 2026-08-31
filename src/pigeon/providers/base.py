from __future__ import annotations

from typing import Any, Optional, Protocol

from ..modelfiles import Modelfile


class ProviderClient(Protocol):
    """The seam between Pigeon and inference. Swap LiteLLM for anything else here."""

    async def run_chat(self, mf: Modelfile, messages: list[dict]) -> str:
        """Send a chat/completion request; return the assistant message text."""
        ...

    async def run_chat_top_logprobs(
        self, mf: Modelfile, messages: list[dict]
    ) -> list[tuple[str, float]]:
        """Send a chat request scored by first-token logprobs (Shieldstral). Return the
        first generated token's top_logprobs as (token, logprob) pairs."""
        ...

    async def run_classifier(
        self, mf: Modelfile, *, text: Optional[str], media_url: Optional[str]
    ) -> Any:
        """Send a text-classification request; return the raw parsed JSON body."""
        ...
