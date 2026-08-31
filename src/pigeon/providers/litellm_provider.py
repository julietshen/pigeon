from __future__ import annotations

from typing import Any, Optional

import httpx

from ..modelfiles import Modelfile

# LiteLLM model-name prefixes per runtime. See https://docs.litellm.ai/docs/providers
_RUNTIME_PREFIX = {
    "ollama": "ollama_chat",
    "vllm": "openai",  # vLLM serves an OpenAI-compatible endpoint
    "hf-inference": "huggingface",
    "endpoint": "openai",  # any OpenAI-compatible base URL
}


def _litellm_model(mf: Modelfile) -> str:
    prefix = _RUNTIME_PREFIX.get(mf.model.runtime, "openai")
    return f"{prefix}/{mf.model.id}"


class LiteLLMProvider:
    """Live provider. Chat/completion goes through LiteLLM; fixed-label classifiers hit an
    HF-style text-classification endpoint directly (LiteLLM targets generative APIs)."""

    def __init__(self, timeout_s: float = 30.0):
        self._timeout = timeout_s

    async def run_chat(self, mf: Modelfile, messages: list[dict]) -> str:
        import litellm

        kwargs: dict[str, Any] = {
            "model": _litellm_model(mf),
            "messages": messages,
            "temperature": 0,
            "timeout": self._timeout,
        }
        if mf.model.endpoint:
            kwargs["api_base"] = mf.model.endpoint
        response = await litellm.acompletion(**kwargs)
        return response.choices[0].message.content or ""

    async def run_chat_top_logprobs(
        self, mf: Modelfile, messages: list[dict]
    ) -> list[tuple[str, float]]:
        import litellm

        kwargs: dict[str, Any] = {
            "model": _litellm_model(mf),
            "messages": messages,
            "temperature": 0,
            "max_tokens": 1,
            "logprobs": True,
            "top_logprobs": 20,
            "timeout": self._timeout,
        }
        if mf.model.endpoint:
            kwargs["api_base"] = mf.model.endpoint
        response = await litellm.acompletion(**kwargs)
        entries = response.choices[0].logprobs.content[0].top_logprobs
        return [(entry.token, entry.logprob) for entry in entries]

    async def run_classifier(
        self, mf: Modelfile, *, text: Optional[str], media_url: Optional[str]
    ) -> Any:
        url = mf.model.endpoint
        if not url:
            raise ValueError(
                f"classifier modelfile '{mf.name}' requires model.endpoint"
            )
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            if media_url is not None:
                media = await client.get(media_url)
                media.raise_for_status()
                response = await client.post(
                    url,
                    content=media.content,
                    headers={
                        "Content-Type": media.headers.get(
                            "content-type", "application/octet-stream"
                        )
                    },
                )
            else:
                response = await client.post(url, json={"inputs": text or ""})
            response.raise_for_status()
            return response.json()
