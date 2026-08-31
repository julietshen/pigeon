from __future__ import annotations

from typing import Any, Optional

from ..modelfiles import Modelfile

"""In-process provider: loads a policy-adaptive classifier (Shieldstral) locally via
transformers on MPS/CUDA/CPU and scores in a single forward pass. Mirrors the reference
loader in ROOST/vibecheck/eval/models/shieldstral.py: the continuous score is the softmax
over the max single-token yes/no logits at the final position. No text is generated.

Requires the optional `local` extra (torch, transformers>=5.0). Models load lazily and are
cached per model id; a single local model instance is the bottleneck, so calls run
sequentially in practice."""


def _single_token_yesno_ids(tok) -> tuple[list[int], list[int]]:
    ids: dict[str, set[int]] = {"yes": set(), "no": set()}
    for word in ("yes", "no"):
        for variant in (word, word.capitalize(), word.upper()):
            for prefix in ("", " "):
                enc = tok.encode(prefix + variant, add_special_tokens=False)
                if len(enc) == 1:
                    ids[word].add(enc[0])
    if not ids["yes"] or not ids["no"]:
        raise ValueError("tokenizer has no single-token yes/no variants")
    return sorted(ids["yes"]), sorted(ids["no"])


class TransformersProvider:
    def __init__(self) -> None:
        # model id -> (tokenizer, model, device, yes_ids, no_ids)
        self._loaded: dict[str, tuple] = {}

    def _load(self, model_id: str) -> tuple:
        cached = self._loaded.get(model_id)
        if cached is not None:
            return cached

        import torch
        from transformers import AutoModelForImageTextToText, AutoTokenizer

        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = (
            AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16)
            .to(device)
            .eval()
        )
        yes_ids, no_ids = _single_token_yesno_ids(tokenizer)
        entry = (tokenizer, model, device, yes_ids, no_ids)
        self._loaded[model_id] = entry
        return entry

    async def run_chat_top_logprobs(
        self, mf: Modelfile, messages: list[dict]
    ) -> list[tuple[str, float]]:
        import torch

        tokenizer, model, device, yes_ids, no_ids = self._load(mf.model.id)
        enc = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        input_ids = (enc["input_ids"] if not torch.is_tensor(enc) else enc).to(device)
        with torch.inference_mode():
            logits = model(input_ids=input_ids).logits[0, -1].float()
        z_yes = max(logits[j].item() for j in yes_ids)
        z_no = max(logits[j].item() for j in no_ids)
        # score_from_yesno_logprobs applies softmax over these two values.
        return [("yes", z_yes), ("no", z_no)]

    async def run_chat(self, mf: Modelfile, messages: list[dict]) -> str:
        raise NotImplementedError(
            "TransformersProvider is scored via first-token logprobs; use parser "
            "type 'logprob_yesno'."
        )

    async def run_classifier(
        self, mf: Modelfile, *, text: Optional[str], media_url: Optional[str]
    ) -> Any:
        raise NotImplementedError(
            "TransformersProvider does not serve HF classifier endpoints."
        )
