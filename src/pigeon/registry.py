from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .modelfiles import Modelfile
from .schemas import LabelSpec, ModelfileSummary
from .store import Store


@dataclass
class Resolved:
    modelfile: Modelfile
    version: str
    bound_policy: Optional[str]


def _titlecase(label: str) -> str:
    return label.replace("_", " ").replace("-", " ").strip().title()


class Registry:
    """Turns loaded modelfiles + per-org policies into the signal list a consumer sees,
    and resolves a model reference ("name" or "name@version") to something callable."""

    def __init__(self, modelfiles: dict[str, Modelfile], store: Store):
        self.modelfiles = modelfiles
        self.store = store

    def ensure_defaults(self, org_id: str) -> None:
        """First contact for an org enables every base modelfile. Prototype convenience."""
        if not self.store.enabled_modelfiles(org_id):
            for name in self.modelfiles:
                self.store.enable(org_id, name)

    def list_signals(self, org_id: str) -> list[ModelfileSummary]:
        self.ensure_defaults(org_id)
        enabled = self.store.enabled_modelfiles(org_id)
        summaries: list[ModelfileSummary] = []

        # Fixed-label classifiers are exposed directly. BYOP bases are exposed only through
        # bound policies (below); completion models are Osprey-only and not signals.
        for name, mf in self.modelfiles.items():
            if name in enabled and mf.kind == "classifier":
                summaries.append(self._classifier_summary(mf))

        for pol in self.store.latest_policies(org_id):
            base = self.modelfiles.get(pol["base"])
            if base is None:
                continue
            summaries.append(
                ModelfileSummary(
                    id=pol["name"],
                    version=str(pol["version"]),
                    kind="byop",
                    base=pol["base"],
                    inputTypes=list(base.input_types),
                    labels=[LabelSpec(id="verdict", display=pol["display"])],
                )
            )
        return summaries

    def _classifier_summary(self, mf: Modelfile) -> ModelfileSummary:
        labels = [LabelSpec(id=lbl, display=_titlecase(lbl)) for lbl in mf.labels]
        return ModelfileSummary(
            id=mf.name,
            version=mf.version,
            kind="classifier",
            inputTypes=list(mf.input_types),
            labels=labels,
        )

    def resolve(self, org_id: str, model_ref: str) -> Resolved:
        name, _, version = model_ref.partition("@")

        pol = self.store.get_policy(
            org_id, name, int(version) if version.isdigit() else None
        )
        if pol is not None:
            base = self.modelfiles.get(pol["base"])
            if base is None:
                raise KeyError(
                    f"base modelfile '{pol['base']}' not found for policy '{name}'"
                )
            return Resolved(
                modelfile=base, version=str(pol["version"]), bound_policy=pol["policy_text"]
            )

        mf = self.modelfiles.get(name)
        if mf is None:
            raise KeyError(f"unknown model '{name}'")
        return Resolved(modelfile=mf, version=mf.version, bound_policy=None)
