from __future__ import annotations

from pigeon.modelfiles import load_modelfiles

from .conftest import MODELFILES_DIR


def test_load_examples():
    mfs = load_modelfiles(MODELFILES_DIR)
    assert set(mfs) >= {"shieldgemma-2b", "shieldstral", "cope-b"}


def test_classifier_kind():
    mfs = load_modelfiles(MODELFILES_DIR)
    assert mfs["shieldgemma-2b"].kind == "classifier"
    assert mfs["shieldgemma-2b"].version == "3"  # coerced from YAML


def test_multimodal_byop_kind():
    mfs = load_modelfiles(MODELFILES_DIR)
    assert mfs["shieldstral"].kind == "byop"
    assert mfs["shieldstral"].policy_argument is True
    assert set(mfs["shieldstral"].input_types) == {"text", "image"}


def test_byop_kind():
    mfs = load_modelfiles(MODELFILES_DIR)
    assert mfs["cope-b"].kind == "byop"
    assert mfs["cope-b"].policy_argument is True
