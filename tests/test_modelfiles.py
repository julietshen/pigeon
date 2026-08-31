from __future__ import annotations

from pigeon.modelfiles import load_modelfiles

from .conftest import MODELFILES_DIR


def test_load_examples():
    mfs = load_modelfiles(MODELFILES_DIR)
    assert set(mfs) >= {"shieldgemma-2b", "gpt-oss-safeguard"}


def test_classifier_kind():
    mfs = load_modelfiles(MODELFILES_DIR)
    assert mfs["shieldgemma-2b"].kind == "classifier"
    assert mfs["shieldgemma-2b"].version == "3"  # coerced from YAML


def test_byop_kind():
    mfs = load_modelfiles(MODELFILES_DIR)
    for name in ("gpt-oss-safeguard", "cope-b"):
        assert mfs[name].kind == "byop"
        assert mfs[name].policy_argument is True
