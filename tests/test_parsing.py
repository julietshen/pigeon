from __future__ import annotations

import pytest

from pigeon.parsing import parse_chat_verdict, parse_classifier_scores, resolve_response_path


def test_classifier_flat_list():
    scores = parse_classifier_scores(
        [{"label": "a", "score": 0.9}, {"label": "b", "score": 0.1}]
    )
    assert scores == {"a": 0.9, "b": 0.1}


def test_classifier_batched_form():
    scores = parse_classifier_scores([[{"label": "a", "score": 0.5}]])
    assert scores["a"] == 0.5


def test_classifier_dict_form():
    assert parse_classifier_scores({"a": 0.3})["a"] == 0.3


def test_classifier_clamps():
    assert parse_classifier_scores([{"label": "a", "score": 1.4}])["a"] == 1.0


def test_classifier_empty_raises():
    with pytest.raises(ValueError):
        parse_classifier_scores([])


def test_verdict_numeric():
    assert parse_chat_verdict("0.8") == 0.8


def test_verdict_json_score():
    assert parse_chat_verdict('{"score": 0.4}') == 0.4


def test_verdict_keywords():
    assert parse_chat_verdict("Yes, this violates the policy") == 1.0
    assert parse_chat_verdict("no") == 0.0
    assert parse_chat_verdict("unsure") == 0.5


def test_verdict_uninterpretable_raises():
    with pytest.raises(ValueError):
        parse_chat_verdict("purple monkey dishwasher")


def test_response_path_nested():
    assert resolve_response_path({"a": {"b": [10, 20]}}, "a.b.1") == 20


def test_response_path_empty_returns_root():
    root = {"x": 1}
    assert resolve_response_path(root, None) is root
