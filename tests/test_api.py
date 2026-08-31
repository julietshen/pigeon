from __future__ import annotations


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_auth_required(client):
    assert client.get("/v1/modelfiles").status_code == 401


def test_discover_lists_classifier_not_byop_base(client, auth):
    body = client.get("/v1/modelfiles", headers=auth).json()
    ids = {m["id"] for m in body["modelfiles"]}
    assert "shieldgemma-2b" in ids
    # BYOP bases are reachable only via a bound policy, never exposed as a signal directly.
    assert "gpt-oss-safeguard" not in ids
    assert "cope-b" not in ids


def test_discover_classifier_fans_out_to_labels(client, auth):
    body = client.get("/v1/modelfiles", headers=auth).json()
    sg = next(m for m in body["modelfiles"] if m["id"] == "shieldgemma-2b")
    label_ids = {lbl["id"] for lbl in sg["labels"]}
    assert label_ids == {"harassment", "hate", "sexual", "dangerous"}


def test_classify_classifier(client, auth):
    resp = client.post(
        "/v1/classify",
        headers=auth,
        json={"model": "shieldgemma-2b", "input": {"text": "you are worthless, harassment"}},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == "shieldgemma-2b@3"
    scores = {r["label"]: r["score"] for r in body["results"]}
    assert scores["harassment"] > 0.5


def test_classify_unknown_model_404(client, auth):
    resp = client.post(
        "/v1/classify",
        headers=auth,
        json={"model": "does-not-exist", "input": {"text": "hi"}},
    )
    assert resp.status_code == 404


def test_byop_pre_bound_policy_flow(client, auth):
    # Author a policy against the BYOP base -> a versioned custom model.
    created = client.post(
        "/v1/policies",
        headers=auth,
        json={
            "name": "my-harassment-policy",
            "base": "gpt-oss-safeguard",
            "policyText": "Flag content that harasses a person.",
            "display": "Harassment policy",
        },
    )
    assert created.status_code == 200
    assert created.json()["version"] == "1"

    # It now shows up in discover as a single-label byop signal.
    disc = client.get("/v1/modelfiles", headers=auth).json()["modelfiles"]
    custom = next(m for m in disc if m["id"] == "my-harassment-policy")
    assert custom["kind"] == "byop"
    assert custom["base"] == "gpt-oss-safeguard"
    assert [lbl["id"] for lbl in custom["labels"]] == ["verdict"]

    # Classify against it: policy is held server-side, request sends no policy.
    resp = client.post(
        "/v1/classify",
        headers=auth,
        json={"model": "my-harassment-policy", "input": {"text": "stop harassing people"}},
    )
    assert resp.status_code == 200
    result = resp.json()["results"][0]
    assert result["label"] == "verdict"
    assert result["score"] > 0.5


def test_byop_policy_versions_increment(client, auth):
    for expected in ("1", "2"):
        r = client.post(
            "/v1/policies",
            headers=auth,
            json={"name": "p", "base": "gpt-oss-safeguard", "policyText": f"v{expected}"},
        )
        assert r.json()["version"] == expected


def test_policy_rejects_non_byop_base(client, auth):
    r = client.post(
        "/v1/policies",
        headers=auth,
        json={"name": "bad", "base": "shieldgemma-2b", "policyText": "x"},
    )
    assert r.status_code == 422


def test_byop_cope_b_policy_flow(client, auth):
    created = client.post(
        "/v1/policies",
        headers=auth,
        json={"name": "cope-harassment", "base": "cope-b", "policyText": "Flag harassment."},
    )
    assert created.status_code == 200

    disc = client.get("/v1/modelfiles", headers=auth).json()["modelfiles"]
    custom = next(m for m in disc if m["id"] == "cope-harassment")
    assert custom["kind"] == "byop"
    assert custom["base"] == "cope-b"

    resp = client.post(
        "/v1/classify",
        headers=auth,
        json={"model": "cope-harassment", "input": {"text": "harass"}},
    )
    assert resp.status_code == 200
    assert resp.json()["results"][0]["score"] > 0.5


def test_byop_runtime_policy_osprey_path(client, auth):
    # Osprey path: call the base directly with a policy supplied at request time.
    resp = client.post(
        "/v1/classify",
        headers=auth,
        json={
            "model": "gpt-oss-safeguard",
            "input": {"text": "harass"},
            "policy": "Flag harassment.",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["results"][0]["label"] == "verdict"
