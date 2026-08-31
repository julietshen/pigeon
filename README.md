# Pigeon

Pigeon makes open-weight safety models usable inside [Coop](https://github.com/roostorg/coop)
and [Osprey](https://github.com/roostorg/osprey) with the same effort as adding a hosted API.
A model is described by a **modelfile** (a small YAML recipe), not a per-model integration
package. Pigeon is a service: consumers call it over HTTP; it holds the modelfiles, the
provider credentials, and the response-parsing logic.

It also runs standalone, as a way to manage and send requests to open weights across any
inference provider, without Coop or Osprey.

> Status: prototype. The contracts and flows are working end to end against a mock provider;
> the eng team takes it forward from here (hardening, real inference, persistence, UI).

## What it does

- **Discover** (`GET /v1/modelfiles`): lists the signals an org can use. Fixed-label
  classifiers fan out to one label each; policy-bound BYOP custom models appear as a single
  verdict signal.
- **Classify** (`POST /v1/classify`): input in, normalized `[{label, score}]` out. The caller
  never sees prompts, wire formats, or response paths.
- **Policy management** (`POST /v1/policies`): author a policy against a policy-steerable base
  (gpt-oss-safeguard, CoPE-B) to mint a versioned custom model. This is the "build a custom
  model from an open-weight base" path.

## Model kinds

| Kind | modelfile | Signals | Policy |
| --- | --- | --- | --- |
| `classifier` | `format: classifier`, fixed `labels` | one per label | none |
| `byop` | `format: chat`, `policy_argument: true` | one per bound policy | pre-bound (Coop) or per-call (Osprey) |
| `completion` | `mode: completion` | not a Coop signal | Osprey-only |

## Quickstart

```bash
uv sync --extra dev
uv run pytest                 # runs against the mock provider, no network

cp .env.example .env
uv run pigeon                 # serves on http://127.0.0.1:8900
```

Try it (dev token from `.env.example`):

```bash
TOKEN="dev-token"
curl -s localhost:8900/v1/modelfiles -H "Authorization: Bearer $TOKEN"

curl -s localhost:8900/v1/classify -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"model":"shieldgemma-2b","input":{"text":"you are worthless"}}'

# Author a policy -> custom model, then classify against it
curl -s localhost:8900/v1/policies -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"name":"my-harassment-policy","base":"gpt-oss-safeguard","policyText":"Flag harassment.","display":"Harassment policy"}'
```

Set `PIGEON_PROVIDER=live` to call real inference through LiteLLM (chat/completion) and
HF-style endpoints (classifiers). Point each modelfile's `model.endpoint` at your runtime.

## Layout

- `src/pigeon/modelfiles.py` - modelfile spec + loader
- `src/pigeon/registry.py` - modelfiles + policies -> the signal list; model-ref resolution
- `src/pigeon/parsing.py` - response normalization (ported from Coop's `modelClient.ts`)
- `src/pigeon/classify.py` - the classify orchestration
- `src/pigeon/providers/` - the inference seam (`mock`, `litellm_provider`)
- `src/pigeon/api.py` - the HTTP endpoints
- `modelfiles/` - example modelfiles

## Known gaps (for the eng team)

- Response caching (one model call serving every label) is not yet ported from Coop.
- `chat-harmony` format is declared but not yet implemented.
- Auth is a static token->org map; wire to real per-org credentials.
- Provider layer calls LiteLLM directly; a LiteLLM-proxy adapter is the likely next step.
