# llamacpp-jev

> A thin wrapper server exposing TypeSafe Jev's `POST /v1/systemone` API in front of an
> **unmodified** `llama-server`. No llama.cpp patch: it uses the primitives the server already
> has — a rendered chat template (`/apply-template`), prompt caching with context checkpoints,
> per-token logprobs (`n_probs`) and GBNF-constrained sampling — the same way
> [ekzhang/openjev-sglang](https://github.com/ekzhang/openjev-sglang) does on SGLang.

Text and **vision**: chat messages may carry `image_url` data-URI parts when `llama-server` runs
with a multimodal projector (`--mmproj`).

## What was verified (2026-09-21, M4 Pro 24 GB, Metal)

Model: `unsloth/Qwen3.5-2B-GGUF` `Qwen3.5-2B-Q8_0.gguf` + `mmproj-F16.gguf`. llama.cpp master
`3d82ef62` (b11063). Full record in [docs/DESIGN.md §5](docs/DESIGN.md).

| Request | Wall (median) | Correct |
|---|---|---|
| 4 typed questions about a **never-seen 448×448 image** (8 images) | **836 ms** (825–898) | 32/32 |
| Same image repeated | 493 ms | 4/4 |
| 4 typed text questions, repeated | 381 ms | — |

The idea for this project came from a tweet ([@kis](https://x.com/kis/status/2101426969971916863),
2026-09-20) claiming that a Jev-compatible server in front of llama.cpp lets "any model behave
JEV-like without modifications", demoed as Qwen3.5-2B answering 4 questions about a 448×448
image in 800 ms. No repository was linked or found. **This independent build reproduces the shape
and the number**: ≈ 0.5 s to encode the image and prefill 245 prefix tokens, ≈ 0.35 s for four
one-token branches. Caveats: the tweet's hardware is unknown; accuracy here is on synthetic
geometric images with unambiguous answers, not natural photos; and both Qwen3.5-0.8B and 2B fail
64-way choices (they collapse onto one label once two-letter labels appear — 4/10/26-way are
correct at every tested position).

## Run

```bash
# 1. llama.cpp with Metal (any recent master; verified at 3d82ef62)
cmake -B build-metal -DGGML_METAL=ON -DLLAMA_CURL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-metal --target llama-server -j

# 2. weights
hf download unsloth/Qwen3.5-2B-GGUF Qwen3.5-2B-Q8_0.gguf mmproj-F16.gguf

# 3. the wrapper (launches llama-server itself; --mmproj is optional for text-only)
uv sync
uv run llamajev serve --model /path/Qwen3.5-2B-Q8_0.gguf --mmproj /path/mmproj-F16.gguf \
  --llama-server /path/build-metal/bin/llama-server --slots 4 --port 8000
```

Or attach to a `llama-server` you already run: `uv run llamajev serve --connect http://127.0.0.1:8080`.
It must have been started with `--ctx-checkpoints 32 --checkpoint-min-step 0` (and `--mmproj` for images).

```bash
curl http://127.0.0.1:8000/v1/systemone -H 'Content-Type: application/json' -d '{
  "model": "jev-latest",
  "state": [{"role": "user", "content": "I was charged twice. Please refund the duplicate."}],
  "questions": {
    "refund":     {"type": "noul",   "instructions": "Does the user request a refund?"},
    "department": {"type": "choice", "instructions": "Which department should handle this?",
                   "criteria": {"billing": "Payments and refunds", "technical": "Software bugs"}},
    "urgency":    {"type": "score",  "instructions": "How urgent is the request?",
                   "criteria": ["Routine", "Urgent", "Emergency"]}
  }}'
```

For an image, make `state` a chat message whose content has an
`{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}` part.
`uv run llamajev smoke` sends a three-question request; `uv run python scripts/bench.py [--image f.png]`
prints per-phase timings from the `Server-Timing` header.

Routes: `POST /v1/systemone`, `GET /v1/models`, `GET /v1/limits`, `GET /health`, `GET /health/live`,
`GET /docs`. Errors are `{"error": {"message", "code"}}` with a closed set of codes (see
[docs/DESIGN.md §0](docs/DESIGN.md)).

## How a request runs

1. `state` plus one extra user turn ("evaluate using the question below … answer with only its
   label" + a marker) is rendered **once** through the GGUF's own chat template via
   `/apply-template` with thinking disabled, then split at the marker into a prefix and an ending.
2. One `/completion` call on the prefix with `n_predict: 1` warms the slot (image encoded here).
3. One `/completion` call per question — `prefix + "Question: … Options: A: … B: …" + ending +
   "Answer:\n"` — with `n_predict: 1`, `temperature: 0`, a GBNF grammar over the labels, and
   `n_probs`. All branches are pinned to the warm-up's slot (`id_slot`), so `llama-server` restores
   that slot's checkpoint and processes only the suffix.
4. The candidate labels' logprobs are read from the pre-sampling `top_logprobs` (exact full-vocab
   log-softmax, unaffected by the grammar) and renormalised: Noul → P(yes); Choice → argmax + full
   distribution; Score → Σ level × p. `confidence = 1 − H/log n`.

Labels are `A`–`Z` then verified single-token pairs, checked against the live tokenizer at startup.
`usage.input_tokens` is the backend's summed prompt counts including cached tokens.

## Tests

```bash
uv run pytest                                       # offline: fake backend
LLAMAJEV_LIVE_URL=http://127.0.0.1:8000 uv run pytest tests/test_live.py -v   # against a running server
```

## Layout

`src/llamajev/`: `config` · `models` (wire contract) · `prompts` (compiler) · `scoring` ·
`backend` (llama-server HTTP) · `service` (N+1 orchestration) · `api` · `runtime` · `cli`.
Design and measurements: `docs/DESIGN.md`. Sibling project with the same contract for diffusion
models on SGLang: [sglang-jev-diffusion](https://github.com/NakliTechie/sglang-jev-diffusion).

## License

MIT.
