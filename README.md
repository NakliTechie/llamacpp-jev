<h1 align="center">llamacpp-jev</h1>

**TypeSafe Jev's `POST /v1/systemone` structured-decision API in front of an unmodified
`llama-server`.** Ask a batch of typed questions — yes/no, choice, score — about a conversation or
an image, and get a probability distribution per question back, from any GGUF, with no llama.cpp patch.

*One wrapper process. macOS · Linux · Windows. No account, no telemetry. Text and vision.*

![license MIT](https://img.shields.io/badge/license-MIT-555?style=flat-square)
![llama.cpp unpatched](https://img.shields.io/badge/llama.cpp-unpatched-555?style=flat-square)
![api /v1/systemone](https://img.shields.io/badge/api-%2Fv1%2Fsystemone-555?style=flat-square)
![vision yes](https://img.shields.io/badge/vision-image__url-555?style=flat-square)

## Install

Needs one `llama-server` binary (built once) and a GGUF; the wrapper itself is `uv`-run.

```bash
# 1. llama.cpp with your backend — verified at master 3d82ef62 (b11063, 2026-09-20)
cmake -B build-metal -DGGML_METAL=ON -DLLAMA_CURL=ON -DCMAKE_BUILD_TYPE=Release   # or -DGGML_CUDA=ON
cmake --build build-metal --target llama-server -j
# 2. weights
hf download unsloth/Qwen3.5-2B-GGUF Qwen3.5-2B-Q8_0.gguf mmproj-F16.gguf
# 3. the wrapper (launches llama-server itself; --mmproj is optional, enables images)
uv sync
uv run llamajev serve --model /path/Qwen3.5-2B-Q8_0.gguf --mmproj /path/mmproj-F16.gguf \
  --llama-server /path/build-metal/bin/llama-server --slots 4 --port 8000
```

It comes up in a few seconds, verifies its single-token labels against the live tokenizer, and
serves on `:8000`. Point any HTTP client at it:

```bash
curl http://127.0.0.1:8000/v1/systemone -H 'Content-Type: application/json' -d '{
  "model": "jev-latest",
  "state": [{"role": "user", "content": "I was charged twice. Please refund the duplicate."}],
  "questions": {
    "refund":     {"type": "noul",   "instructions": "Does the user request a refund?"},
    "department": {"type": "choice", "instructions": "Which team?",
                   "criteria": {"billing": "Payments and refunds", "technical": "Software bugs"}},
    "urgency":    {"type": "score",  "instructions": "How urgent?", "criteria": ["Routine", "Urgent", "Emergency"]}
  }}'
```

No config file, no account, no restart between models. `uv run llamajev smoke` sends a
three-question request against a running server, so you can see the response shape without wiring a
client first. For an image, make a `state` message whose content has an
`{"type": "image_url", "image_url": {"url": "data:image/png;base64,…"}}` part.

## Why

You have a GGUF running under llama.cpp and you want it to answer a fixed set of typed questions —
is this a refund request, which team, how urgent — and hand back probabilities, not prose. TypeSafe
Jev does that behind a hosted API; doing it yourself normally means adopting their stack.

llamajev serves that same `/v1/systemone` contract from a **stock** `llama-server`, using only
primitives it already has: a rendered chat template (`/apply-template`), prompt caching with context
checkpoints, per-token logprobs (`n_probs`) and GBNF-constrained sampling — the same way
[openjev-sglang](https://github.com/ekzhang/openjev-sglang) does on SGLang. A client cannot tell
which engine is behind it.

**Use something else if:** you need TypeSafe Jev's calibrated (RLCD) probabilities or more than 64
choice options → TypeSafe Jev; you are on SGLang or a GPU cluster →
[openjev-sglang](https://github.com/ekzhang/openjev-sglang), the sibling this mirrors; you only want
free-form generation, not typed decisions → a stock `llama-server` alone. llamajev's probabilities
are raw label softmax, **not calibrated** — the full contract comparison is in
[docs/DESIGN.md §1](docs/DESIGN.md).

## Provenance

The idea came from a tweet ([@kis](https://x.com/kis/status/2101426969971916863), 2026-09-20)
claiming a Jev-compatible server in front of llama.cpp lets "any model behave JEV-like without
modifications", demoed as Qwen3.5-2B answering 4 questions about a 448×448 image in 800 ms. No
repository was linked or found; this is an independent build that reproduces the shape of the claim
and lands at or under the number (see below).

## Commands

```bash
uv run llamajev serve --model M.gguf --mmproj P.gguf --llama-server .../llama-server --slots 4  # launch llama-server + the API
uv run llamajev serve --connect http://127.0.0.1:8080                                           # attach to a running llama-server
uv run llamajev smoke [URL]                                                                     # one three-question request, prints timings
uv run python scripts/bench.py [URL] [--image f.png]                                            # per-phase timings from Server-Timing
uv run python scripts/fresh_bench.py URL IMAGE_DIR                                              # cold per-image latency + correctness
```

A `--connect` target must be started with `--ctx-checkpoints 32 --checkpoint-min-step 0 --cache-ram 0`
(and `--mmproj` for images). `--cache-ram 0` matters: the default 8192 MiB stalled for minutes on
repeated image prompts under concurrent GPU load here
([docs/llama-server-checkpoint-stall.md](docs/llama-server-checkpoint-stall.md)).

## Verify it yourself

```bash
uv run pytest                                                                  # offline, fake backend
LLAMAJEV_LIVE_URL=http://127.0.0.1:8000 uv run pytest tests/test_live.py -v    # against a running server
```

Measured 2026-09-22 on Qwen3.5-2B-Q8_0, M4 Pro 24 GB, Metal, default config (4 slots, slot pinning,
`--cache-ram 0`), GPU uncontended:

| Request | Wall (median) | Correct |
|---|---|---|
| 4 typed questions on a never-seen 448×448 image (8 images) | **526 ms** (525–546) | 32/32 |
| Same image repeated | 254 ms | 4/4 |
| 4 typed text questions, repeated | 238 ms | — |

Accuracy is synthetic geometric images (32/32) plus a 6-photo hand-labelled spot check (22/22), not
a benchmark. Known gap: Qwen3.5-0.8B/2B collapse on 64-way choices (4/10/26-way are correct at every
tested position); cause not established. Full record and per-config rows: [docs/DESIGN.md §5](docs/DESIGN.md).

## License

MIT. — [design & measurements](docs/DESIGN.md) · [checkpoint-stall report](docs/llama-server-checkpoint-stall.md) · sibling [sglang-jev-diffusion](https://github.com/NakliTechie/sglang-jev-diffusion)
