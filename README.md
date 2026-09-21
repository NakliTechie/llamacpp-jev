# llamacpp-jev

> A thin wrapper server exposing TypeSafe Jev's `/v1/systemone` API in front of an **unmodified** `llama-server`, using primitives llama.cpp already has. No core patch — unlike its sibling project [sglang-jev-diffusion](https://github.com/NakliTechie/sglang-jev-diffusion), which needs one.

## Why this is a different shape of project than its sibling

[sgl-project/sglang](https://github.com/sgl-project/sglang) needed a core patch for Jev-like serving on **diffusion** models specifically, because a diffusion LM's fixed-width iterative canvas has no equivalent in the engine's existing API. `llama.cpp` doesn't have that problem, for two reasons:

1. **llama.cpp has no diffusion-language-model support at all.** There's no DiffusionGemma-class target to serve, so there's no canvas-seeding/pinning gap to fill in the first place.
2. **For autoregressive models, llama.cpp's own server already exposes the primitives needed** — per-token logprobs (`n_probs`) and grammar-constrained/GBNF decoding — the same class of primitive [ekzhang/openjev-sglang](https://github.com/ekzhang/openjev-sglang) used to get Jev-compatible serving on SGLang's autoregressive path with **zero core engine changes**.

So this project should need no llama.cpp source changes. It's an application-layer server, structured the same way as openjev-sglang: render the chat template once, split a common prefix, issue N+1 single-token calls (one cache-warming call plus one per typed question), read exact logprobs for the candidate answer tokens, renormalize with softmax.

## Where the idea came from — and what it doesn't verify yet

A Japanese-language tweet ([@kis](https://x.com/kis/status/2101426969971916863), 2026-09-20) claimed: "if you integrate a JEV-compatible server into llama.cpp, any model can behave JEV-like without modifications," demoed with Qwen3.5-2B answering 4 questions about a 448×448 image in 800ms. **No repo was linked in that tweet**, and none was found searching around it. This project exists partly to build the thing described and partly to **independently check whether the claim is accurate** — including the vision angle, which is a step beyond openjev-sglang's text-only design and needs llama.cpp's multimodal (LLaVA-style) support specifically.

## Design

Same request contract as [sglang-jev-diffusion](https://github.com/NakliTechie/sglang-jev-diffusion) and TypeSafe's own Jev API, so a client doesn't need to know which engine is behind it:

- `POST /v1/systemone` — `state` (text, JSON, or chat messages) + `questions` (typed `choice` / `score` / `noul`) in, typed answers + calibrated-looking probabilities out.
- Backed by an ordinary `llama-server` process, driven via its existing OpenAI-compatible `/v1/chat/completions` or `/completion` endpoint with `n_probs` (or `logprobs`) and a constrained single-token response.

## Status

Scaffolded. No code yet — see `plan/workplan.md`.

## License

MIT (matching `ekzhang/openjev-sglang`, which this project's structure closely follows).

## Context

Scaffolded alongside `sglang-jev-diffusion` from the same same-day research pass — see `plan/history.md` for the full source trail from the knowledge vault (`~/Code/knowledge`).
