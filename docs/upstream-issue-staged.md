# Staged: llama.cpp feature-request issue (per-token-id logprobs)

Status: staged, NOT posted. Nothing has been sent to GitHub.

This file is a scaffold, not the issue text. llama.cpp's issue template says "Please fill out this template yourself". Its CONTRIBUTING.md bans AI-written issues, and the penalty is a ban. The facts below are for the author to use. The prose in each section must be the author's own.

Where to post: https://github.com/ggml-org/llama.cpp/issues/new?template=020-enhancement.yml
The template first suggests a post in Discussions > Ideas if there is no consensus yet: https://github.com/ggml-org/llama.cpp/discussions/categories/ideas

## Title

The template prefills `Feature Request: `. The author adds a short title after it.

## Prerequisites (4 checkboxes, all required)

- [ ] Running latest code. Facts: prototype built on master `5cf3a35` (2026-09-25). Wrapper evidence on `3d82ef62` (b11063).
- [ ] Followed README.md.
- [ ] Searched existing issues. Facts: 14 `gh search` queries over issues and PRs on 2026-09-25 found no request for this. Related: #27174 (prompt/echo logprobs), #28368 (cache_prompt changes logprobs).
- [ ] Reviewed Discussions. Facts: #8221 "Get Specific logits" (Q&A, 2024-06-30) describes the same need. The asker's workaround bans every other token with `logit_bias` and requests `n_probs: 2`, which they reported as slower. https://github.com/ggml-org/llama.cpp/discussions/8221

## Feature Description (required, author's words)

Facts to draw on:
- Request field: an array of token ids (prototype name `token_ids_logprob`, which is SGLang's name).
- Output: next to `top_logprobs`, one `{id, token, bytes, logprob}` object for each requested id, in request order (`token_ids_logprobs`; `token_ids_probs` with `post_sampling_probs`).
- Values come from the same full-vocabulary softmax that `n_probs` already computes. There is no depth limit.

## Motivation (required, author's words)

Facts to draw on:
- Use case: classification or typed-decision scoring. The caller reads P(label) for K known labels at one position, for example Jev-style `/v1/systemone`, yes/no judges, or multiple choice.
- Today the only route is top-N. A low-probability label can sit deep in the list. Measured on Qwen3.5-2B-Q8_0: label `ZZ` (id 32424) ranked 28,732, so only `n_probs >= 28733` returns it.
- Cost of the workaround, same machine (M4 Pro, Metal, warm cache, n_predict 1, median of 20 requests): `n_probs=32768` 99.1 ms; `n_probs=256` 33.7 ms; per-id readout for 4 ids 28.1 ms.
- The llamacpp-jev wrapper retries at depth 4096, then 32768, then fails with `readout_truncated`. With per-id reads it needs one call and no retries.
- Precedent: SGLang ships `token_ids_logprob`, and ekzhang/openjev-sglang uses it for the same readout.
- Another user with the same need: Discussion #8221.

## Possible Implementation (optional, author's words)

Facts to draw on:
- Prototype: branch `server-token-probs` in `~/Code/llama.cpp-tokprobs`. 4 C++ files, +66/-12 lines. One new test in the existing `tools/server/tests/unit/test_completion.py`, +27 lines.
- It reuses `get_token_probabilities` (`server-common.cpp:1526`), the `server-schema.cpp` field system and the `completion_token_output` JSON. No existing output changes.
- An id outside `[0, n_vocab)` returns HTTP 400. `logit_bias`, by contrast, drops bad ids silently.
- Accuracy check: per-id values match an `n_probs=32768` readout within 3.1e-5 for all 4 ids. The offset is the same for every id, which is consistent with float32 rounding in the softmax sum.
- Works on `/completion`, streamed `/completion`, and `/v1/chat/completions` (the key appears in `logprobs.content[]`).
- Design options and rejected alternatives: `docs/upstream-token-logprobs.md`.
- The rules say features start as an issue and a PR follows only after maintainer interest. Whether to mention the prototype is the author's call.

## Before posting

- [ ] Read the prototype diff and be able to explain every line: `git -C ~/Code/llama.cpp-tokprobs diff origin/master`.
- [ ] Choose the field name and the out-of-range behaviour (open points 1 and 2 in `docs/upstream-token-logprobs.md`).
- [ ] For any later PR, CONTRIBUTING.md requires AI disclosure ("Explicitly disclose the manner in which AI was employed").
