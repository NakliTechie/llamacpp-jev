# Upstream option: per-token-id logprobs in llama-server

Status: design options for Chirag to choose from. The prototype builds the recommended option (A) on a private branch. Nothing is filed or pushed.

## The gap

`llama-server` returns probabilities only as a top-N list: `n_probs` (alias `logprobs`), `top_logprobs` on the OAI endpoints, and `post_sampling_probs`. You cannot name a token id and get its probability. llamacpp-jev therefore maps each option to a single-token label and reads the label out of a top-N list. When a label falls outside the list, the wrapper retries at depth 4096, then 32768, then fails with `readout_truncated` (`src/llamajev/service.py:16-17`, `:155-180`).

Upstream check on master `5cf3a35` (2026-09-25): no field, open PR or issue covers this. The nearest open issue is #27174 (prompt/echo logprobs for loglikelihood evals). SGLang has the feature as `token_ids_logprob`, and `ekzhang/openjev-sglang` uses it for the same Jev readout.

## Upstream rules that shape the contribution

From `AGENTS.md` and `CONTRIBUTING.md` on master:

- Features start as an issue, not a PR. A new public API field has a higher bar: the issue must say why existing fields are not enough.
- Chirag writes the issue, the PR description, the commit messages and all replies personally. AI-written text leads to closure or a ban.
- Chirag must own the design and be able to explain every line without help.
- Don't add new files under `tests/`. The server tests belong in the existing `tools/server/tests/unit/test_completion.py`.
- Commit trailer `Assisted-by: Claude`, never `Co-authored-by`.

## Options

### A. New request field `token_ids_logprob: [int]`, output next to `top_logprobs` (recommended)

- Request: an array of token ids on `/completion` (and on the OAI endpoints, which share the parser in `server-schema.cpp`).
- Output: each entry in `completion_probabilities` (and in `logprobs.content`) gets `token_ids_logprobs`: one `{id, token, bytes, logprob}` object for each requested id, in request order. With `post_sampling_probs: true`, the key is `token_ids_probs` and the values are `prob`. This mirrors the existing `top_logprobs` / `top_probs` switch.
- Pre-sampling values come from the same full-vocabulary softmax that `n_probs` already computes (`get_token_probabilities`, `server-common.cpp:1526`). They are exact for every id, with no depth limit.
- Post-sampling values come from the sampler candidate list. An id that the chain removed reports `prob: 0.0`. The existing `top_probs` list drops such tokens instead.
- An id outside `[0, n_vocab)` returns HTTP 400. `logit_bias` silently drops bad ids, but a readout must not silently lose a value.
- Cost: one O(n_vocab) pass per generated token, only when the field is set. This is the same order as the existing `n_probs` softmax.
- Size: +66/-12 lines of C++ in `server-schema.cpp`, `server-task.{h,cpp}` and `server-context.cpp`, plus one test in the existing file.

Why this shape: it reuses the existing parser, the existing probability code and the existing output object. It adds one field and changes no existing output. The name matches SGLang, so a client that targets both servers sends the same field.

### B. Accept an array in `n_probs` (rejected)

This saves a field name. But `n_probs` is an integer, and its alias `logprobs` is a boolean or an integer on the OAI endpoints. An array there overloads one field with three types. It also blurs "top N" with "these ids".

### C. Merge the requested ids into `top_logprobs` (rejected)

This adds no output key. But `top_logprobs` stops being sorted and stops being "top". OAI clients that read `top_logprobs[0]` as the argmax would break.

### D. New endpoint, e.g. `/score` (rejected)

A new endpoint duplicates prompt handling, slot selection and cache reuse. The maintainers ask for reuse of existing infrastructure over new subsystems.

### E. Prompt/echo logprobs, issue #27174 (complementary, not a substitute)

Echo logprobs score tokens that are already in the prompt. A Jev readout needs K candidate ids at one position. With echo it would need K forced continuations. Option A answers the one-position case in a single call. The issue could mention #27174 as a related request.

### F. Token strings, as `logit_bias` accepts (deferred)

A string can tokenize to more than one token, and a readout at one position is then undefined. Start with ids only. The client already has `/tokenize`.

## Open design points for Chirag

1. The field name: `token_ids_logprob` (SGLang parity) or a llama.cpp-style name such as `probs_token_ids`.
2. Whether the out-of-range 400 is right, or whether to match `logit_bias` and drop bad ids.
3. Whether to cap the list length. The prototype does not cap it; the cost is linear in n_vocab, not in the list length.
4. Speculative decoding: issue #27972 reports wrong logprobs with it enabled. The prototype inherits whatever `n_probs` does there. It does not fix that bug.

## Prototype record (2026-09-25)

- Branch `server-token-probs` in the worktree `~/Code/llama.cpp-tokprobs`, based on master `5cf3a35`. Uncommitted: llama.cpp's AGENTS.md asks for human approval of each commit. Patch copy: `docs/evidence/server-token-probs-5cf3a35.patch` (4 C++ files +66/-12, 1 test +31).
- Build: `cmake --build build-metal --target llama-server` compiles with no errors or warnings in the changed files.
- Raw server, Qwen3.5-2B-Q8_0, M4 Pro Metal: per-id values match an `n_probs=32768` readout within 3.1e-5 for 4 ids (ranks 3, 30, 186, 28,732). Median of 20: per-id 28.1 ms, `n_probs=256` 33.7 ms, `n_probs=32768` 99.1 ms. Bad id returns 400. The field also works on streamed `/completion` and on `/v1/chat/completions`.
- The unmodified server (same commit, `bin-base/`) ignores the field and returns 200 without `token_ids_logprobs`. The wrapper detects support from that response shape.
- llamacpp-jev on the patched server: `/health` shows `readout: token_ids`, `tests/test_live.py` passes 5/5, and 3 of 3 64-way requests return 200 with 0 retries. On the unpatched server: `readout: top_n`, 5/5 pass. Answers from the two servers differ by at most 1.7e-8 over 12 numbers.
- Upstream suite, run from `tools/server/tests` with `LLAMA_SERVER_BIN_PATH=<bin>/llama-server python -m pytest unit/test_chat_completion.py unit/test_completion.py --deselect unit/test_completion.py::test_completion_stream_with_openai_library_stops`: patched 91 passed, 1 skipped; unpatched 90 passed, 1 skipped (new test deselected). The new test `test_n_probs_token_ids` fails on the unpatched binary (`KeyError: 'token_ids_logprobs'`) and passes on the patched one.
- The deselected test downloads `bartowski/Phi-3.5-mini-instruct-GGUF` (multi-GB, not approved). It fails 3 of 3 on both binaries because the model is absent. Other `tools/server/tests/unit` files were not run.
- A first version of the new test compared values across two requests and failed by 8.3e-4. The second request reused the prompt cache, the effect reported in #28368. The test now compares both lists inside one response, where they are equal.
