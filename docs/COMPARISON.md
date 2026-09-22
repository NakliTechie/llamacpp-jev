# How llamacpp-jev compares

Where a local llama.cpp logprob wrapper sits among the tools people reach for when they want a
*typed decision* out of an LLM. Vendor performance/calibration numbers below are **their claims**;
no independent benchmark of the hosted products was found, and they are marked as such.

## The distinction that splits the field

Most tools **force valid structure** — a grammar mask that zeroes out illegal tokens so the output
is one of your labels. That is a different product from **returning a normalized probability
distribution over a closed option set**. A grammar guarantees a legal answer; it does not hand you
`P(yes)=0.73`. Getting the distribution means reading logprobs over the label tokens and
normalizing — which is what llamacpp-jev does and what most constrained-decoding libraries leave to
you. A second axis: **normalized ≠ calibrated** (see the honest gap below).

## The alternatives

| Tool / class | Local | Distribution over options | Any open model | Vision | Why pick it over llamacpp-jev |
|---|---|---|---|---|---|
| **Constrained decoding** — Outlines, Guidance, XGrammar, lm-format-enforcer, jsonformer | yes | no (forces structure; logprobs are DIY) | yes | via the model only | rich JSON/CFG/regex and control flow, not a single typed decision |
| **Structured-output SDKs** — Instructor, BAML | mixed | no (validated objects) | yes | via provider | Pydantic/contract-first typed clients, retries, broad provider matrix |
| **OpenAI structured outputs / Anthropic tool use** | no | OpenAI: via separate logprobs API; Anthropic: none | no | yes | you are already on that provider |
| **Verdict** (Haize Labs) | model-agnostic | yes (judgment distribution over scales) | yes | not documented | composable multi-unit judge protocols, inference-time scaling |
| **TypeSafe Jev** (`api.typesafe.ai/v1/systemone`) | **no (hosted)** | yes, **calibrated** (claimed, RLCD) | no (bespoke model) | not documented | trained calibration + speed at scale, if a proprietary hosted dependency is acceptable |
| **OpenJev** (`zhangcy122/OpenJev`) | yes | yes + calibration + abstention | yes | not documented | the same idea with a calibration/abstention layer, if you run vLLM/SGLang |
| **DIY llama.cpp** — GBNF + `n_probs` | yes | yes (you normalize) | yes | via the model | you want to hand-roll it; llamacpp-jev is this, packaged |

Sources: [TypeSafe Jev — Tom's Hardware](https://www.tomshardware.com/tech-industry/artificial-intelligence/typesafe-ais-jev-offers-an-alternative-to-llms-that-claims-to-be-193x-faster-and-445x-cheaper-system-one-type-model-is-bespoke-for-probabilistic-decision-making), [DataCamp](https://www.datacamp.com/blog/system-one-models-jev); [OpenJev](https://github.com/zhangcy122/OpenJev); [Verdict](https://github.com/haizelabs/verdict) + [judgment-distribution paper](https://arxiv.org/abs/2503.03064); [XGrammar](https://blog.mlc.ai/2024/11/22/achieving-efficient-flexible-portable-structured-generation-with-xgrammar), [Outlines](https://github.com/dottxt-ai/outlines), [lm-format-enforcer](https://github.com/noamgat/lm-format-enforcer); [llama.cpp server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md). TypeSafe Jev's "40–200× / 96.3%" and OpenJev's parity claims are vendor/author numbers, independently unverified.

## The niche

No single competitor holds all of this at once:

1. **Sovereign, zero-infra, any GGUF** — local, no engine fork, no API key, on plain `llama-server`
   (CPU / Metal / consumer GPU). TypeSafe Jev is hosted; OpenJev and the vLLM-backed libraries want
   a vLLM/SGLang stack.
2. **The distribution is the output** — not a validated object you post-process, and not a structure
   mask you reconstruct probabilities from.
3. **Text and vision in one decision API** on local open VLMs — TypeSafe Jev, OpenJev, and Verdict
   do not document vision.

## The honest gap: normalized ≠ calibrated

llamacpp-jev returns *normalized* logprobs; TypeSafe Jev (RLCD) and OpenJev (temperature/Platt +
abstention) *calibrate*. That is the one axis where a raw wrapper is behind. Our own measurement
([docs/BENCHMARK.md](BENCHMARK.md)) shows it is largely a small-model effect: **Qwen3.5-4B is already
~calibrated on in-domain routing** (ECE 0.054, recalibration gain 0.019), while 2B is overconfident
(gain 0.155) and both stay overconfident on the harder out-of-domain mix. A one-parameter
temperature step at serve time would close it — the obvious roadmap item, not a claim made today.
