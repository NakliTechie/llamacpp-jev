# llama-server stalls for minutes inside `create_checkpoint` with `--cache-ram > 0` on a hybrid model with image prompts

Status 2026-09-21: reproduced 4 of 4 times through this wrapper and 1 of 1 with the standalone
script below; stack sample captured. Not yet filed upstream. Root cause is a hypothesis (§4).

## 1. Setup

- llama.cpp master `3d82ef62` (b11063, 2026-09-20), macOS 25.5 / Apple M4 Pro 24 GB, Metal build
  (`-DGGML_METAL=ON`).
- Model: `unsloth/Qwen3.5-2B-GGUF` `Qwen3.5-2B-Q8_0.gguf` + `mmproj-F16.gguf` (Qwen3.5 is a hybrid:
  gated-delta linear attention with full attention every 4th layer, so it uses recurrent memory +
  KV cache and relies on context checkpoints to rewind).
- Server: `llama-server -m … --mmproj … -np 4 -c 8192 -ngl 99 --cache-ram 4096 --ctx-checkpoints 32 --checkpoint-min-step 0 --reasoning-budget 0`
- Does **not** happen with `--cache-ram 0` (13 of 13 requests of the same shape completed).

## 2. Request pattern

For each "evaluation": one `/completion` call on a shared prefix that contains one 448×448 image
(`multimodal_data`, ≈196 image tokens, 245 prefix tokens total, `n_predict: 1`), then four
`/completion` calls `prefix + ~35-token suffix`, `n_predict: 1`, sent concurrently with
`id_slot` pinned to the slot the warm-up call reported, `cache_prompt: true`.

Sequence that triggers it (`scripts/repro_checkpoint_stall.py`): image A ×2 · 8 different images
×1 · image A ×5 · text-only ×3 · image A ×5. The stall hits on a repeat of image A after the
slots and the RAM prompt cache have been populated with other states — in the standalone run on
the second evaluation of the last phase; through the wrapper on the first or second repeat after
the fresh-image sweep. A fresh server running only "image A ×3 with 4 pinned branches" does not
stall.

## 3. Symptom

The warm-up call on the repeated image hits the cache (`selected slot by LCP similarity,
f_sim_best = 1.000`, 4 tokens processed). The first pinned branch launches
(`launch_slot_: id 2 | task 280 | processing task`) and produces nothing for > 60 s (the wrapper's
timeout; 180 s in the standalone run). The other three branches queue behind it
(`selected slot by id (2)` ×3). After the client cancels, the next task on that slot runs normally.
Earlier in the same sequence the same path shows progressive slowdown before it "hangs":
`prompt eval time = 2888 ms / 51 tokens` (task 160), `550 ms / 4 tokens` (task 158).

Stack sample of the server process during the stall (`/usr/bin/sample`, 2 s, 1255 samples of the
main thread; full file `docs/evidence/checkpoint-stall-sample-2026-09-21.txt`):

```
1255 llama_server → server_queue::start_loop
 1249  server_context_impl::update_slots
 1248   server_context_impl::pre_decode
 1245    pre_decode()::lambda3 (per-slot prompt processing)
 1243     server_context_impl::create_checkpoint
  968      common_prompt_checkpoint::update_tgt
  961       llama_context::state_seq_get_data
  960        llama_io_write_host::~llama_io_write_host      ← ggml_backend_tensor_get loop
  273      common_prompt_checkpoint::update_tgt (+176)       ← llama_state_seq_get_size path
```

The main loop is inside `create_checkpoint()` serialising the slot's memory state. At the leaf,
959 of the 1255 samples are in `_platform_memmove` under `~llama_io_write_host()` (which replays
every `write_tensor()` as a `ggml_backend_tensor_get()`, a `memcpy` from the shared Metal buffer —
`src/llama-context.cpp:2611-2616`) and 273 in `__bzero` under `update_tgt` (zeroing the
checkpoint buffer). So the time goes to copying and zeroing checkpoint memory. Whether the loop
would eventually finish was not established: the 180 s client timeout fired first every time.

## 4. Hypotheses (neither verified)

`llama_kv_cache::state_write_data` emits one `write_tensor` per contiguous cell range per layer per
K/V (`src/llama-kv-cache.cpp:2263-2328`), and `llama_memory_recurrent::state_write` one per range
too (`src/llama-memory-recurrent.cpp:769-844`). With `--cache-ram`, slot states are repeatedly
saved to and loaded from the RAM prompt cache and partially removed (`seq_rm` of branch suffixes),
which fragments the sequence's cells. A fragmented 245-cell sequence across the full-attention
layers turns into thousands of tiny `ggml_backend_tensor_get` calls, each a synchronous Metal
readback, inside every checkpoint creation — and checkpoints are created before every batch that
starts a user message. That matches the progressive slowdown and the disappearance of the problem
with `--cache-ram 0` (no cross-slot state traffic, cells stay contiguous). Media chunks may
contribute (the image's 196 cells are written like any other), but the same prompt shape without
the RAM cache is fine, so they are not sufficient on their own.

**Alternative: oversized or repeated checkpoints.** The `memmove`/`bzero` profile is equally
consistent with each checkpoint simply being very large (for example the whole cache rather than
one sequence's cells) or being created far more often than expected, so that a few copies of
gigabytes dominate. Fragmentation would show as many small copies; this would show as few large
ones. The sample cannot tell them apart.

What would settle it: run with a verbosity that prints the server's own
`created context checkpoint … size = … MiB` line (`-lv 3` did not), and log `cr.data.size()`
(number of cell ranges) and `winfos.size()` (number of tensor copies) plus the byte total per
checkpoint. Then repeat controlled cache-on/off runs from fresh servers so the two configurations
differ only in `--cache-ram`.

## 5. Reproduce

```bash
python scripts/make_shapes.py /tmp/fresh 8
llama-server -m Qwen3.5-2B-Q8_0.gguf --mmproj mmproj-F16.gguf -np 4 -c 8192 -ngl 99 --cache-ram 4096 \
  --ctx-checkpoints 32 --checkpoint-min-step 0 --reasoning-budget 0 --port 8093 -lv 3 --log-file server.log &
python scripts/repro_checkpoint_stall.py http://127.0.0.1:8093 /tmp/fresh tests/assets/shapes-448.png $! /tmp
```

Output of the run that produced the sample: `docs/evidence/checkpoint-stall-repro-2026-09-21.out`.
