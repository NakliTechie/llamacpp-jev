# llama-server stalls for minutes inside `create_checkpoint` with `--cache-ram > 0` on a hybrid model with image prompts

Status 2026-09-22: reproduced 5 of 5 on 2026-09-21 (4 through this wrapper, 1 standalone) while the
GPU was serving another process; stack sample captured. On 2026-09-22, with the GPU otherwise idle,
the same sequence ran 4 of 4 times to completion with no stall (slowest call 298 ms). The stall is
**load-dependent**, not deterministic. Root cause resolved by instrumentation (§4): **not
fragmentation** — the checkpoint is a small number of large contiguous synchronous Metal readbacks,
and their volume (frequency × size, roughly doubled by `--cache-ram`) blocks for minutes only when a
concurrent GPU workload contends for the Metal command queue. Not yet filed upstream.

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
checkpoint buffer). So the time goes to copying and zeroing checkpoint memory. That 959 samples sit
in **one** `_platform_memmove` region (not spread across many call sites) already argues for a few
large copies rather than thousands of tiny ones — confirmed by the instrumentation below.

### Instrumented run (2026-09-22, GPU idle, `-lv 4`)

`-lv 4` (not `-lv 3`) surfaces the server's own `created context checkpoint … size = %.3f MiB` line
(`SLT_TRC`, gated on `LOG_LEVEL_TRACE = 4`). Across a full repro sequence with `--cache-ram 4096`:

- 201 checkpoints created, each **19.266 MiB** (≈ 88 KiB per cell for a ~256-cell sequence); ~8–9
  checkpoints per evaluation, driven by `--checkpoint-min-step 0`.
- A one-line patch in `llama_kv_cache::state_write` and `llama_memory_recurrent::state_write` logged
  the cell-range counts. **Every serialization was contiguous**: KV path 424 calls, all
  `cell_ranges = 1`; recurrent path 826 calls, all `cell_count = 1, cell_ranges = 1`. No sequence
  was ever split into multiple ranges.
- Checkpoints are created at the same rate with `--cache-ram 0` (207) as with `--cache-ram 4096`
  (201). What `--cache-ram` adds is the RAM-prompt-cache serialization traffic (the 424 KV
  `state_write` calls above, which fire only with the cache on) layered on top of the checkpoints.
- On the idle GPU each of these readbacks costs tens of ms and the whole sequence completes; the
  stall did not occur in 4 of 4 runs.

## 4. Resolved cause (instrumented 2026-09-22)

**Fragmentation is ruled out.** The original primary hypothesis was that `--cache-ram`'s repeated
save/load/`seq_rm` cycles fragment a sequence's cells so that `state_write_data` emits thousands of
tiny `write_tensor`/`ggml_backend_tensor_get` calls. The instrumentation refutes it: every
serialization of every sequence was a single contiguous range (`cell_ranges = 1`, KV and recurrent
alike, 1250 samples). The cells never fragment. This is consistent with the stack sample, where the
readback time collapses into one large `_platform_memmove` region rather than many small call sites.

**The cost is the volume of large contiguous synchronous readbacks, not their shape.** Each
checkpoint serialises one contiguous ~19 MiB block (`state_seq_get_data` → `~llama_io_write_host`
replaying `write_tensor` as per-tensor `ggml_backend_tensor_get`, a synchronous `memcpy` from the
shared Metal buffer, `src/llama-context.cpp:2611-2616`). Checkpoints are created ~8–9 times per
evaluation (`--checkpoint-min-step 0`, `--ctx-checkpoints 32`), and with `--cache-ram` the RAM
prompt cache adds a second stream of the same full-sequence readbacks. The per-readback cost is
tens of ms when the Metal command queue is idle, so on a free GPU the run completes (4 of 4). Under
a concurrent GPU workload (the 2026-09-21 condition), these synchronous readbacks serialise behind
the other process's Metal work and each checkpoint blocks for many seconds; with ~8–9 per evaluation
plus the doubled `--cache-ram` traffic, an evaluation stalls for minutes. `--cache-ram 0` avoids it
by removing the extra readback stream, not by keeping cells contiguous (they were always
contiguous).

**Why it looks like a hybrid-model problem.** Qwen3.5's gated-delta layers need context checkpoints
to rewind, so this path is exercised heavily; a pure-attention model with the same request shape
creates far fewer checkpoints. The underlying issue — `TODO: add backend support to batch
tensor_get?` at `src/llama-context.cpp:2610` — is that `~llama_io_write_host` issues one blocking
`ggml_backend_tensor_get` per tensor with no batching or async path, so checkpoint serialisation is
serially latency-bound on the backend queue. That is the fix surface: batch the readbacks, or make
checkpoint creation yield instead of blocking the slot's decode loop.

## 5. Reproduce

```bash
python scripts/make_shapes.py /tmp/fresh 8
llama-server -m Qwen3.5-2B-Q8_0.gguf --mmproj mmproj-F16.gguf -np 4 -c 8192 -ngl 99 --cache-ram 4096 \
  --ctx-checkpoints 32 --checkpoint-min-step 0 --reasoning-budget 0 --port 8093 -lv 3 --log-file server.log &
python scripts/repro_checkpoint_stall.py http://127.0.0.1:8093 /tmp/fresh tests/assets/shapes-448.png $! /tmp
```

Output of the run that produced the sample: `docs/evidence/checkpoint-stall-repro-2026-09-21.out`.
