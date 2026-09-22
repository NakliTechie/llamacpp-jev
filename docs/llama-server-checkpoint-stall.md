# llama-server stalls for minutes inside `create_checkpoint` with `--cache-ram > 0` on a hybrid model with image prompts

Status 2026-09-22: reproduced 5 times on 2026-09-21 (4 through this wrapper, 1 standalone) while the
machine was serving another GPU process; stack sample captured. On 2026-09-22, with the machine
otherwise idle, the same sequence ran 4 of 4 times to completion with no stall (slowest call 298 ms).
The stall appears **load-associated**, but this is not established as causation: the two conditions
also differ in day, binary (2026-09-22 adds instrumentation), thermal state, and initial cache
state, and the 5 earlier failures were not 5 identical trials. Instrumentation (§4) narrows the
cause — **no fragmentation was observed** — but does **not** resolve it: no stall was ever captured
with the instrumentation attached, so the behaviour during a stall is still unmeasured. Not filed
upstream.

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
checkpoint buffer). Caveat: an aggregated `_platform_memmove` total cannot by itself distinguish a
few large copies from many copies through the same loop; the count and size distribution of the
individual tensor copies during a stall is what would settle that, and was not captured.

### Instrumented run (2026-09-22, machine idle, `-lv 4`) — no stall reproduced

`-lv 4` (not `-lv 3`) surfaces the server's own `created context checkpoint … size = %.3f MiB` line
(`SLT_TRC`, gated on `LOG_LEVEL_TRACE = 4`). Across a full repro sequence with `--cache-ram 4096`
(which did **not** stall):

- 201 checkpoints created, each **19.266 MiB** for a ~256-cell sequence; ~8–9 checkpoints per
  evaluation, driven by `--checkpoint-min-step 0`. Note 19.266 MiB is the *destination buffer size*
  reported by `cur.size()`, not evidence of a single contiguous transfer — the recurrent writer
  emits a separate tensor copy per layer.
- A one-line patch in `llama_kv_cache::state_write` and `llama_memory_recurrent::state_write` logged
  the cell-range counts. **No fragmentation was observed in these runs**: every serialization used a
  single cell range (KV path all `cell_ranges = 1`; recurrent path all `cell_count = 1,
  cell_ranges = 1`). This is a property of the non-stalling runs only; range counts during an actual
  stall were not captured.
- The instrument fires on both serialization passes — `state_seq_get_size`
  (`llama_io_write_dummy`, size estimation) and `state_seq_get_data` (`llama_io_write_host`, the real
  write) both call `state_write` (`src/llama-context.cpp:3096,3113,3348`). The raw 424 (KV) / 826
  (recurrent) line counts therefore mix both passes and are not readback counts; only the host pass
  performs `ggml_backend_tensor_get`.
- Checkpoints are created at a similar rate with `--cache-ram 0` (207) and `--cache-ram 4096` (201).
- Every call completed in tens to low-hundreds of ms; the stall did not occur in 4 of 4 runs, so
  per-readback timing under stall conditions remains unmeasured.

## 4. What the instrumentation settled, and what it did not

**Fragmentation was not observed** (the original primary hypothesis is unsupported, not proven
false). That hypothesis was that `--cache-ram`'s save/load/`seq_rm` cycles fragment a sequence's
cells so `state_write_data` emits thousands of tiny `write_tensor`/`ggml_backend_tensor_get` calls.
In every instrumented run every serialization used a single cell range (`cell_ranges = 1`, KV and
recurrent). But all instrumented runs were on an idle machine and none stalled, so this rules out
fragmentation *in the non-stalling case*; the cell-range count during an actual stall was never
captured.

**The expensive operation, and why it is not obviously GPU-bound.** Checkpoint creation runs
`state_seq_get_data` → `~llama_io_write_host`, whose destructor replays each `write_tensor` as a
`ggml_backend_tensor_get` (`src/llama-context.cpp:2606-2615`). On Apple-silicon unified memory the
tensor buffers are shared, so `ggml_metal_buffer_get_tensor` takes the `if (buf->is_shared) memcpy`
path and returns without touching the Metal command queue
(`ggml/src/ggml-metal/ggml-metal-device.m:2377`). So the earlier "Metal command-queue contention"
explanation is wrong: the readbacks are plain host `memcpy`s (plus a `bzero` of the destination),
which is what the stack sample shows. Why those `memcpy`/`bzero` operations would block for minutes
under concurrent load is **not established** — memory-bandwidth or allocator contention is a
plausible but unmeasured hypothesis.

**Frequency is real but its effect is unquantified.** Checkpoints are created ~8–9 times per
evaluation (`--checkpoint-min-step 0`, `--ctx-checkpoints 32`); `state_write` is also invoked twice
per serialization (a size-estimation `llama_io_write_dummy` pass and the real `llama_io_write_host`
pass), and `--cache-ram` adds prompt-cache serializations. The raw instrument line counts conflate
these, so no traffic multiplier is claimed here.

**Candidate fix surface (a hypothesis to test, not a validated remedy).** The standing
`TODO: add backend support to batch tensor_get?` at `src/llama-context.cpp:2610` means
`~llama_io_write_host` issues one blocking `ggml_backend_tensor_get` per tensor with no batching.
Batching the readbacks, or making checkpoint creation yield instead of blocking the slot's decode
loop, are the natural things to try — but each should be measured against a captured stall before
being asserted as the fix.

**To actually resolve it:** capture a stall *with* the instrumentation attached, and record
per-tensor copy count, sizes, and durations plus thermal and memory-pressure state during the hang.

Attempt (2026-09-22): re-ran the reproducer on the idle machine while a second `llama-server`
(same 2B model) ran continuous generation on the same GPU — 4 streams × 512 tokens, then 8 streams
× 1024 tokens. The reproducer slowed (worst call 199 ms idle → 413 ms / 321 ms under load) but did
**not** stall. Total this day: 6 reproducer runs, 0 stalls. So the stall does not reproduce under
synthetic same-model inference contention; the 2026-09-21 condition ("another process needed the
GPU") was a different, unidentified workload that has not been recreated. Recreating it precisely —
knowing what that workload was — is the missing input for a confirmed mechanism.

## 5. Reproduce

```bash
python scripts/make_shapes.py /tmp/fresh 8
llama-server -m Qwen3.5-2B-Q8_0.gguf --mmproj mmproj-F16.gguf -np 4 -c 8192 -ngl 99 --cache-ram 4096 \
  --ctx-checkpoints 32 --checkpoint-min-step 0 --reasoning-budget 0 --port 8093 -lv 3 --log-file server.log &
python scripts/repro_checkpoint_stall.py http://127.0.0.1:8093 /tmp/fresh tests/assets/shapes-448.png $! /tmp
```

Output of the run that produced the sample: `docs/evidence/checkpoint-stall-repro-2026-09-21.out`.
