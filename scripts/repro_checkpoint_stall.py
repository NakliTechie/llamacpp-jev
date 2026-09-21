"""Reproduce the llama-server checkpoint stall seen with --cache-ram > 0 (see docs/llama-server-checkpoint-stall.md).

Talks to llama-server directly (no wrapper): for each image, one warm-up call on the prefix, then 4
one-token branches pinned to the warm-up's slot (id_slot). Sequence: one image twice, 8 fresh images,
that image 5x, text-only 3x, the image 5x again. On a call that stalls > 20 s the script runs
/usr/bin/sample on the server pid and writes hang-sample.txt.

Usage: python scripts/repro_checkpoint_stall.py http://127.0.0.1:8093 FRESH_DIR tests/assets/shapes-448.png SERVER_PID OUT_DIR
Server: llama-server -m Qwen3.5-2B-Q8_0.gguf --mmproj mmproj-F16.gguf -np 4 -c 8192 -ngl 99 --cache-ram 4096 \
        --ctx-checkpoints 32 --checkpoint-min-step 0 --reasoning-budget 0
"""
import asyncio
import base64
import subprocess
import sys
import time
from pathlib import Path

import httpx

B, fresh_dir, shapes, pid = sys.argv[1], Path(sys.argv[2]), sys.argv[3], sys.argv[4]
S = Path(sys.argv[5])
async def main():
    async with httpx.AsyncClient(base_url=B, timeout=180) as c:
        async def render(img_b64, text="Look at the image carefully."):
            content = [{"type":"image_url","image_url":{"url":"data:image/png;base64,"+img_b64}},{"type":"text","text":text}] if img_b64 else text
            msgs = [{"role":"user","content":content},{"role":"user","content":"Choose exactly one option and answer with only its label.\n\nMARK"}]
            r = (await c.post("/apply-template", json={"messages": msgs, "chat_template_kwargs": {"enable_thinking": False}})).json()["prompt"]
            return r.split("MARK")
        qs = ["What shape is the red object?", "How many shapes are there?", "Is there a blue square?", "Where is the circle?"]
        async def call(name, prompt, img_b64, **kw):
            body = {"prompt": ({"prompt_string": prompt, "multimodal_data": [img_b64]} if img_b64 else prompt), "n_predict": 1, "temperature": 0, "cache_prompt": True, **kw}
            t0 = time.perf_counter()
            task = asyncio.create_task(c.post("/completion", json=body))
            done, _ = await asyncio.wait({task}, timeout=20)
            if not done:
                print(f"  {name}: STALLED >20s -> sampling pid {pid}", flush=True)
                await asyncio.to_thread(
                    subprocess.run, ["/usr/bin/sample", pid, "3", "-file", str(S / "hang-sample.txt")], capture_output=True, check=False
                )
                r = await task
            else:
                r = task.result()
            j = r.json(); tm = j.get("timings", {})
            print(f"  {name}: slot={j.get('id_slot')} {j.get('content')!r} prompt_n={tm.get('prompt_n')} cache_n={tm.get('cache_n')} {1000*(time.perf_counter()-t0):.0f} ms", flush=True)
            return j.get("id_slot")
        async def evaluate(tag, img_b64):
            prefix, ending = await render(img_b64)
            slot = await call(f"{tag} warm", prefix, img_b64, n_probs=0)
            await asyncio.gather(*[call(f"{tag} b{i}", prefix + f"Question: {q}\n\nOptions:\nA: yes\nB: no\nC: maybe" + ending + "Answer:\n", img_b64, n_probs=8, grammar='root ::= "A" | "B" | "C"', id_slot=slot) for i, q in enumerate(qs)])
        sh = base64.b64encode(Path(shapes).read_bytes()).decode()
        print("phase 0: shapes x2"); [await evaluate("shapes", sh) for _ in range(2)]
        print("phase 1: 8 fresh images")
        for p in sorted(fresh_dir.glob("img*.png")):
            await evaluate(p.name, base64.b64encode(p.read_bytes()).decode())
        print("phase 2: shapes x5"); [await evaluate("shapes", sh) for _ in range(5)]
        print("phase 3: text x3"); [await evaluate("text", None) for _ in range(3)]
        print("phase 4: shapes x5"); [await evaluate("shapes", sh) for _ in range(5)]
        print("DONE")
asyncio.run(main())
