#!/usr/bin/env python3
"""Benchmark llamacpp-jev on ZefanCai/Open-Jev: accuracy, calibration (ECE, Brier), per-source.

Open-Jev is a public typed-decision benchmark (state + a choice/score/noul question + a gold
`target` distribution). Its wire shape is TypeSafe Jev's `/v1/systemone`, so this points straight
at a running `llamajev serve`. Calibration is computed from the wrapper's own returned
probabilities (choice/score `probabilities`, noul `noul`), not from any engine-specific field.

Get the data once:  hf download ZefanCai/Open-Jev --repo-type dataset
Then either pass the HF snapshot dir (built into rows here) or a prebuilt rows JSON:

  python3 scripts/openjev_bench.py --openjev-dir ~/.cache/huggingface/.../Open-Jev/snapshots/<rev> \\
      --url http://127.0.0.1:8000 --per-source 40 --out openjev.json --curve pairs.json
  python3 scripts/openjev_bench.py --rows openjev-rows.json --url http://127.0.0.1:8000
"""
import argparse
import glob
import json
import os
import random
import urllib.error
import urllib.request
from collections import defaultdict

ap = argparse.ArgumentParser()
src = ap.add_mutually_exclusive_group(required=True)
src.add_argument("--rows", help="Prebuilt rows JSON (list of {id,source,kind,question,options,target,state_json}).")
src.add_argument("--openjev-dir", help="HF snapshot dir of ZefanCai/Open-Jev; the test split is read via pyarrow.")
ap.add_argument("--url", default="http://127.0.0.1:8000")
ap.add_argument("--model", default="jev-latest")
ap.add_argument("--per-source", type=int, default=40)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--out", help="Write {summary, ...} JSON here.")
ap.add_argument("--curve", help="Write [[confidence, correct], ...] pairs here for a reliability diagram.")
ap.add_argument("--label", default="llamacpp-jev")
args = ap.parse_args()


def load_rows() -> list[dict]:
    if args.rows:
        return json.load(open(os.path.expanduser(args.rows)))
    import pyarrow.parquet as pq  # only needed for the --openjev-dir path

    files = glob.glob(os.path.expanduser(args.openjev_dir) + "/**/test-*.parquet", recursive=True)
    if not files:
        raise SystemExit(f"no test-*.parquet under {args.openjev_dir}")
    cols = ["id", "source", "kind", "question", "options", "target", "state_json"]
    return pq.read_table(files[0], columns=cols).to_pylist()


rng = random.Random(args.seed)
bysrc: dict[str, list] = defaultdict(list)
for r in load_rows():
    bysrc[r["source"]].append(r)
sample = []
for rs in bysrc.values():
    rng.shuffle(rs)
    sample += rs[: args.per_source]
rng.shuffle(sample)
print(f"[{args.label}] {len(sample)} items across {len(bysrc)} sources ({args.per_source}/source)", flush=True)


def parse_opts(kind, options):
    if kind == "noul":
        return None, ["no", "yes"]
    if kind == "score":
        return [str(o) for o in options], [str(i) for i in range(len(options))]
    crit, keys = {}, []
    for o in options:
        k, d = (o.split(": ", 1) if ": " in o else (o, None))
        crit[k] = d
        keys.append(k)
    return crit, keys


def ask(state, kind, question, crit):
    wq = {"type": kind, "instructions": question}
    if crit is not None:
        wq["criteria"] = crit
    body = json.dumps({"model": args.model, "state": state, "questions": {"q": wq}}).encode()
    req = urllib.request.Request(args.url.rstrip("/") + "/v1/systemone", data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return json.load(r)
    except (urllib.error.URLError, OSError, ValueError):
        return None


def maybe_json(s):
    """A state stored as a JSON string is parsed; plain text is returned unchanged."""
    if isinstance(s, str) and s.strip()[:1] in "[{\"":
        try:
            return json.loads(s)
        except (ValueError, TypeError):
            return s
    return s


def ece(pairs, bins=10):
    if not pairs:
        return None
    buckets = [[] for _ in range(bins)]
    for c, ok in pairs:
        buckets[min(bins - 1, int(c * bins))].append((c, ok))
    e = 0.0
    for bk in buckets:
        if bk:
            acc = sum(o for _, o in bk) / len(bk)
            conf = sum(c for c, _ in bk) / len(bk)
            e += len(bk) / len(pairs) * abs(acc - conf)
    return round(e, 4)


correct = n = failed = 0
cal, brier, by_src = [], [], defaultdict(lambda: [0, 0])
for i, r in enumerate(sample):
    kind = r["kind"]
    crit, labels = parse_opts(kind, r["options"])
    state = maybe_json(r["state_json"])
    if not isinstance(state, str):
        state = json.dumps(state, ensure_ascii=False)
    tgt = r["target"]
    exp = labels[max(range(len(tgt)), key=lambda j: tgt[j])]
    resp = ask(state, kind, r["question"], crit)
    ans = resp.get("answers", {}).get("q") if resp else None
    n += 1
    by_src[r["source"]][1] += 1
    if ans is None:
        failed += 1
        continue
    pred = ans.get("choice") if kind == "choice" else ("yes" if ans.get("noul", 0) >= 0.5 else "no") if kind == "noul" else str(ans.get("level"))
    ok = pred == exp
    correct += ok
    by_src[r["source"]][0] += ok
    if kind == "noul":
        pt = ans.get("noul")
        p = {"yes": pt, "no": 1 - pt} if pt is not None else None
    else:
        pr = ans.get("probabilities")
        p = {lb: pr.get(lb, 0.0) for lb in labels} if isinstance(pr, dict) else None
    if p:
        cal.append((max(p.values()), bool(ok)))
        gold = {labels[j]: tgt[j] for j in range(len(labels))}
        brier.append(sum((p.get(lb, 0) - gold.get(lb, 0)) ** 2 for lb in labels))
    if (i + 1) % 40 == 0:
        print(f"  {i+1}/{len(sample)} acc {correct}/{n}", flush=True)

summary = {
    "label": args.label, "n": n, "accuracy": round(correct / n, 3), "failed": failed,
    "calibration_n": len(cal), "ece": ece(cal),
    "brier_vs_gold": round(sum(brier) / len(brier), 4) if brier else None,
    "by_source": {k: {"acc": round(v[0] / v[1], 3), "n": v[1]} for k, v in sorted(by_src.items())},
}
print(json.dumps(summary, indent=1))
if args.out:
    with open(args.out, "w") as f:
        json.dump({"summary": summary}, f, indent=1)
if args.curve:
    with open(args.curve, "w") as f:
        json.dump(cal, f)
