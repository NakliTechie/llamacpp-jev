"""Latency benchmark for a llamajev server: N repetitions of one request, median Server-Timing.

Usage: python scripts/bench.py [URL] [--image PATH] [--runs N]
With --image, the state is a chat message carrying the image (data: URI) plus a caption, and the
four questions match the @kis tweet's demo shape (4 typed questions about one 448x448 image).
"""

import argparse
import base64
import json
import mimetypes
import statistics
import sys
import time

import httpx

TEXT_STATE = [
    {"role": "system", "content": "You are a support assistant."},
    {"role": "user", "content": "I was charged twice for order 8841. Please refund the duplicate today."},
]
TEXT_QUESTIONS = {
    "refund": {"type": "noul", "instructions": "Does the user request a refund?"},
    "department": {"type": "choice", "instructions": "Which department should handle this?",
                   "criteria": {"billing": "Payments and refunds", "technical": "Software bugs", "shipping": "Delivery"}},
    "urgency": {"type": "score", "instructions": "How urgent is the request?", "criteria": ["Routine", "Urgent", "Emergency"]},
    "order": {"type": "noul", "instructions": "Does the message mention an order number?"},
}
IMAGE_QUESTIONS = {
    "red_shape": {"type": "choice", "instructions": "What shape is the red object?",
                  "criteria": {"circle": None, "square": None, "triangle": None, "none": "There is no red object"}},
    "count": {"type": "choice", "instructions": "How many distinct shapes are in the image?",
              "criteria": {"1": None, "2": None, "3": None, "4": None, "5": None}},
    "blue_square": {"type": "noul", "instructions": "Is there a blue square in the image?"},
    "circle_quadrant": {"type": "choice", "instructions": "In which quadrant of the image is the circle?",
                        "criteria": {"top_left": None, "top_right": None, "bottom_left": None, "bottom_right": None}},
}


def parse_timing(header: str) -> dict[str, float]:
    out = {}
    for part in header.split(","):
        name, _, dur = part.strip().partition(";dur=")
        out[name] = float(dur)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url", nargs="?", default="http://127.0.0.1:8000")
    ap.add_argument("--image")
    ap.add_argument("--runs", type=int, default=7)
    args = ap.parse_args()

    if args.image:
        mime = mimetypes.guess_type(args.image)[0] or "image/png"
        with open(args.image, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        state = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            {"type": "text", "text": "Look at the image carefully."},
        ]}]
        questions = IMAGE_QUESTIONS
    else:
        state, questions = TEXT_STATE, TEXT_QUESTIONS
    payload = {"model": "jev-latest", "state": state, "questions": questions}

    with httpx.Client(base_url=args.url, timeout=300) as client:
        health = client.get("/health").json()
        print("health:", json.dumps(health))
        rows = []
        last = None
        for i in range(args.runs):
            t0 = time.perf_counter()
            r = client.post("/v1/systemone", json=payload)
            wall = (time.perf_counter() - t0) * 1000
            if r.status_code != 200:
                print("ERROR", r.status_code, r.text)
                sys.exit(1)
            t = parse_timing(r.headers["server-timing"])
            rows.append((wall, t))
            last = r
            print(f"run {i}: wall={wall:7.1f} ms  prepare={t['prepare']:6.1f}  prefill={t['prefill']:7.1f}  branches={t['branches']:7.1f}"
                  f"  prefix_tokens={r.headers['x-llamajev-prefix-tokens']} cached={r.headers['x-llamajev-cached-tokens']} retries={r.headers['x-llamajev-readout-retries']}")
        warm = rows[1:] if len(rows) > 1 else rows

        def med(k):
            return statistics.median(t[k] for _, t in warm)

        print(f"\nmedian over runs 1..{len(rows)-1} (run 0 = cold): wall={statistics.median(w for w,_ in warm):.1f} ms  "
              f"prepare={med('prepare'):.1f}  prefill={med('prefill'):.1f}  branches={med('branches'):.1f}")
        print("answers:", json.dumps(last.json()["answers"], indent=1))


if __name__ == "__main__":
    main()
