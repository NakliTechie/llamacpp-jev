"""One request per never-seen image: per-image cold latency and correctness against truth.json.

Usage: python scripts/fresh_bench.py URL IMAGE_DIR   (make IMAGE_DIR with scripts/make_shapes.py)
"""

import base64
import json
import statistics
import sys
import time
from pathlib import Path

import httpx

QUESTIONS = {
    "red_shape": {"type": "choice", "instructions": "What shape is the red object?",
                  "criteria": {"circle": None, "square": None, "triangle": None, "none": "There is no red object"}},
    "count": {"type": "choice", "instructions": "How many distinct shapes are in the image?",
              "criteria": {"1": None, "2": None, "3": None, "4": None, "5": None}},
    "blue_square": {"type": "noul", "instructions": "Is there a blue square in the image?"},
    "circle_quadrant": {"type": "choice", "instructions": "In which quadrant of the image is the circle?",
                        "criteria": {"top_left": None, "top_right": None, "bottom_left": None, "bottom_right": None}},
}


def main():
    url, folder = sys.argv[1], Path(sys.argv[2])
    truth = json.loads((folder / "truth.json").read_text())
    walls, prefills, branches, correct, total, failed = [], [], [], 0, 0, 0
    with httpx.Client(base_url=url, timeout=300) as client:
        print("health:", client.get("/health").json())
        for name, gt in truth.items():
            b64 = base64.b64encode((folder / name).read_bytes()).decode()
            state = [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": "Look at the image carefully."},
            ]}]
            t0 = time.perf_counter()
            r = client.post("/v1/systemone", json={"model": "jev-latest", "state": state, "questions": QUESTIONS})
            wall = (time.perf_counter() - t0) * 1000
            if r.status_code != 200:
                print(f"{name}: ERROR {r.status_code} {r.text}")
                failed += 1
                total += len(QUESTIONS)
                continue
            timing = dict(p.strip().split(";dur=") for p in r.headers["server-timing"].split(","))
            a = r.json()["answers"]
            got = {"red_shape": a["red_shape"]["choice"], "count": a["count"]["choice"],
                   "blue_square": a["blue_square"]["noul"] > 0.5, "circle_quadrant": a["circle_quadrant"]["choice"]}
            ok = {k: got[k] == gt[k] for k in got}
            correct += sum(ok.values())
            total += len(ok)
            walls.append(wall)
            prefills.append(float(timing["prefill"]))
            branches.append(float(timing["branches"]))
            print(f"{name}: wall={wall:7.1f} ms prefill={float(timing['prefill']):6.1f} branches={float(timing['branches']):7.1f} "
                  f"cached={r.headers['x-llamajev-cached-tokens']:>5} | {sum(ok.values())}/4 correct | miss={[k for k, v in ok.items() if not v]}")
    if walls:
        print(f"\nfresh-image median wall={statistics.median(walls):.0f} ms (min {min(walls):.0f}, max {max(walls):.0f}); "
              f"median prefill={statistics.median(prefills):.0f}; median branches={statistics.median(branches):.0f}; "
              f"accuracy {correct}/{total}; failed requests {failed}/{len(truth)}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
