"""End-to-end against a real llamajev server. Set LLAMAJEV_LIVE_URL to enable."""

import os

import httpx
import pytest

URL = os.environ.get("LLAMAJEV_LIVE_URL")
pytestmark = pytest.mark.skipif(not URL, reason="LLAMAJEV_LIVE_URL not set")


@pytest.fixture(scope="module")
def client():
    with httpx.Client(base_url=URL, timeout=120) as c:
        assert c.get("/health").status_code == 200
        yield c


def ask(client, state, questions):
    r = client.post("/v1/systemone", json={"model": "jev-latest", "state": state, "questions": questions})
    assert r.status_code == 200, r.text
    return r.json()["answers"], r.headers


def test_unambiguous_facts(client):
    answers, headers = ask(
        client,
        "The sky is blue. Paris is the capital of France. The number seven is odd.",
        {
            "paris": {"type": "choice", "instructions": "What is the capital of France?", "criteria": {"berlin": "Berlin", "paris": "Paris", "rome": "Rome"}},
            "odd": {"type": "noul", "instructions": "Is seven an odd number?"},
            "sky": {"type": "choice", "instructions": "What colour is the sky, per the text?", "criteria": {"red": None, "blue": None, "green": None}},
        },
    )
    assert answers["paris"]["choice"] == "paris"
    assert answers["odd"]["noul"] > 0.5
    assert answers["sky"]["choice"] == "blue"
    assert int(headers["x-llamajev-cached-tokens"]) > 0


def test_score_monotone(client):
    rubric = ["Cosmetic", "Degraded but workaround exists", "Blocking, no workaround"]
    low, _ = ask(client, "The export button is misaligned by a few pixels.", {"s": {"type": "score", "instructions": "How severe is the reported issue?", "criteria": rubric}})
    high, _ = ask(client, "Every export fails and nobody can get their data out. Production is down.", {"s": {"type": "score", "instructions": "How severe is the reported issue?", "criteria": rubric}})
    assert low["s"]["score"] < high["s"]["score"]


def test_twenty_six_way(client):
    criteria = {f"opt{i}": f"The number {i}" for i in range(26)}
    criteria["opt20"] = "The answer to life, the universe and everything"
    answers, _ = ask(client, "Douglas Adams said the answer is forty-two.", {"q": {"type": "choice", "instructions": "Which option matches the text?", "criteria": criteria}})
    assert answers["q"]["choice"] == "opt20"


def test_sixty_four_way_is_well_formed(client):
    """Structural only: Qwen3.5-0.8B/2B collapse onto one label once two-letter labels appear
    (observed 2026-09-21: 'Q' chosen regardless of the correct position). The contract still
    accepts 64 options; correctness at that width needs a larger model."""
    criteria = {f"opt{i}": f"The number {i}" for i in range(64)}
    criteria["opt42"] = "The answer to life, the universe and everything"
    answers, headers = ask(client, "Douglas Adams said the answer is forty-two.", {"q": {"type": "choice", "instructions": "Which option matches the text?", "criteria": criteria}})
    probs = answers["q"]["probabilities"]
    assert len(probs) == 64 and abs(sum(probs.values()) - 1) < 1e-6
    # Escalation model (f0bd69c): all 64 labels are in the readout, so retries is a
    # non-negative count; an exhausted readout would have 502'd, not returned 200 here.
    assert int(headers["x-llamajev-readout-retries"]) >= 0


def test_vision_shapes(client):
    """Batch C shape: one 448x448 image, four typed questions. Skips on a text-only backend."""
    import base64
    from pathlib import Path

    if not client.get("/health").json().get("vision"):
        pytest.skip("backend has no vision modality (start llama-server with --mmproj)")
    image = Path(__file__).parent / "assets" / "shapes-448.png"
    b64 = base64.b64encode(image.read_bytes()).decode()
    state = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        {"type": "text", "text": "Look at the image carefully."},
    ]}]
    answers, headers = ask(client, state, {
        "red_shape": {"type": "choice", "instructions": "What shape is the red object?", "criteria": {"circle": None, "square": None, "triangle": None, "none": "There is no red object"}},
        "count": {"type": "choice", "instructions": "How many distinct shapes are in the image?", "criteria": {"1": None, "2": None, "3": None, "4": None, "5": None}},
        "blue_square": {"type": "noul", "instructions": "Is there a blue square in the image?"},
        "circle_quadrant": {"type": "choice", "instructions": "In which quadrant of the image is the circle?", "criteria": {"top_left": None, "top_right": None, "bottom_left": None, "bottom_right": None}},
    })
    assert answers["red_shape"]["choice"] == "circle"
    assert answers["count"]["choice"] == "3"
    assert answers["blue_square"]["noul"] > 0.5
    assert answers["circle_quadrant"]["choice"] == "top_left"
    assert int(headers["x-llamajev-prefix-tokens"]) > 200  # image tokens are in the prefix
