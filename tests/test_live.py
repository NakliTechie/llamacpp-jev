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
    assert int(headers["x-llamajev-truncated-labels"]) <= 64
