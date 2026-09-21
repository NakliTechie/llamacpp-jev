import math

import httpx
import pytest

from llamajev.api import create_app
from llamajev.backend import Generation, Props
from llamajev.config import Settings
from llamajev.prompts import PromptCompiler, label_candidates
from llamajev.service import EvaluationService

ENDING = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def fake_labels():
    return [(label, 1000 + i) for i, label in zip(range(64), label_candidates())]


class FakeBackend:
    """Stands in for LlamaClient: records calls, returns deterministic logprobs."""

    def __init__(self, vision: bool = False):
        self.calls: list[dict] = []
        self.vision = vision
        self.healthy = True
        self.readout_depth: int | None = None  # cap on returned entries, to simulate truncation

    async def health(self):
        return self.healthy

    async def apply_template(self, messages):
        self.calls.append({"apply_template": messages})
        parts = []
        for m in messages:
            content = m["content"]
            if isinstance(content, list):
                content = "".join(
                    p["text"] if p["type"] == "text" else "<__media__>" for p in content
                )
            parts.append(f"<|im_start|>{m['role']}\n{content}<|im_end|>\n")
        return "".join(parts)[: -len("<|im_end|>\n")] + ENDING

    async def complete(self, prompt, *, images=None, n_probs=0, grammar=None, id_slot=None):
        self.calls.append({"prompt": prompt, "images": images, "n_probs": n_probs, "grammar": grammar, "id_slot": id_slot})
        if n_probs == 0:
            return Generation(sampled="Question", prompt_tokens=50, cached_tokens=0, id_slot=2)
        labels = fake_labels()
        # label i gets logprob -i: A most likely, then B, C ...
        depth = min(n_probs, self.readout_depth or n_probs)
        logprobs = {token_id: -float(i) for i, (_, token_id) in enumerate(labels[:depth])}
        return Generation(sampled="A", logprobs=logprobs, prompt_tokens=30, cached_tokens=50, id_slot=id_slot)


@pytest.fixture
def backend():
    return FakeBackend()


@pytest.fixture
def settings():
    return Settings(top_n=256)


@pytest.fixture
async def api(backend, settings):
    props = Props(model_path="/models/Qwen3.5-2B-Q8_0.gguf", n_ctx=8192, n_slots=4, vision=backend.vision, media_marker="<__media__>")
    service = EvaluationService(settings, PromptCompiler(fake_labels()), backend, props)
    app = create_app(settings, service=service)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        client.app = app
        async with app.router.lifespan_context(app):
            yield client, backend


@pytest.fixture
def payload():
    return {
        "model": "jev-latest",
        "state": [
            {"role": "system", "content": "You are a support assistant."},
            {"role": "user", "content": "I was charged twice."},
        ],
        "questions": {
            "yes": {"type": "noul", "instructions": "Refund requested?"},
            "team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Payments", "tech": None}},
            "level": {"type": "score", "instructions": "How urgent?", "criteria": ["Low", "Medium", "High"]},
        },
    }


def sigmoid(x):
    return 1 / (1 + math.exp(-x))
