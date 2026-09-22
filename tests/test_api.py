
import httpx as _httpx
import pytest
from conftest import FakeBackend, fake_labels, sigmoid

from llamajev.api import create_app
from llamajev.backend import Props
from llamajev.config import Settings
from llamajev.prompts import PromptCompiler
from llamajev.service import EvaluationService


async def test_request_prefills_then_branches(api, payload):
    client, backend = api
    response = await client.post("/v1/systemone", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    calls = [c for c in backend.calls if "prompt" in c]
    assert len(calls) == 4  # warm-up + 3 branches
    prefix = calls[0]["prompt"]
    assert calls[0]["n_probs"] == 0 and calls[0]["grammar"] is None
    assert calls[0]["id_slot"] == 0  # leased slot, warm-up pinned too
    for branch in calls[1:]:
        assert branch["id_slot"] == 0
        assert branch["prompt"].startswith(prefix)
        assert branch["prompt"].endswith("Answer:\n")
        assert branch["n_probs"] == 256
        assert branch["grammar"].startswith('root ::= "A" | "B"')
    assert data["model"] == "jev-latest"
    assert data["usage"] == {"input_tokens": 50 + 3 * 80, "output_tokens": 4}
    # A (true) has logprob 0, B (false) has -1
    assert data["answers"]["yes"]["noul"] == pytest.approx(sigmoid(1))
    team = data["answers"]["team"]
    assert team["choice"] == "billing"
    assert sum(team["probabilities"].values()) == pytest.approx(1)
    score = data["answers"]["level"]
    assert score["legend"] == {"0": "Low", "1": "Medium", "2": "High"}
    assert score["score"] == pytest.approx(sum(int(i) * p for i, p in score["probabilities"].items()))
    assert response.headers["x-llamajev-cached-tokens"] == str(3 * 50)
    assert response.headers["x-llamajev-prefix-tokens"] == "50"
    assert response.headers["x-llamajev-readout-retries"] == "0"
    assert response.headers["x-llamajev-slot"] == "0"
    assert "server-timing" in response.headers


async def test_missing_labels_escalate_then_error(api, payload):
    client, backend = api
    payload["questions"] = {"big": {"type": "choice", "instructions": "x", "criteria": {str(i): f"o{i}" for i in range(64)}}}
    backend.readout_depth = 10  # backend's top-n only ever reaches 10 of the 64 labels
    response = await client.post("/v1/systemone", json=payload)
    assert response.status_code == 502, response.text
    assert response.json()["error"]["code"] == "readout_truncated"
    depths = [c["n_probs"] for c in backend.calls if "prompt" in c][1:]
    assert depths == [16 * 64, 4096, 32768]


async def test_missing_labels_recover_on_retry(api, payload):
    client, backend = api
    payload["questions"] = {"big": {"type": "choice", "instructions": "x", "criteria": {str(i): f"o{i}" for i in range(64)}}}
    backend.readout_depth, backend.full_at = 10, 4096  # first depth (1024) misses labels, 4096 covers them
    response = await client.post("/v1/systemone", json=payload)
    assert response.status_code == 200, response.text
    assert response.headers["x-llamajev-readout-retries"] == "1"
    assert len(response.json()["answers"]["big"]["probabilities"]) == 64
    # The failed first attempt ran on the backend and must be billed: warm (50) +
    # failed attempt (80) + successful attempt (80); cached: failed (50) + successful (50).
    assert response.json()["usage"]["input_tokens"] == 50 + 2 * 80
    assert response.headers["x-llamajev-cached-tokens"] == str(2 * 50)


async def test_structured_criteria_and_instructions(api):
    client, backend = api
    payload = {"model": "jev-latest", "state": {"ticket": "Card declined twice"}, "questions": {
        "n": {"type": "noul", "instructions": {"question": "Refund?", "notes": ["be strict"]}, "criteria": {"true": {"means": "asks for money back"}, "false": ["anything else"]}},
        "c": {"type": "choice", "instructions": "Which?", "criteria": {"a": {"covers": "payments"}, "b": None}},
        "s": {"type": "score", "instructions": "Severity", "criteria": [{"level": "low"}, "high"]},
    }}
    response = await client.post("/v1/systemone", json=payload)
    assert response.status_code == 200, response.text
    prompts = [c["prompt"] for c in backend.calls if "prompt" in c]
    assert 'A: {"means":"asks for money back"}' in prompts[1] and 'B: ["anything else"]' in prompts[1]
    assert 'Question: {"question":"Refund?","notes":["be strict"]}' in prompts[1]
    assert "A: {\"covers\":\"payments\"}\nB: b" in prompts[2]
    assert response.json()["answers"]["s"]["legend"] == {"0": '{"level":"low"}', "1": "high"}


@pytest.mark.parametrize("state", [[{"role": [], "content": "x"}], {"n": 2**80}, [{"role": "user", "content": [{"type": "text", "text": 5}]}]])
async def test_malformed_state_is_a_422_not_a_500(api, state):
    client, _ = api
    response = await client.post("/v1/systemone", json={"model": "jev-latest", "state": state, "questions": {"q": {"type": "noul", "instructions": "x"}}})
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] in {"unsupported_content", "validation"}


async def test_evaluation_deadline(api, payload):
    client, backend = api
    backend.delay = 0.2
    client.app.state.service.settings = client.app.state.service.settings.model_copy(update={"request_timeout": 0.05})
    response = await client.post("/v1/systemone", json=payload)
    assert response.status_code == 504
    assert response.json()["error"]["code"] == "backend_timeout"


async def test_concurrent_evaluations_get_distinct_slots(api, payload):
    import asyncio

    client, backend = api
    backend.delay = 0.01
    responses = await asyncio.gather(*[client.post("/v1/systemone", json=payload) for _ in range(6)])
    assert all(r.status_code == 200 for r in responses)
    slots = [r.headers["x-llamajev-slot"] for r in responses]
    assert set(slots) <= {"0", "1", "2", "3"}
    # every call of one evaluation used its leased slot; leases were reused after release
    assert len(set(slots)) >= 2


@pytest.mark.parametrize("count,status", [(1, 422), (2, 200), (64, 200), (65, 422)])
async def test_answer_limits(api, count, status):
    client, _ = api
    response = await client.post(
        "/v1/systemone",
        json={"model": "jev-latest", "state": "t", "questions": {"q": {"type": "choice", "instructions": "p", "criteria": {str(i): f"o{i}" for i in range(count)}}}},
    )
    assert response.status_code == status, response.text
    if status == 422:
        assert response.json()["error"]["code"] == "validation"


async def test_unknown_model(api, payload):
    client, backend = api
    payload["model"] = "gpt-9"
    response = await client.post("/v1/systemone", json=payload)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "unknown_model"
    assert not [c for c in backend.calls if "prompt" in c]


async def test_served_model_name_accepted(api, payload):
    client, _ = api
    payload["model"] = "Qwen3.5-2B-Q8_0"
    response = await client.post("/v1/systemone", json=payload)
    assert response.status_code == 200, response.text
    assert response.json()["model"] == "Qwen3.5-2B-Q8_0"


async def test_images_rejected_without_vision(api, payload):
    client, _ = api
    payload["state"] = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}}]}]
    response = await client.post("/v1/systemone", json=payload)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "unsupported_content"


async def test_body_limit(api, payload):
    client, _ = api
    payload["state"] = "x" * (3 * 1024 * 1024)
    response = await client.post("/v1/systemone", json=payload)
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "body_too_large"


async def test_models_limits_health(api):
    client, backend = api
    models = await client.get("/v1/models")
    assert {m["name"] for m in models.json()["models"]} == {"jev-latest", "Qwen3.5-2B-Q8_0"}
    limits = await client.get("/v1/limits")
    assert limits.json()["max_questions"] == 64 and limits.json()["top_n"] == 256
    health = await client.get("/health")
    assert health.status_code == 200
    assert health.json()["labels_verified"] == 64 and health.json()["vision"] is False
    backend.healthy = False
    assert (await client.get("/health")).status_code == 503


# --- hardening: no credential/500 leaks, serialization edges ---


async def _make_client(settings):
    backend = FakeBackend()
    props = Props(model_path="/models/Qwen3.5-2B-Q8_0.gguf", n_ctx=8192, n_slots=4, vision=False, media_marker="<__media__>")
    service = EvaluationService(settings, PromptCompiler(fake_labels()), backend, props)
    app = create_app(settings, service=service)
    client = _httpx.AsyncClient(transport=_httpx.ASGITransport(app=app), base_url="http://test")
    client.app = app
    return client, app


async def test_health_does_not_leak_backend_credentials():
    settings = Settings(top_n=256, backend_url="http://user:sekret@10.0.0.9:8090/x?token=abcd")
    client, app = await _make_client(settings)
    async with client, app.router.lifespan_context(app):
        r = await client.get("/health")
    text = r.text
    assert "sekret" not in text and "user" not in text and "token" not in text and "abcd" not in text
    assert r.json()["backend_url"] == "http://10.0.0.9:8090"


async def test_unicode_served_model_name_does_not_500(payload):
    settings = Settings(top_n=256, served_model_name="模型")  # 模型
    client, app = await _make_client(settings)
    async with client, app.router.lifespan_context(app):
        r = await client.post("/v1/systemone", json=payload)
    assert r.status_code == 200, r.text
    assert r.headers["x-llamajev-model"].isascii()


async def test_oversized_int_in_chat_message_is_422_not_500(api):
    client, _ = api
    payload = {"model": "jev-latest",
               "state": [{"role": "user", "content": "x", "metadata": 2 ** 80}],
               "questions": {"q": {"type": "noul", "instructions": "x"}}}
    r = await client.post("/v1/systemone", json=payload)
    assert r.status_code == 422, r.text
    assert r.json()["error"]["code"] == "validation"


async def test_lone_surrogate_question_key_is_not_500(api):
    client, _ = api
    body = '{"model":"jev-latest","state":"x","questions":{"\\ud800":{"type":"noul","instructions":"x"}}}'
    r = await client.post("/v1/systemone", content=body.encode("utf-8", "surrogatepass"),
                          headers={"content-type": "application/json"})
    assert r.status_code in {400, 422}, r.text
