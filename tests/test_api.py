
import pytest
from conftest import sigmoid


async def test_request_prefills_then_branches(api, payload):
    client, backend = api
    response = await client.post("/v1/systemone", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    calls = [c for c in backend.calls if "prompt" in c]
    assert len(calls) == 4  # warm-up + 3 branches
    prefix = calls[0]["prompt"]
    assert calls[0]["n_probs"] == 0 and calls[0]["grammar"] is None
    for branch in calls[1:]:
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
    assert response.headers["x-llamajev-truncated-labels"] == "0"
    assert "server-timing" in response.headers


async def test_truncated_labels_are_counted(api, payload):
    client, backend = api
    payload["questions"] = {"big": {"type": "choice", "instructions": "x", "criteria": {str(i): f"o{i}" for i in range(64)}}}
    backend.readout_depth = 10  # backend's top-n only reaches 10 of the 64 labels
    response = await client.post("/v1/systemone", json=payload)
    assert response.status_code == 200, response.text
    assert response.headers["x-llamajev-truncated-labels"] == "54"
    assert [c for c in backend.calls if "prompt" in c][-1]["n_probs"] == 16 * 64
    probs = response.json()["answers"]["big"]["probabilities"]
    assert probs["0"] > probs["9"] > 0 and probs["10"] == 0


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
