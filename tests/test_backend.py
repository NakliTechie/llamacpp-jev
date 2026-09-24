"""LlamaClient against an httpx MockTransport: request bodies, response parsing, error mapping."""

import json

import httpx
import pytest

from llamajev.backend import BackendError, LlamaClient

COMPLETION = {
    "content": "A",
    "id_slot": 2,
    "timings": {"prompt_n": 36, "cache_n": 224, "prompt_ms": 41.2},
    "completion_probabilities": [{"token": "A", "logprob": -0.05, "top_logprobs": [
        {"id": 32, "token": "A", "logprob": -0.05}, {"id": 33, "token": "B", "logprob": -3.1}]}],
}


def make_client(handler):
    transport = httpx.MockTransport(handler)
    return LlamaClient(httpx.AsyncClient(transport=transport, base_url="http://backend"))


async def test_complete_request_shape_and_parse():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json=COMPLETION)

    gen = await make_client(handler).complete("prefix suffix", images=["QUJD"], n_probs=64, grammar='root ::= "A"', id_slot=2)
    assert seen["prompt"] == {"prompt_string": "prefix suffix", "multimodal_data": ["QUJD"]}
    assert seen["n_predict"] == 1 and seen["temperature"] == 0 and seen["cache_prompt"] is True
    assert seen["n_probs"] == 64 and seen["grammar"] == 'root ::= "A"' and seen["id_slot"] == 2
    assert gen.sampled == "A" and gen.logprobs == {32: -0.05, 33: -3.1}
    assert gen.prompt_tokens == 36 and gen.cached_tokens == 224 and gen.total_prompt_tokens == 260
    assert gen.id_slot == 2


async def test_text_prompt_is_a_plain_string():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={**COMPLETION, "completion_probabilities": []})

    await make_client(handler).complete("just text")
    assert seen["prompt"] == "just text" and "grammar" not in seen and "id_slot" not in seen


@pytest.mark.parametrize("body,code,status", [
    ({"error": {"message": "the request exceeds the available context size"}}, "too_many_tokens", 422),
    ({"error": {"message": "Failed to parse grammar"}}, "validation", 422),
])
async def test_backend_400_mapping(body, code, status):
    client = make_client(lambda request: httpx.Response(400, json=body))
    with pytest.raises(BackendError) as exc:
        await client.complete("x")
    assert exc.value.code == code and exc.value.status == status


async def test_backend_503_is_overloaded():
    client = make_client(lambda request: httpx.Response(503, text="loading"))
    with pytest.raises(BackendError) as exc:
        await client.complete("x")
    assert exc.value.code == "overloaded" and exc.value.status == 529


async def test_unreachable_and_timeout():
    def down(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(BackendError) as exc:
        await make_client(down).complete("x")
    assert exc.value.code == "backend_unreachable" and "llamajev serve" in str(exc.value)

    def slow(request):
        raise httpx.ReadTimeout("slow")

    with pytest.raises(BackendError) as exc:
        await make_client(slow).apply_template([])
    assert exc.value.code == "backend_timeout"


@pytest.mark.parametrize("payload", [[], {"prompt": 42}, "not json"])
async def test_malformed_apply_template_is_backend_error(payload):
    def handler(request):
        return httpx.Response(200, text=payload) if isinstance(payload, str) else httpx.Response(200, json=payload)

    with pytest.raises(BackendError) as exc:
        await make_client(handler).apply_template([{"role": "user", "content": "x"}])
    assert exc.value.code == "backend_error" and exc.value.status == 502


@pytest.mark.parametrize("bad", [
    {"completion_probabilities": []},
    {"completion_probabilities": [{"top_logprobs": []}]},
    {"completion_probabilities": [{"top_logprobs": [{"id": 1, "logprob": "nan"}]}]},
])
async def test_malformed_completion_is_backend_error(bad):
    client = make_client(lambda request: httpx.Response(200, json={**COMPLETION, **bad}))
    with pytest.raises(BackendError) as exc:
        await client.complete("x", n_probs=8)
    assert exc.value.code == "backend_error"


async def test_verify_labels_checks_in_context():
    def tid(label):
        return 500 + ord(label[0])

    def handler(request):
        text = json.loads(request.content)["content"]
        label = text.removeprefix("Answer:\n")
        if len(label) > 1:  # two-letter candidates split in this fake tokenizer
            return httpx.Response(200, json={"tokens": [{"id": 7, "piece": label[0]}, {"id": 8, "piece": label[1:]}]})
        if text.startswith("Answer:\n"):
            toks = [{"id": 1, "piece": "Answer"}, {"id": 2, "piece": ":"}, {"id": 3, "piece": "\n"}]
            if label == "B":  # simulate a tokenizer merging "\nB" so B is rejected in context
                return httpx.Response(200, json={"tokens": toks[:-1] + [{"id": 99, "piece": "\nB"}]})
            return httpx.Response(200, json={"tokens": toks + [{"id": tid(label), "piece": label}]})
        return httpx.Response(200, json={"tokens": [{"id": tid(label), "piece": label}]})

    labels = await make_client(handler).verify_labels()
    names = [name for name, _ in labels]
    assert names[:3] == ["A", "C", "D"] and "B" not in names
    assert len(labels) == 25  # 26 single letters minus B; two-letter candidates split


async def test_props_parsing():
    client = make_client(lambda request: httpx.Response(200, json={
        "model_path": "/m/Qwen3.5-2B-Q8_0.gguf", "total_slots": 4, "media_marker": "<__media_x__>",
        "modalities": {"vision": True}, "default_generation_settings": {"n_ctx": 2048}}))
    props = await client.props()
    assert (props.n_slots, props.n_ctx, props.vision, props.media_marker) == (4, 2048, True, "<__media_x__>")


async def test_token_ids_request_shape_and_parse():
    seen = {}
    body = {**COMPLETION, "completion_probabilities": [{"token": "A", "logprob": -0.05, "top_logprobs": [],
        "token_ids_logprobs": [{"id": 33, "token": "B", "logprob": -3.1}, {"id": 32424, "token": "ZZ", "logprob": -19.4}]}]}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json=body)

    gen = await make_client(handler).complete("x", token_ids=[33, 32424])
    assert seen["token_ids_logprob"] == [33, 32424] and seen["n_probs"] == 0
    assert gen.logprobs == {33: -3.1, 32424: -19.4}


@pytest.mark.parametrize("entry,expected", [
    ({"token": "A", "logprob": -0.1, "top_logprobs": [], "token_ids_logprobs": [{"id": 5, "logprob": -1.0}]}, True),
    ({"token": "A", "logprob": -0.1, "top_logprobs": []}, False),  # unmodified server ignores the field
])
async def test_supports_token_ids_probe(entry, expected):
    client = make_client(lambda request: httpx.Response(200, json={**COMPLETION, "completion_probabilities": [entry]}))
    assert await client.supports_token_ids(5) is expected
