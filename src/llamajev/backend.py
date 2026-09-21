"""HTTP primitives against an unmodified llama-server: /props, /apply-template, /tokenize, /completion."""

import asyncio
import math
from dataclasses import dataclass, field
from typing import Any

import httpx
import orjson

from .config import MAX_ANSWERS
from .models import ErrorCode
from .prompts import label_candidates


class BackendError(Exception):
    def __init__(self, message: str, status: int = 502, code: ErrorCode = "backend_error"):
        super().__init__(message)
        self.status = status
        self.code = code


@dataclass(frozen=True)
class Props:
    model_path: str
    n_ctx: int
    n_slots: int
    vision: bool
    media_marker: str


@dataclass(frozen=True)
class Generation:
    sampled: str
    logprobs: dict[int, float] = field(default_factory=dict)  # token id -> log p (full-vocab softmax)
    prompt_tokens: int = 0  # tokens processed this call
    cached_tokens: int = 0  # tokens reused from the slot / prompt cache
    prompt_ms: float = 0.0

    @property
    def total_prompt_tokens(self) -> int:
        return self.prompt_tokens + self.cached_tokens


class LlamaClient:
    def __init__(self, client: httpx.AsyncClient, max_concurrent_branches: int):
        self.client = client
        self.slots = asyncio.Semaphore(max_concurrent_branches)

    async def health(self) -> bool:
        try:
            response = await self.client.get("/health", timeout=5)
            return response.is_success
        except httpx.HTTPError:
            return False

    async def props(self) -> Props:
        data = (await self._get("/props")).json()
        settings = data.get("default_generation_settings", {})
        modalities = data.get("modalities") or {}
        return Props(
            model_path=data.get("model_path", ""),
            n_ctx=int(settings.get("n_ctx", 0)),
            n_slots=int(data.get("total_slots", 1)),
            vision=bool(modalities.get("vision", False)),
            media_marker=data.get("media_marker", "<__media__>"),
        )

    async def apply_template(self, messages: list[dict[str, Any]]) -> str:
        body = {"messages": messages, "chat_template_kwargs": {"enable_thinking": False}}
        response = await self._post("/apply-template", body)
        try:
            return response.json()["prompt"]
        except (KeyError, ValueError) as exc:
            raise BackendError(f"Invalid /apply-template response: {exc}") from exc

    async def tokenize(self, text: str) -> list[tuple[int, str]]:
        body = {"content": text, "add_special": False, "with_pieces": True}
        response = await self._post("/tokenize", body)
        try:
            return [(t["id"], t["piece"]) for t in response.json()["tokens"]]
        except (KeyError, TypeError, ValueError) as exc:
            raise BackendError(f"Invalid /tokenize response: {exc}") from exc

    async def verify_labels(self) -> list[tuple[str, int]]:
        """Single-token answer labels, checked against the live model's tokenizer."""
        found: list[tuple[str, int]] = []
        seen: set[int] = set()
        for label in label_candidates():
            tokens = await self.tokenize(label)
            if len(tokens) == 1 and tokens[0][1] == label and tokens[0][0] not in seen:
                found.append((label, tokens[0][0]))
                seen.add(tokens[0][0])
            if len(found) == MAX_ANSWERS:
                break
        return found

    async def complete(
        self,
        prompt: str,
        *,
        images: list[str] | None = None,
        n_probs: int = 0,
        grammar: str | None = None,
    ) -> Generation:
        body: dict[str, Any] = {
            "prompt": {"prompt_string": prompt, "multimodal_data": images} if images else prompt,
            "n_predict": 1,
            "temperature": 0,
            "cache_prompt": True,
            "n_probs": n_probs,
            "stream": False,
        }
        if grammar:
            body["grammar"] = grammar
        async with self.slots:
            response = await self._post("/completion", body)
        try:
            return parse_generation(response.json(), n_probs > 0)
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            raise BackendError(f"Invalid /completion response: {exc}") from exc

    async def _get(self, path: str) -> httpx.Response:
        try:
            response = await self.client.get(path)
        except httpx.TimeoutException as exc:
            raise BackendError("llama-server request timed out", 504, "backend_timeout") from exc
        except httpx.HTTPError as exc:
            raise BackendError(unreachable(self.client), 503, "backend_unreachable") from exc
        return check(response)

    async def _post(self, path: str, body: dict[str, Any]) -> httpx.Response:
        try:
            response = await self.client.post(
                path, content=orjson.dumps(body), headers={"Content-Type": "application/json"}
            )
        except httpx.TimeoutException as exc:
            raise BackendError("llama-server request timed out", 504, "backend_timeout") from exc
        except httpx.HTTPError as exc:
            raise BackendError(unreachable(self.client), 503, "backend_unreachable") from exc
        return check(response)


def unreachable(client: httpx.AsyncClient) -> str:
    return (
        f"llama-server unreachable at {client.base_url} — start it with "
        "`llamajev serve --model <gguf>` or point --connect at a running server"
    )


def check(response: httpx.Response) -> httpx.Response:
    if response.is_success:
        return response
    try:
        message = response.json()["error"]["message"]
    except (ValueError, KeyError, TypeError):
        message = response.text[:200]
    status = response.status_code
    if status == 400 and ("context" in message or "exceed" in message):
        raise BackendError(f"Prompt exceeds the backend context: {message}", 422, "too_many_tokens")
    if status in {400, 422}:
        raise BackendError(f"llama-server rejected the request: {message}", 422, "validation")
    if status in {429, 503, 529}:
        raise BackendError(f"llama-server is overloaded: {message}", 529, "overloaded")
    raise BackendError(f"llama-server returned HTTP {status}: {message}", 502, "backend_error")


def parse_generation(data: dict, want_probs: bool) -> Generation:
    timings = data.get("timings") or {}
    logprobs: dict[int, float] = {}
    if want_probs:
        positions = data["completion_probabilities"]
        if len(positions) != 1:
            raise ValueError(f"expected one readout position, got {len(positions)}")
        for entry in positions[0]["top_logprobs"]:
            value = float(entry["logprob"])
            if math.isnan(value) or value == math.inf:
                raise ValueError("non-finite logprob")
            logprobs[int(entry["id"])] = value
        if not logprobs:
            raise ValueError("empty top_logprobs")
    return Generation(
        sampled=str(data.get("content", "")),
        logprobs=logprobs,
        prompt_tokens=int(timings.get("prompt_n", 0)),
        cached_tokens=int(timings.get("cache_n", 0)),
        prompt_ms=float(timings.get("prompt_ms", 0.0)),
    )
