"""HTTP primitives against an unmodified llama-server: /props, /apply-template, /tokenize, /completion."""

import math
from dataclasses import dataclass, field
from typing import Any

import httpx
import orjson

from .config import MAX_ANSWERS
from .models import ErrorCode
from .prompts import READOUT_PREFIX, label_candidates


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
    cached_tokens: int = 0  # tokens reused from the slot cache
    prompt_ms: float = 0.0
    id_slot: int | None = None

    @property
    def total_prompt_tokens(self) -> int:
        return self.prompt_tokens + self.cached_tokens


class LlamaClient:
    def __init__(self, client: httpx.AsyncClient):
        self.client = client

    async def health(self) -> bool:
        try:
            response = await self.client.get("/health", timeout=5)
            return response.is_success
        except httpx.HTTPError:
            return False

    async def props(self) -> Props:
        data = json_object(await self._get("/props"), "/props")
        settings = data.get("default_generation_settings")
        settings = settings if isinstance(settings, dict) else {}
        modalities = data.get("modalities")
        modalities = modalities if isinstance(modalities, dict) else {}
        try:
            return Props(
                model_path=str(data.get("model_path", "")),
                n_ctx=int(settings.get("n_ctx", 0)),
                n_slots=int(data.get("total_slots", 1)),
                vision=bool(modalities.get("vision", False)),
                media_marker=str(data.get("media_marker", "<__media__>")),
            )
        except (TypeError, ValueError) as exc:
            raise BackendError(f"Invalid /props response: {exc}") from exc

    async def apply_template(self, messages: list[dict[str, Any]]) -> str:
        body = {"messages": messages, "chat_template_kwargs": {"enable_thinking": False}}
        data = json_object(await self._post("/apply-template", body), "/apply-template")
        prompt = data.get("prompt")
        if not isinstance(prompt, str):
            raise BackendError("Invalid /apply-template response: prompt is not a string")
        return prompt

    async def tokenize(self, text: str) -> list[tuple[int, str]]:
        body = {"content": text, "add_special": False, "with_pieces": True}
        data = json_object(await self._post("/tokenize", body), "/tokenize")
        try:
            return [(int(t["id"]), t["piece"]) for t in data["tokens"]]
        except (KeyError, TypeError, ValueError) as exc:
            raise BackendError(f"Invalid /tokenize response: {exc}") from exc

    async def verify_labels(self) -> list[tuple[str, int]]:
        """Single-token answer labels, checked alone and after the readout prefix ("Answer:\\n")."""
        found: list[tuple[str, int]] = []
        seen: set[int] = set()
        for label in label_candidates():
            alone = await self.tokenize(label)
            if len(alone) != 1 or alone[0][1] != label or alone[0][0] in seen:
                continue
            in_context = await self.tokenize(READOUT_PREFIX + label)
            if not in_context or in_context[-1][0] != alone[0][0]:
                continue
            found.append((label, alone[0][0]))
            seen.add(alone[0][0])
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
        id_slot: int | None = None,
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
        if id_slot is not None:
            body["id_slot"] = id_slot
        data = json_object(await self._post("/completion", body), "/completion")
        try:
            return parse_generation(data, n_probs > 0)
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
            content = orjson.dumps(body)
        except (TypeError, ValueError, OverflowError) as exc:
            raise BackendError(f"request is not JSON-serializable: {exc}", 422, "validation") from exc
        try:
            response = await self.client.post(
                path, content=content, headers={"Content-Type": "application/json"}
            )
        except httpx.TimeoutException as exc:
            raise BackendError("llama-server request timed out", 504, "backend_timeout") from exc
        except httpx.HTTPError as exc:
            raise BackendError(unreachable(self.client), 503, "backend_unreachable") from exc
        return check(response)


def sanitize_url(url: str) -> str:
    """Origin only (scheme://host:port) — drop any userinfo credentials and query so a URL that
    reaches a client (health, error text) never carries a token or password."""
    try:
        parsed = httpx.URL(url)
        host = parsed.host or ""
        port = f":{parsed.port}" if parsed.port else ""
        return f"{parsed.scheme}://{host}{port}" if host else "the backend"
    except (httpx.InvalidURL, ValueError):
        return "the backend"


def unreachable(client: httpx.AsyncClient) -> str:
    return (
        f"llama-server unreachable at {sanitize_url(str(client.base_url))} — start it with "
        "`llamajev serve --model <gguf>` or point --connect at a running server"
    )


def json_object(response: httpx.Response, path: str) -> dict:
    try:
        data = response.json()
    except ValueError as exc:
        raise BackendError(f"Invalid {path} response: not JSON") from exc
    if not isinstance(data, dict):
        raise BackendError(f"Invalid {path} response: expected an object")
    return data


def check(response: httpx.Response) -> httpx.Response:
    if response.is_success:
        return response
    try:
        message = str(response.json()["error"]["message"])
    except (ValueError, KeyError, TypeError):
        message = response.text[:200]
    status = response.status_code
    if status == 400 and ("context" in message or "exceed" in message):
        raise BackendError(f"Prompt exceeds the backend context: {message}", 422, "too_many_tokens")
    if status in {400, 422}:
        raise BackendError(f"llama-server rejected the request: {message}", 422, "validation")
    if status in {429, 503, 529}:
        raise BackendError(f"llama-server is overloaded: {message}", 529, "overloaded")
    # Unexpected backend status: the raw body can carry internal detail (paths, config), so keep it
    # out of the client-facing error — the status alone tells the operator where to look in the logs.
    raise BackendError(f"llama-server returned HTTP {status}", 502, "backend_error")


def parse_generation(data: dict, want_probs: bool) -> Generation:
    timings = data.get("timings")
    timings = timings if isinstance(timings, dict) else {}
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
    id_slot = data.get("id_slot")
    return Generation(
        sampled=str(data.get("content", "")),
        logprobs=logprobs,
        prompt_tokens=int(timings.get("prompt_n", 0)),
        cached_tokens=int(timings.get("cache_n", 0)),
        prompt_ms=float(timings.get("prompt_ms", 0.0)),
        id_slot=int(id_slot) if id_slot is not None else None,
    )
