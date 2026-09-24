import asyncio
import time
from contextlib import asynccontextmanager
from uuid import uuid4

import httpx
import orjson
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from . import __version__
from .backend import BackendError, LlamaClient, sanitize_url
from .config import MAX_ANSWERS, MAX_QUESTIONS, Settings
from .models import (
    ErrorResponse,
    HealthResponse,
    LimitsResponse,
    ModelsResponse,
    SystemOneRequest,
    SystemOneResponse,
)
from .prompts import PromptCompiler
from .service import EvaluationService, RequestError


class ORJSONResponse(JSONResponse):
    def render(self, content) -> bytes:
        return orjson.dumps(content)


def error(status: int, code: str, message: str, headers: dict | None = None) -> ORJSONResponse:
    return ORJSONResponse({"error": {"message": message, "code": code}}, status, headers=headers)


class BodyLimit:
    """Bound even chunked request bodies before JSON parsing."""

    def __init__(self, app: ASGIApp, max_bytes: int, read_timeout: float = 30.0):
        self.app = app
        self.max_bytes = max_bytes
        self.read_timeout = read_timeout

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http" or scope["method"] not in {"POST", "PUT", "PATCH"}:
            return await self.app(scope, receive, send)
        body = bytearray()
        try:
            deadline = asyncio.get_running_loop().time() + self.read_timeout
            while True:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise TimeoutError
                event = await asyncio.wait_for(receive(), timeout=remaining)
                if event["type"] == "http.disconnect":
                    return
                body.extend(event.get("body", b""))
                if len(body) > self.max_bytes:
                    return await error(413, "body_too_large", "Request body is too large")(scope, receive, send)
                if not event.get("more_body", False):
                    break
        except TimeoutError:
            # A slow/stalled upload must not hold a worker before admission control; cut it off.
            return await error(408, "validation", "Request body was not received in time")(scope, receive, send)
        delivered = False

        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


def create_app(
    settings: Settings | None = None,
    *,
    service: EvaluationService | None = None,
    launch_backend: bool = False,
) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        started = time.monotonic()
        if service is not None:
            app.state.service = service
            app.state.startup_seconds = 0.0
            yield
            return
        from .runtime import start_backend, stop_backend, wait_ready

        process = start_backend(settings) if launch_backend else None
        try:
            async with httpx.AsyncClient(
                base_url=settings.backend_url,
                timeout=httpx.Timeout(settings.request_timeout, connect=5),
                limits=httpx.Limits(max_connections=128, max_keepalive_connections=128),
            ) as client:
                backend = LlamaClient(client)
                await wait_ready(backend, process, settings.startup_timeout)
                props = await backend.props()
                compiler = PromptCompiler(await backend.verify_labels())
                token_ids = settings.token_ids_readout and await backend.supports_token_ids(compiler.labels[0][1])
                app.state.service = EvaluationService(settings, compiler, backend, props, token_ids)
                app.state.startup_seconds = round(time.monotonic() - started, 2)
                yield
        finally:
            if process is not None:
                stop_backend(process)

    app = FastAPI(
        title="llamajev",
        version=__version__,
        lifespan=lifespan,
        default_response_class=ORJSONResponse,
        description=(
            "TypeSafe Jev-compatible `/v1/systemone` served by an unmodified llama-server: "
            "one chat-template render, one cache-warming call, one single-token call per question, "
            "candidate-label logprobs renormalized with softmax."
        ),
    )
    app.add_middleware(BodyLimit, max_bytes=settings.max_body_bytes, read_timeout=settings.body_read_timeout)

    @app.exception_handler(RequestError)
    @app.exception_handler(BackendError)
    async def handled(request: Request, exc: RequestError | BackendError):
        headers = {"Retry-After": "1"} if exc.status in {429, 503, 529} else None
        return error(exc.status, exc.code, str(exc), headers)

    @app.exception_handler(RequestValidationError)
    async def invalid(request: Request, exc: RequestValidationError):
        first = exc.errors()[0] if exc.errors() else {}
        loc = ".".join(str(x) for x in first.get("loc", ()) if x != "body")
        return error(422, "validation", f"{loc or 'body'}: {first.get('msg', 'invalid request')}")

    @app.post(
        "/v1/systemone",
        response_model=SystemOneResponse,
        responses={code: {"model": ErrorResponse} for code in (413, 422, 502, 503, 504, 529)},
        summary="Evaluate typed questions against a shared state",
    )
    async def systemone(payload: SystemOneRequest, request: Request):
        async def disconnect():
            while True:
                if (await request.receive())["type"] == "http.disconnect":
                    return

        evaluation = asyncio.create_task(request.app.state.service.evaluate(payload))
        watcher = asyncio.create_task(disconnect())
        try:
            await asyncio.wait([evaluation, watcher], return_when=asyncio.FIRST_COMPLETED)
            if not evaluation.done():
                raise RequestError("Client disconnected", 499, "client_disconnected")
            result = await evaluation
            service = request.app.state.service
            return ORJSONResponse(
                result.response.model_dump(),
                headers={
                    "x-typesafe-request-id": uuid4().hex,
                    # HTTP headers are Latin-1; keep the real (possibly Unicode) name in the JSON body,
                    # expose an ASCII-safe form in the header so a Unicode model name cannot 500.
                    "x-llamajev-model": service.served_model_name.encode("ascii", "replace").decode() or "model",
                    "x-llamajev-prefix-tokens": str(result.prefix_tokens),
                    "x-llamajev-cached-tokens": str(result.cached_tokens),
                    "x-llamajev-readout-retries": str(result.readout_retries),
                    **({"x-llamajev-slot": str(result.slot)} if result.slot is not None else {}),
                    "Server-Timing": (
                        f"prepare;dur={result.prepare_ms:.2f}, "
                        f"prefill;dur={result.prefill_ms:.2f}, "
                        f"branches;dur={result.branches_ms:.2f}"
                    ),
                },
            )
        finally:
            for task in (evaluation, watcher):
                if not task.done():
                    task.cancel()
            await asyncio.gather(evaluation, watcher, return_exceptions=True)

    @app.get("/v1/models", response_model=ModelsResponse)
    async def models(request: Request):
        service = request.app.state.service
        names = service.accepted_models()
        return {
            "object": "list",
            "data": [{"id": n, "object": "model", "owned_by": "llamajev"} for n in names],
            "models": [
                {"name": n, "description": f"llamajev on {service.served_model_name}", "release_date": "2026-09-21"}
                for n in names
            ],
        }

    @app.get("/v1/limits", response_model=LimitsResponse)
    async def limits(request: Request):
        return {
            "max_answers_per_question": MAX_ANSWERS,
            "max_questions": MAX_QUESTIONS,
            "max_body_bytes": settings.max_body_bytes,
            "max_input_tokens": request.app.state.service.props.n_ctx,
            "max_concurrent_requests": settings.max_concurrent_requests,
            "top_n": settings.top_n,
        }

    @app.get("/health", response_model=HealthResponse, responses={503: {"model": HealthResponse}})
    async def health(request: Request):
        service = getattr(request.app.state, "service", None)
        healthy = service is not None and await service.backend.health()
        body = {
            "status": "ok" if healthy else "unavailable",
            "backend_url": sanitize_url(settings.backend_url),
            "startup_seconds": getattr(request.app.state, "startup_seconds", None),
        }
        if service is not None:
            body.update(
                model=service.served_model_name,
                n_slots=service.props.n_slots,
                n_ctx=service.props.n_ctx,
                vision=service.props.vision,
                labels_verified=len(service.compiler.labels),
                readout="token_ids" if service.token_ids else "top_n",
            )
        return ORJSONResponse(body, status_code=200 if healthy else 503)

    @app.get("/health/live")
    async def live():
        return {"status": "ok"}

    return app
