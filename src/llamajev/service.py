"""N+1 orchestration: render once, lease a slot, warm the prefix, score every branch, renormalize."""

import asyncio
import logging
import time
from dataclasses import dataclass

from .backend import BackendError, LlamaClient, Props
from .config import Settings
from .models import ErrorCode, SystemOneRequest, SystemOneResponse, Usage
from .prompts import Branch, ContentError, PromptCompiler, build_messages
from .scoring import answer

logger = logging.getLogger(__name__)

READOUT_PER_LABEL = 16
READOUT_ESCALATION = (4096, 32768)  # retried n_probs when a label is missing from the readout


class RequestError(Exception):
    def __init__(self, message: str, status: int, code: ErrorCode):
        super().__init__(message)
        self.status = status
        self.code = code


@dataclass(frozen=True)
class Evaluation:
    response: SystemOneResponse
    prefix_tokens: int
    cached_tokens: int
    readout_retries: int
    slot: int | None
    prepare_ms: float
    prefill_ms: float
    branches_ms: float


class SlotPool:
    """Leases one llama-server slot per evaluation so the warm-up and all its branches share it."""

    def __init__(self, n_slots: int):
        self.queue: asyncio.Queue[int] = asyncio.Queue()
        for slot in range(n_slots):
            self.queue.put_nowait(slot)

    async def acquire(self) -> int:
        return await self.queue.get()

    def release(self, slot: int) -> None:
        self.queue.put_nowait(slot)


class EvaluationService:
    def __init__(self, settings: Settings, compiler: PromptCompiler, backend: LlamaClient, props: Props):
        self.settings = settings
        self.compiler = compiler
        self.backend = backend
        self.props = props
        self.admission = asyncio.Semaphore(settings.max_concurrent_requests)
        self.slots = SlotPool(props.n_slots) if settings.pin_slot else None
        stem = props.model_path.rsplit("/", 1)[-1].removesuffix(".gguf")
        self.served_model_name = settings.served_model_name or stem or "llama-server"

    def accepted_models(self) -> list[str]:
        return list(dict.fromkeys([self.settings.model_alias, self.served_model_name]))

    async def evaluate(self, request: SystemOneRequest) -> Evaluation:
        if request.model not in self.accepted_models():
            raise RequestError(f"Unknown model: {request.model}", 422, "unknown_model")
        if self.admission.locked():
            raise RequestError("Too many concurrent evaluations", 529, "overloaded")
        async with self.admission:
            try:
                async with asyncio.timeout(self.settings.request_timeout):
                    return await self._evaluate(request)
            except TimeoutError as exc:
                raise RequestError(
                    f"Evaluation exceeded {self.settings.request_timeout:g}s (LLAMAJEV_REQUEST_TIMEOUT)",
                    504,
                    "backend_timeout",
                ) from exc

    async def _evaluate(self, request: SystemOneRequest) -> Evaluation:
        t0 = time.perf_counter()
        try:
            messages, images, marker = build_messages(
                request,
                self.props.vision,
                max_images=self.settings.max_images,
                max_image_bytes=self.settings.max_image_bytes,
                max_image_pixels=self.settings.max_image_pixels,
            )
            rendered = await self.backend.apply_template(messages)
            prepared = self.compiler.compile(request, rendered, marker, images)
        except ContentError as exc:
            raise RequestError(str(exc), 422, "unsupported_content") from exc
        budget = self.props.n_ctx
        if budget and any(len(prepared.prefix) + len(b.suffix) > budget * 16 for b in prepared.branches):
            # Chars per token is never below ~1/16 on these tokenizers; the backend rejects the rest exactly.
            raise RequestError("A question branch exceeds the backend context window", 422, "too_many_tokens")
        t1 = time.perf_counter()

        slot = await self.slots.acquire() if self.slots else None
        try:
            warm = await self.backend.complete(prepared.prefix, images=prepared.images or None, id_slot=slot)
            t2 = time.perf_counter()
            results, retries, retry_cached, retry_input = await self._branches(
                prepared.prefix, prepared.branches, prepared.images, slot
            )
            t3 = time.perf_counter()
        finally:
            if self.slots and slot is not None:
                self.slots.release(slot)

        answers = {}
        cached = warm.cached_tokens + retry_cached
        input_tokens = warm.total_prompt_tokens + retry_input
        for b, gen in results:
            logprobs = [gen.logprobs[token_id] for token_id in b.label_ids]
            try:
                answers[b.question_id] = answer(b, logprobs, self.settings.temperature)
            except ValueError as exc:
                raise BackendError(f"Question {b.question_id!r}: {exc}") from exc
            best = b.labels[max(range(len(logprobs)), key=logprobs.__getitem__)]
            if gen.sampled not in b.labels:
                logger.warning("grammar did not constrain output: %r not in %s", gen.sampled, b.labels)
            elif gen.sampled != best:
                logger.warning("sampled %r but readout argmax is %r", gen.sampled, best)
            cached += gen.cached_tokens
            input_tokens += gen.total_prompt_tokens

        return Evaluation(
            response=SystemOneResponse(
                model=request.model,
                answers=answers,
                usage=Usage(input_tokens=input_tokens, output_tokens=len(results) + 1),
            ),
            prefix_tokens=warm.total_prompt_tokens,
            cached_tokens=cached,
            readout_retries=retries,
            slot=slot,
            prepare_ms=(t1 - t0) * 1000,
            prefill_ms=(t2 - t1) * 1000,
            branches_ms=(t3 - t2) * 1000,
        )

    async def _branches(self, prefix: str, branches: list[Branch], images: list[str], slot: int | None):
        retries = 0
        # Failed readout attempts still ran on the backend; their token counts must be
        # billed, or usage/cached headers undercount real work whenever a retry happened.
        retry_cached = 0
        retry_input = 0

        async def run(b: Branch):
            nonlocal retries, retry_cached, retry_input
            depths = [max(self.settings.top_n, READOUT_PER_LABEL * len(b.labels))]
            depths += [d for d in READOUT_ESCALATION if d > depths[0]]
            for attempt, n_probs in enumerate(depths):
                gen = await self.backend.complete(
                    prefix + b.suffix,
                    images=images or None,
                    n_probs=n_probs,
                    grammar=b.grammar,
                    id_slot=slot,
                )
                missing = [label for label, tid in zip(b.labels, b.label_ids, strict=True) if tid not in gen.logprobs]
                if not missing:
                    return b, gen
                retries += 1
                retry_cached += gen.cached_tokens
                retry_input += gen.total_prompt_tokens
                logger.info("question %r: %d labels outside top-%d readout, retrying deeper", b.question_id, len(missing), n_probs)
            raise BackendError(
                f"Question {b.question_id!r}: labels {missing} not in the top-{depths[-1]} readout; "
                "the model assigns them negligible probability and llama-server cannot report exact values",
                502,
                "readout_truncated",
            )

        tasks = [asyncio.create_task(run(b)) for b in branches]
        try:
            results = await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        return results, retries, retry_cached, retry_input
