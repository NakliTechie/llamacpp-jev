"""N+1 orchestration: render once, warm the prefix, score every branch, renormalize."""

import asyncio
import logging
import math
import time
from dataclasses import dataclass

from .backend import BackendError, LlamaClient, Props
from .config import Settings
from .models import ErrorCode, SystemOneRequest, SystemOneResponse, Usage
from .prompts import ContentError, PromptCompiler, build_messages
from .scoring import answer

logger = logging.getLogger(__name__)

READOUT_PER_LABEL = 16


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
    truncated_labels: int
    prepare_ms: float
    prefill_ms: float
    branches_ms: float


class EvaluationService:
    def __init__(self, settings: Settings, compiler: PromptCompiler, backend: LlamaClient, props: Props):
        self.settings = settings
        self.compiler = compiler
        self.backend = backend
        self.props = props
        self.admission = asyncio.Semaphore(settings.max_concurrent_requests)
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
            return await self._evaluate(request)

    async def _evaluate(self, request: SystemOneRequest) -> Evaluation:
        t0 = time.perf_counter()
        try:
            messages, images, marker = build_messages(request, self.props.vision)
            rendered = await self.backend.apply_template(messages)
            prepared = self.compiler.compile(request, rendered, marker, images)
        except ContentError as exc:
            raise RequestError(str(exc), 422, "unsupported_content") from exc
        t1 = time.perf_counter()

        warm = await self.backend.complete(prepared.prefix, images=prepared.images or None)
        t2 = time.perf_counter()

        async def branch(b):
            # Readout depth scales with the option count: a 64-way question on a small model
            # regularly leaves labels outside the top 256 (observed on Qwen3.5-0.8B).
            n_probs = max(self.settings.top_n, READOUT_PER_LABEL * len(b.labels))
            return b, await self.backend.complete(
                b.prompt,
                images=prepared.images or None,
                n_probs=n_probs,
                grammar=b.grammar,
            )

        tasks = [asyncio.create_task(branch(b)) for b in prepared.branches]
        try:
            results = await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            raise
        t3 = time.perf_counter()

        answers = {}
        truncated = 0
        cached = warm.cached_tokens
        input_tokens = warm.total_prompt_tokens
        for b, gen in results:
            logprobs = [gen.logprobs.get(token_id, -math.inf) for token_id in b.label_ids]
            truncated += sum(1 for value in logprobs if value == -math.inf)
            try:
                answers[b.question_id] = answer(b, logprobs, self.settings.temperature)
            except ValueError as exc:
                raise BackendError(f"Question {b.question_id!r}: {exc}") from exc
            best = b.labels[max(range(len(logprobs)), key=logprobs.__getitem__)]
            if gen.sampled not in b.labels:
                logger.warning("grammar did not constrain output: %r not in %s", gen.sampled, b.labels)
            elif gen.sampled != best and logprobs[b.labels.index(gen.sampled)] != -math.inf:
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
            truncated_labels=truncated,
            prepare_ms=(t1 - t0) * 1000,
            prefill_ms=(t2 - t1) * 1000,
            branches_ms=(t3 - t2) * 1000,
        )
