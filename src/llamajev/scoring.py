import math

from .models import ChoiceAnswer, NoulAnswer, NoulQuestion, ScoreAnswer, ScoreQuestion
from .prompts import Branch, serialize


def normalize(logprobs: list[float], temperature: float = 1.0) -> list[float]:
    if not logprobs or any(math.isnan(x) or x == math.inf for x in logprobs):
        raise ValueError("Invalid label logprobs from llama-server")
    peak = max(logprobs)
    if peak == -math.inf:
        raise ValueError("No label appeared in the backend's top-n readout")
    weights = [math.exp((value - peak) / temperature) for value in logprobs]
    total = math.fsum(weights)
    return [weight / total for weight in weights]


def confidence(probabilities: list[float]) -> float:
    """1 minus normalized entropy. TypeSafe's exact statistic is unpublished."""
    entropy = -math.fsum(p * math.log(p) for p in probabilities if p > 0)
    return min(1.0, max(0.0, 1 - entropy / math.log(len(probabilities))))


def answer(branch: Branch, logprobs: list[float], temperature: float):
    probabilities = normalize(logprobs, temperature)
    distribution = dict(zip(branch.option_keys, probabilities, strict=True))
    if isinstance(branch.question, NoulQuestion):
        return NoulAnswer(noul=distribution["true"])
    if isinstance(branch.question, ScoreQuestion):
        return ScoreAnswer(
            score=math.fsum(i * p for i, p in enumerate(probabilities)),
            legend={str(i): serialize(d) for i, d in enumerate(branch.question.criteria)},
            probabilities=distribution,
            confidence=confidence(probabilities),
        )
    return ChoiceAnswer(
        choice=max(distribution, key=distribution.__getitem__),
        probabilities=distribution,
        confidence=confidence(probabilities),
    )
