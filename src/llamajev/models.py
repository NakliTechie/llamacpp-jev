"""Wire contract for POST /v1/systemone — TypeSafe Jev field names, kept identical to openjev-sglang."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from .config import MAX_ANSWERS, MAX_QUESTIONS

Content = str | dict[str, JsonValue] | list[JsonValue]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class NoulCriteria(StrictModel):
    yes: Content = Field(default="Yes", alias="true", description="Meaning of a positive answer.")
    no: Content = Field(default="No", alias="false", description="Meaning of a negative answer.")


class NoulQuestion(StrictModel):
    type: Literal["noul"]
    instructions: Content = Field(description="The yes/no question or evaluation instructions.")
    criteria: NoulCriteria = Field(
        default_factory=NoulCriteria, description="Optional definitions of true and false."
    )


class ChoiceQuestion(StrictModel):
    type: Literal["choice"]
    instructions: Content = Field(description="Question or instructions for choosing one option.")
    criteria: dict[str, Content | None] = Field(
        min_length=2,
        max_length=MAX_ANSWERS,
        description=(
            "Map of 2-64 option keys to descriptions (string, object or array). The model sees "
            "only the description, or the option key when its description is null. Returned "
            "answers use the keys. TypeSafe allows 255 options; this server's single-token "
            "labels cap it at 64."
        ),
    )


class ScoreQuestion(StrictModel):
    type: Literal["score"]
    instructions: Content = Field(description="Question or instructions for applying the rubric.")
    criteria: list[Content] = Field(
        min_length=2,
        max_length=MAX_ANSWERS,
        description=(
            "Ordered rubric, lowest to highest (strings, objects or arrays). Scores use "
            "zero-based indices: three descriptions correspond to levels 0, 1, and 2."
        ),
    )


Question = Annotated[NoulQuestion | ChoiceQuestion | ScoreQuestion, Field(discriminator="type")]


class SystemOneRequest(StrictModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model": "jev-latest",
                "state": "I was charged twice. Please refund the duplicate.",
                "questions": {
                    "refund": {"type": "noul", "instructions": "Does the user request a refund?"},
                    "department": {
                        "type": "choice",
                        "instructions": "Which department should handle this?",
                        "criteria": {"billing": "Payments and refunds", "technical": "Software bugs"},
                    },
                    "urgency": {
                        "type": "score",
                        "instructions": "How urgent is this?",
                        "criteria": ["Routine", "Urgent", "Emergency"],
                    },
                },
            }
        }
    )
    state: Content = Field(
        description=(
            "Shared text, structured JSON, or a chat transcript to evaluate. Chat messages may "
            "carry image_url parts (data: URIs) when the backend was started with an mmproj."
        )
    )
    model: str = Field(min_length=1, description="jev-latest or the served model name.")
    questions: dict[str, Question] = Field(
        min_length=1,
        max_length=MAX_QUESTIONS,
        description="1-64 questions keyed by your own identifiers; the response preserves them.",
    )


class NoulAnswer(StrictModel):
    type: Literal["noul"] = "noul"
    noul: float = Field(ge=0, le=1, description="Probability of true, normalized over true/false.")


class ChoiceAnswer(StrictModel):
    type: Literal["choice"] = "choice"
    choice: str = Field(description="Option key with the highest probability.")
    probabilities: dict[str, float] = Field(description="Probability per option; sums to 1.")
    confidence: float = Field(
        ge=0, le=1, description="1 minus normalized entropy. Not calibrated correctness."
    )


class ScoreAnswer(StrictModel):
    type: Literal["score"] = "score"
    score: float = Field(ge=0, description="Expected zero-based level index.")
    legend: dict[str, str] = Field(description="Stringified level index to rubric description (structured levels serialized as JSON).")
    probabilities: dict[str, float] = Field(description="Probability per level index; sums to 1.")
    confidence: float = Field(
        ge=0, le=1, description="1 minus normalized entropy. Not calibrated correctness."
    )


class Usage(StrictModel):
    input_tokens: int = Field(
        ge=0, description="Backend prompt tokens summed over warm-up and branches, cached included."
    )
    output_tokens: int = Field(ge=0, description="N + 1 for N questions.")


class SystemOneResponse(StrictModel):
    model: str
    answers: dict[str, NoulAnswer | ChoiceAnswer | ScoreAnswer]
    usage: Usage


ErrorCode = Literal[
    "validation",
    "body_too_large",
    "unknown_model",
    "too_many_tokens",
    "unsupported_content",
    "backend_unreachable",
    "backend_error",
    "backend_timeout",
    "readout_truncated",
    "overloaded",
    "client_disconnected",
]


class ErrorDetail(StrictModel):
    message: str
    code: ErrorCode


class ErrorResponse(StrictModel):
    error: ErrorDetail


class HealthResponse(StrictModel):
    status: Literal["ok", "unavailable"]
    backend_url: str
    model: str | None = None
    n_slots: int | None = None
    n_ctx: int | None = None
    vision: bool | None = None
    labels_verified: int | None = None
    startup_seconds: float | None = None


class ModelsResponse(StrictModel):
    object: Literal["list"] = "list"
    data: list[dict[str, str]]
    models: list[dict[str, str]]


class LimitsResponse(StrictModel):
    max_answers_per_question: int
    max_questions: int
    max_body_bytes: int
    max_input_tokens: int
    max_concurrent_requests: int
    top_n: int
