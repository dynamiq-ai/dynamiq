import asyncio
import json
import time
from enum import Enum
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dynamiq.callbacks.base import BaseCallbackHandler
from dynamiq.connections import TypeSafe
from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes import Node, NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.llms import BaseLLM
from dynamiq.nodes.node import ensure_config
from dynamiq.nodes.schema_utils import apply_param_modes
from dynamiq.nodes.types import ActionType, Authored, InputParamMode, NamedField
from dynamiq.prompts import Message, Prompt
from dynamiq.runnables import RunnableConfig, RunnableResult, RunnableStatus
from dynamiq.types.cancellation import check_cancellation
from dynamiq.utils import generate_uuid
from dynamiq.utils.json_parser import parse_llm_json_output
from dynamiq.utils.logger import logger
from dynamiq.utils.utils import CHARS_PER_TOKEN

DESCRIPTION_JUDGEMENT = """Answers typed judgement questions about a state and returns calibrated probabilities.

Key Capabilities:
- Yes/no (noul), single-choice and ordered-score questions, each answered with a probability distribution and a confidence
- Judges a text, an object of named fields or a list of messages against the configured questions, or against questions given at call time
- Decisions ready for routing: a boolean for a yes/no question, the option or level name otherwise

Usage Strategy:
- Ask one specific thing per question and combine the answers; never ask a compound question
- Read the probability and the confidence before acting: a value near 0.5 or a low confidence means the state is ambiguous
- Pass the state as a string, an object or a list of turns; judge a part of a long document rather than all of it

Parameter Guide:
- state: The content to judge (required unless the node declares input fields)
- questions: Extra or replacement questions, each {"name", "type": "noul"|"choice"|"score", "instructions", "options": [{"name", "description"}]}

Examples:
- {"state": "My card was charged twice, fix it today", "questions": [{"name": "is_urgent", "type": "noul", "instructions": "The customer needs a response today"}]}
- {"state": {"subject": "Refund", "body": "..."}, "questions": [{"name": "team", "type": "choice", "instructions": "Which team handles this?", "options": [{"name": "billing"}, {"name": "support"}]}]}"""  # noqa: E501

SYSTEM_ONE_PATH = "/v1/systemone"
# The documented context limit for the state and the longest question together; a rough estimate here
# fails early with a clear message, and the API's own 422 stays the backstop.
SYSTEM_ONE_STATE_MAX_TOKENS = 32_000
# The API asks for a backoff on these; the platform's own retry has no way to read Retry-After.
_TRANSIENT_STATUSES = frozenset({408, 429, 500, 502, 503, 504, 529})
_MAX_ATTEMPTS = 3
_MAX_WAIT_SECONDS = 10.0
_MAX_OPTIONS = 255
_MAX_LEVELS = 10
_MAX_SAMPLES = 10
_EVIDENCE_CHARS = 500

JUDGE_PROMPT_TEMPLATE = """You are a calibrated judge. Answer every question below about the state, using only what the state contains; do not assume facts it does not state.

<state>
{state}
</state>

Questions:
{questions}

Reply with a single JSON object and nothing else. It must match this JSON schema:
{schema}
{guidance}"""  # noqa: E501
_VERBALIZED_GUIDANCE = (
    "Probabilities express how sure you are: put most of the weight on one answer when the state is clear, "
    "and spread it when the state is ambiguous or silent on the question."
)
_RATIONALE_GUIDANCE = "The rationale quotes or names what in the state decided the answer, in one or two sentences."


class QuestionType(str, Enum):
    """What a question asks for: a yes/no probability, one option of a set, or one level of an ordered scale."""

    NOUL = "noul"
    CHOICE = "choice"
    SCORE = "score"


class ConfidenceMode(str, Enum):
    """Where an LLM or agent judge's probabilities come from.

    `verbalized` asks the judge to state a distribution with its answer. `sampling` asks it to answer several
    times and takes the vote shares, which costs `samples` calls but does not depend on the judge's own sense
    of its uncertainty. A System One judge returns calibrated probabilities in one call and uses neither.
    """

    VERBALIZED = "verbalized"
    SAMPLING = "sampling"


class JudgementOption(Authored):
    """One option of a choice question, or one level of a score question."""

    id: str = Field(default_factory=generate_uuid)
    name: str
    description: str | None = None


class JudgementQuestion(Authored):
    """One question a Judgement node asks about the state.

    `name` keys the answer and must be an identifier, so downstream nodes read it by path. A choice question
    lists its `options`; a score question lists its levels as `options` from lowest to highest. A yes/no
    question may say what counts as a yes (`yes_when`) and as a no (`no_when`) where the boundary is not obvious.
    """

    id: str = Field(default_factory=generate_uuid)
    name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    type: QuestionType = QuestionType.NOUL
    instructions: str
    options: list[JudgementOption] = []
    yes_when: str | None = None
    no_when: str | None = None
    enabled: bool = True

    @field_validator("instructions")
    @classmethod
    def instructions_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("instructions must not be empty")
        return value

    @model_validator(mode="after")
    def options_fit_type(self):
        if self.type == QuestionType.NOUL:
            return self
        names = [option.name.strip() for option in self.options]
        limit = _MAX_OPTIONS if self.type == QuestionType.CHOICE else _MAX_LEVELS
        what = "options" if self.type == QuestionType.CHOICE else "levels"
        if not 2 <= len(names) <= limit:
            raise ValueError(f"question {self.name!r} needs between 2 and {limit} {what}, got {len(names)}")
        if any(not name for name in names):
            raise ValueError(f"question {self.name!r} has an unnamed {what[:-1]}")
        if len(set(names)) != len(names):
            raise ValueError(f"question {self.name!r} names {'an option' if what == 'options' else 'a level'} twice")
        return self

    @property
    def option_names(self) -> list[str]:
        return [option.name.strip() for option in self.options]


class JudgementInputSchema(BaseModel):
    model_config = ConfigDict(extra="allow")

    state: str | dict[str, Any] | list[Any] | None = Field(
        default=None,
        description=(
            "The content to judge: a text, an object of named fields or a list of messages. "
            "Required unless the node declares input fields, which are then read by name."
        ),
    )
    questions: list[JudgementQuestion] | None = Field(
        default=None,
        description=(
            "Questions to ask, by name: one replaces the configured question with the same name and the others "
            "are added. Each is {name, type: noul|choice|score, instructions, options: [{name, description}], "
            "yes_when, no_when}; options are the choices of a choice question or the levels of a score question, "
            "low to high."
        ),
    )


def confidence_of(probabilities: list[float]) -> float:
    """How concentrated a distribution is, from 0 (uniform) to 1 (one outcome).

    The same measure System One reports with its answers, so a threshold tuned on one judge holds on another:
    for `n` outcomes, `(n * max - 1) / (n - 1)`; a yes/no answer is the two-outcome case, `|2p - 1|`.
    """
    count = len(probabilities)
    if count < 2:
        return 1.0
    return max(0.0, min(1.0, (count * max(probabilities) - 1) / (count - 1)))


def _weights(names: list[str], raw: Any) -> dict[str, float]:
    """A non-negative weight per name, tolerating the case and whitespace an LLM may change in its keys."""
    raw = raw if isinstance(raw, dict) else {}
    folded = {str(key).strip().lower(): value for key, value in raw.items()}
    values: dict[str, float] = {}
    for name in names:
        try:
            value = float(raw[name] if name in raw else folded.get(name.strip().lower(), 0))
        except (TypeError, ValueError):
            value = 0.0
        values[name] = max(value, 0.0)
    return values


def _distribution(names: list[str], raw: Any, anchor: str | None) -> dict[str, float]:
    """A probability per name, normalized to sum to 1; a missing or unusable distribution becomes the anchor."""
    values = _weights(names, raw)
    total = sum(values.values())
    if total <= 0:
        if anchor not in names:
            raise ValueError("no usable probabilities and no answer")
        return {name: 1.0 if name == anchor else 0.0 for name in names}
    return {name: value / total for name, value in values.items()}


def _probability(value: Any) -> float:
    """A probability read from a judge, refused when it is not a number between 0 and 1."""
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    probability = float(value)
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"probability {probability} is outside 0..1")
    return probability


def _weighted_index(distribution: dict[str, float]) -> float:
    return sum(index * probability for index, probability in enumerate(distribution.values()))


def _top(distribution: dict[str, float]) -> str:
    return max(distribution, key=distribution.__getitem__)


class _Verdict(BaseModel):
    """One question answered: a distribution over the outcomes, and what the judge said about it."""

    question: JudgementQuestion
    distribution: dict[str, float]
    confidence: float | None = None
    score: float | None = None
    rationale: str | None = None


class Judgement(Node):
    """Asks typed questions about a state and returns calibrated answers a workflow can route on.

    The node is the contract; the judge behind it is pluggable. A `connection` to TypeSafe sends the questions
    to a System One model, which answers every one in a single call with probabilities calibrated for that
    purpose. A `judge` node instead asks an LLM, with a JSON schema built from the questions, or an agent,
    which may use its tools before answering. Exactly one of the two is set. The three read the same questions
    and produce the same output, so a flow can start on an LLM and move to System One, or the reverse,
    without touching the nodes after it.

    The state is what the question is about: a text, an object or a list of messages. It comes in as `state`,
    or, when the node declares `input_fields`, as those fields by name, which is how a workflow wires a record
    in without templating it. Questions are configured on the node and may be replaced or extended at call
    time by name, which is how an agent asks its own; set `allow_agent_questions` off to keep the rubric fixed.

    Every answer carries a probability distribution and a `confidence` in it, computed the same way for every
    judge (see `confidence_of`). `decisions` holds one value per question ready for a Choice or a Decision
    Table: a boolean for a yes/no question, decided by `noul_threshold`, the option or the level name
    otherwise. `min_confidence` names the answers whose confidence falls below it under `low_confidence` and
    sets `needs_review`, so uncertain records reach a person instead of a default branch. An LLM or agent
    judge can also explain itself: `include_rationale` adds one per answer, and an agent's tool calls are
    returned as `evidence`.
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str | None = "judgement"
    description: str = DESCRIPTION_JUDGEMENT
    action_type: ActionType = ActionType.JUDGEMENT
    is_parallel_execution_allowed: bool = True

    connection: TypeSafe | None = None
    judge: Node | None = Field(default=None, description="An LLM or an agent node that answers the questions.")
    model: str = Field(default="jev-latest", description="The System One model; ignored by an LLM or agent judge.")
    input_fields: list[NamedField] = []
    questions: list[JudgementQuestion] = []
    noul_threshold: float = Field(default=0.5, ge=0, le=1, description="A yes/no probability at or above it is a yes.")
    min_confidence: float | None = Field(
        default=None, ge=0, le=1, description="An answer with a confidence below it needs review."
    )
    allow_agent_questions: bool = Field(
        default=True, description="Whether an agent using the node as a tool may ask questions of its own."
    )
    confidence_mode: ConfidenceMode = ConfidenceMode.VERBALIZED
    samples: int = Field(default=1, ge=1, le=_MAX_SAMPLES, description="Answers averaged in sampling mode.")
    include_rationale: bool = Field(default=False, description="Ask an LLM or agent judge to explain each answer.")
    timeout: float = Field(default=30, gt=0, description="Seconds to wait for a System One answer.")
    input_cost_per_million_tokens: float = Field(
        default=0.042, ge=0, description="What System One charges per million input tokens, for usage tracking."
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)
    input_schema: ClassVar[type[JudgementInputSchema]] = JudgementInputSchema

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Hidden from the agent-facing schema only; a workflow may still pass questions from upstream.
        if not self.allow_agent_questions:
            modes = self.input_param_modes | {"questions": InputParamMode.HIDDEN}
            self._resolved_input_schema = apply_param_modes(self.input_schema, modes)

    @model_validator(mode="after")
    def validate_judge(self):
        if (self.connection is None) == (self.judge is None):
            raise ValueError(
                f"Judgement '{self.name}' needs exactly one judge: a TypeSafe connection, or a judge node "
                "(an LLM or an agent)"
            )
        if self.judge is not None and not (isinstance(self.judge, BaseLLM) or self.judge.group == NodeGroup.AGENTS):
            raise ValueError(
                f"Judgement '{self.name}': the judge must be an LLM or an agent node, "
                f"not {type(self.judge).__name__}"
            )
        if self.confidence_mode == ConfidenceMode.SAMPLING:
            if self.connection is not None:
                raise ValueError(
                    f"Judgement '{self.name}': sampling needs an LLM or agent judge; "
                    "System One returns calibrated probabilities in one call"
                )
            if self.samples < 2:
                raise ValueError(f"Judgement '{self.name}': sampling needs at least 2 samples")
        names = [question.name for question in self.questions]
        if duplicates := sorted({name for name in names if names.count(name) > 1}):
            raise ValueError(f"Judgement '{self.name}': question names used more than once: {', '.join(duplicates)}")
        return self

    def init_components(self, connection_manager: ConnectionManager | None = None):
        connection_manager = connection_manager or ConnectionManager()
        super().init_components(connection_manager)
        if self.judge is not None and self.judge.is_postponed_component_init:
            self.judge.init_components(connection_manager)

    @property
    def to_dict_exclude_params(self):
        return super().to_dict_exclude_params | {"judge": True}

    def to_dict(self, **kwargs) -> dict:
        data = super().to_dict(**kwargs)
        # The judge that is not in use is left out rather than written as null: a YAML dump reads a connection
        # entry by its id, and a flow reads better without the slot it does not fill.
        if self.connection is None:
            data.pop("connection", None)
        if self.judge is None:
            data.pop("judge", None)
        else:
            data["judge"] = self.judge.to_dict(**kwargs)
        return data

    @property
    def backend(self) -> str:
        if self.connection is not None:
            return "system_one"
        return "llm" if isinstance(self.judge, BaseLLM) else "agent"

    def execute(self, input_data: JudgementInputSchema, config: RunnableConfig = None, **kwargs) -> dict[str, Any]:
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)
        questions = self._active_questions(input_data.questions)
        state = self._state_of(input_data)

        if self.connection is not None:
            data = self._send_system_one(self._system_one_payload(state, questions), config)
            verdicts = self._read_system_one(data, questions)
            self.run_on_node_execute_run(config.callbacks, usage_data=self._usage_of(data), **kwargs)
            return self._output(verdicts, model=str(data.get("model") or self.model), usage=self._usage_summary(data))

        runs = [self._run_judge(state, questions, config, **kwargs) for _ in range(self._sample_count())]
        return self._judge_output(runs, questions)

    async def execute_async(
        self, input_data: JudgementInputSchema, config: RunnableConfig = None, **kwargs
    ) -> dict[str, Any]:
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)
        questions = self._active_questions(input_data.questions)
        state = self._state_of(input_data)

        if self.connection is not None:
            data = await self._send_system_one_async(self._system_one_payload(state, questions), config)
            verdicts = self._read_system_one(data, questions)
            self.run_on_node_execute_run(config.callbacks, usage_data=self._usage_of(data), **kwargs)
            return self._output(verdicts, model=str(data.get("model") or self.model), usage=self._usage_summary(data))

        runs = await asyncio.gather(
            *(self._run_judge_async(state, questions, config, **kwargs) for _ in range(self._sample_count()))
        )
        return self._judge_output(list(runs), questions)

    def _active_questions(self, call_questions: list[JudgementQuestion] | None) -> list[JudgementQuestion]:
        merged = {question.name: question for question in self.questions}
        for question in call_questions or []:
            merged[question.name] = question
        active = [question for question in merged.values() if question.enabled]
        if not active:
            raise ToolExecutionException(
                f"Judgement '{self.name}' has no enabled question; configure one or pass `questions`.",
                recoverable=True,
            )
        return active

    def _state_of(self, input_data: JudgementInputSchema) -> str | dict[str, Any] | list[Any]:
        if input_data.state is not None:
            return input_data.state
        if not self.input_fields:
            raise ToolExecutionException(
                f"Judgement '{self.name}' has nothing to judge: pass `state`, or declare input fields on the node.",
                recoverable=True,
            )
        provided = input_data.model_extra or {}
        state: dict[str, Any] = {}
        for field in self.input_fields:
            if field.name not in provided:
                logger.warning(f"Judgement '{self.name}': input field {field.name!r} was not provided")
            state[field.name] = provided.get(field.name)
        return state

    def _sample_count(self) -> int:
        return self.samples if self.confidence_mode == ConfidenceMode.SAMPLING else 1

    # -- System One -------------------------------------------------------------------------------------------

    def _system_one_payload(self, state: Any, questions: list[JudgementQuestion]) -> dict[str, Any]:
        text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, default=str)
        if len(text) / CHARS_PER_TOKEN > SYSTEM_ONE_STATE_MAX_TOKENS:
            raise ToolExecutionException(
                f"Judgement '{self.name}': the state is about {len(text) // CHARS_PER_TOKEN:,} tokens; System One "
                f"reads at most {SYSTEM_ONE_STATE_MAX_TOKENS:,} with a question. Judge a part of it, or summarize it "
                "first.",
                recoverable=True,
            )
        return {
            "model": self.model,
            "state": state,
            "questions": {question.name: self._wire_question(question) for question in questions},
        }

    @staticmethod
    def _wire_question(question: JudgementQuestion) -> dict[str, Any]:
        wire: dict[str, Any] = {"type": question.type.value, "instructions": question.instructions}
        if question.type == QuestionType.NOUL:
            criteria = {key: text for key, text in (("true", question.yes_when), ("false", question.no_when)) if text}
            if criteria:
                wire["criteria"] = criteria
        elif question.type == QuestionType.CHOICE:
            wire["criteria"] = {option.name.strip(): option.description or option.name for option in question.options}
        else:
            wire["criteria"] = [option.description or option.name for option in question.options]
        return wire

    def _endpoint(self) -> tuple[str, dict[str, str]]:
        url = self.connection.url.rstrip("/") + SYSTEM_ONE_PATH
        return url, {"Authorization": f"Bearer {self.connection.api_key}"}

    def _send_system_one(self, payload: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
        requests = self.connection.connect()
        url, headers = self._endpoint()
        for attempt in range(_MAX_ATTEMPTS):
            check_cancellation(config)
            try:
                response = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
            except requests.RequestException as e:
                if attempt + 1 == _MAX_ATTEMPTS:
                    raise self._transport_error(e)
                time.sleep(self._wait_seconds(attempt, {}))
                continue
            if response.status_code in _TRANSIENT_STATUSES and attempt + 1 < _MAX_ATTEMPTS:
                time.sleep(self._wait_seconds(attempt, response.headers))
                continue
            return self._read_response(response.status_code, response.text)
        raise AssertionError("unreachable")

    async def _send_system_one_async(self, payload: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
        import httpx

        url, headers = self._endpoint()
        async with await self.connection.connect_async() as client:
            for attempt in range(_MAX_ATTEMPTS):
                check_cancellation(config)
                try:
                    response = await client.post(url, json=payload, headers=headers, timeout=self.timeout)
                except httpx.HTTPError as e:
                    if attempt + 1 == _MAX_ATTEMPTS:
                        raise self._transport_error(e)
                    await asyncio.sleep(self._wait_seconds(attempt, {}))
                    continue
                if response.status_code in _TRANSIENT_STATUSES and attempt + 1 < _MAX_ATTEMPTS:
                    await asyncio.sleep(self._wait_seconds(attempt, response.headers))
                    continue
                return self._read_response(response.status_code, response.text)
        raise AssertionError("unreachable")

    @staticmethod
    def _wait_seconds(attempt: int, headers: Any) -> float:
        """How long to wait before the next attempt: what the server asked for, else a doubling backoff."""
        wait = 0.5 * 2**attempt
        for header, scale in (("retry-after-ms", 1000.0), ("Retry-After", 1.0)):
            try:
                wait = float(headers.get(header)) / scale
                break
            except (AttributeError, TypeError, ValueError):
                continue
        return max(0.0, min(wait, _MAX_WAIT_SECONDS))

    def _transport_error(self, error: Exception) -> ToolExecutionException:
        logger.error(f"Judgement '{self.name}': System One request failed. Error: {error}")
        return ToolExecutionException(
            f"Judgement '{self.name}': the System One request failed ({error}); retry later.", recoverable=True
        )

    def _read_response(self, status: int, text: str) -> dict[str, Any]:
        if status == 200:
            try:
                data = json.loads(text)
            except ValueError:
                data = None
            if not isinstance(data, dict):
                raise ToolExecutionException(
                    f"Judgement '{self.name}': System One returned a body that is not a JSON object.",
                    recoverable=True,
                )
            return data
        detail = self._error_detail(text)
        if status in (401, 403):
            # A credential problem does not go away by asking again.
            raise ValueError(f"Judgement '{self.name}': System One rejected the API key (HTTP {status}): {detail}")
        if status == 422:
            message = f"System One rejected the request (HTTP 422): {detail}"
        elif status in _TRANSIENT_STATUSES:
            message = f"System One is rate limited or unavailable (HTTP {status}); retry later. {detail}".rstrip()
        else:
            message = f"System One answered HTTP {status}: {detail}"
        logger.error(f"Judgement '{self.name}': {message}")
        raise ToolExecutionException(f"Judgement '{self.name}': {message}", recoverable=True)

    @staticmethod
    def _error_detail(text: str) -> str:
        try:
            body = json.loads(text)
        except ValueError:
            body = None
        if isinstance(body, dict):
            error = body.get("error")
            detail = error.get("message") if isinstance(error, dict) else body.get("detail") or body.get("message")
            if detail:
                return detail if isinstance(detail, str) else json.dumps(detail)
        return text.strip()[:_EVIDENCE_CHARS]

    def _read_system_one(self, data: dict[str, Any], questions: list[JudgementQuestion]) -> list[_Verdict]:
        answers = data.get("answers")
        if not isinstance(answers, dict):
            raise ToolExecutionException(f"Judgement '{self.name}': System One returned no answers.", recoverable=True)
        verdicts = []
        for question in questions:
            try:
                verdicts.append(self._system_one_verdict(question, answers.get(question.name)))
            except (TypeError, ValueError, AttributeError, LookupError) as e:
                raise ToolExecutionException(
                    f"Judgement '{self.name}': System One's answer to {question.name!r} is unreadable: {e}",
                    recoverable=True,
                ) from e
        return verdicts

    @staticmethod
    def _system_one_verdict(question: JudgementQuestion, answer: Any) -> _Verdict:
        if not isinstance(answer, dict):
            raise ValueError("no answer")
        if answer.get("type") != question.type.value:
            raise ValueError(f"answer type {answer.get('type')!r} is not {question.type.value!r}")
        confidence = answer.get("confidence")
        confidence = float(confidence) if isinstance(confidence, (int, float)) else None
        if question.type == QuestionType.NOUL:
            probability = _probability(answer["noul"])
            return _Verdict(question=question, distribution={"yes": probability, "no": 1 - probability})
        names = question.option_names
        if question.type == QuestionType.CHOICE:
            distribution = _distribution(names, answer.get("probabilities"), answer.get("choice"))
            return _Verdict(question=question, distribution=distribution, confidence=confidence)
        # Score probabilities and the legend are keyed by level index; the level names are the node's own.
        by_name = {names[int(index)]: value for index, value in (answer.get("probabilities") or {}).items()}
        distribution = _distribution(names, by_name, None)
        score = answer.get("score")
        return _Verdict(
            question=question,
            distribution=distribution,
            confidence=confidence,
            score=float(score) if isinstance(score, (int, float)) else None,
        )

    def _usage_of(self, data: dict[str, Any]) -> dict[str, Any]:
        """Usage in the shape LLM nodes report, so the platform's cost tracking needs no special case."""
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        input_tokens = int(usage.get("input_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or 0)
        cost = input_tokens / 1_000_000 * self.input_cost_per_million_tokens
        return {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "prompt_tokens_cost_usd": cost,
            "completion_tokens_cost_usd": 0.0,
            "total_tokens_cost_usd": cost,
        }

    def _usage_summary(self, data: dict[str, Any]) -> dict[str, Any]:
        usage = self._usage_of(data)
        return {
            "input_tokens": usage["prompt_tokens"],
            "output_tokens": usage["completion_tokens"],
            "cost_usd": usage["total_tokens_cost_usd"],
        }

    # -- LLM and agent judges ---------------------------------------------------------------------------------

    def _verbalized(self) -> bool:
        return self.confidence_mode == ConfidenceMode.VERBALIZED

    def _response_schema(self, questions: list[JudgementQuestion]) -> dict[str, Any]:
        properties = {}
        for question in questions:
            fields: dict[str, Any] = {}
            if question.type == QuestionType.NOUL:
                fields["answer"] = {"type": "boolean", "description": "true when the answer is yes"}
                if self._verbalized():
                    fields["probability"] = {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "The probability that the answer is yes",
                    }
            else:
                names = question.option_names
                what = "option" if question.type == QuestionType.CHOICE else "level"
                fields["answer"] = {"type": "string", "enum": names, "description": f"The {what} that fits best"}
                if self._verbalized():
                    fields["probabilities"] = {
                        "type": "object",
                        "properties": {name: {"type": "number", "minimum": 0, "maximum": 1} for name in names},
                        "required": names,
                        "additionalProperties": False,
                        "description": f"The probability of each {what}; they sum to 1",
                    }
            if self.include_rationale:
                fields["rationale"] = {"type": "string", "description": "What in the state decided the answer"}
            properties[question.name] = {
                "type": "object",
                "properties": fields,
                "required": list(fields),
                "additionalProperties": False,
            }
        return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}

    def _brief(self, state: Any, questions: list[JudgementQuestion], schema: dict[str, Any]) -> str:
        lines = []
        for number, question in enumerate(questions, start=1):
            if question.type == QuestionType.NOUL:
                lines.append(f"{number}. {question.name} (yes/no): {question.instructions.strip()}")
                if question.yes_when:
                    lines.append(f"   Yes when: {question.yes_when.strip()}")
                if question.no_when:
                    lines.append(f"   No when: {question.no_when.strip()}")
            else:
                kind = "choose one" if question.type == QuestionType.CHOICE else "score on levels from low to high"
                lines.append(f"{number}. {question.name} ({kind}): {question.instructions.strip()}")
                for option in question.options:
                    described = f": {option.description.strip()}" if option.description else ""
                    lines.append(f"   - {option.name.strip()}{described}")
        guidance = []
        if self._verbalized():
            guidance.append(_VERBALIZED_GUIDANCE)
        if self.include_rationale:
            guidance.append(_RATIONALE_GUIDANCE)
        return JUDGE_PROMPT_TEMPLATE.format(
            state=state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, indent=2, default=str),
            questions="\n".join(lines),
            schema=json.dumps(schema, indent=2),
            guidance="\n".join(guidance),
        )

    def _judge_for_run(self) -> Node:
        # Never the configured judge itself: samples run concurrently on the async path, and
        # `is_parallel_execution_allowed` lets a calling agent invoke this node twice at once. A judge
        # carries per-run state - an agent resets its loop state and rebuilds its prompt on every
        # execute - so a shared instance would let one run wipe another's out from under it.
        return self.judge.clone()

    def _judge_call(self, state: Any, questions: list[JudgementQuestion], **kwargs) -> tuple[dict, dict]:
        schema = self._response_schema(questions)
        brief = self._brief(state, questions, schema)
        child_kwargs = kwargs | {"parent_run_id": kwargs.get("run_id"), "run_depends": []}
        if isinstance(self.judge, BaseLLM):
            input_data = {}
            child_kwargs |= {
                "prompt": Prompt(messages=[Message(role="user", content=brief, static=True)]),
                "response_format": {"type": "json_schema", "json_schema": {"name": "judgement", "schema": schema}},
            }
        else:
            input_data = {"input": brief}
        return input_data, child_kwargs

    def _run_judge(self, state: Any, questions: list[JudgementQuestion], config: RunnableConfig, **kwargs) -> dict:
        input_data, child_kwargs = self._judge_call(state, questions, **kwargs)
        recorder = _EvidenceRecorder()
        check_cancellation(config)
        result = self._judge_for_run().run(input_data=input_data, config=recorder.config_for(config), **child_kwargs)
        return self._judge_answers(result, recorder)

    async def _run_judge_async(
        self, state: Any, questions: list[JudgementQuestion], config: RunnableConfig, **kwargs
    ) -> dict:
        input_data, child_kwargs = self._judge_call(state, questions, **kwargs)
        recorder = _EvidenceRecorder()
        check_cancellation(config)
        result = await self._judge_for_run().run_async(
            input_data=input_data, config=recorder.config_for(config), **child_kwargs
        )
        return self._judge_answers(result, recorder)

    def _judge_answers(self, result: RunnableResult, recorder: "_EvidenceRecorder") -> dict:
        if result.status != RunnableStatus.SUCCESS:
            # Recoverable whatever the judge's own verdict on its error: an agent using the node as a tool can ask
            # again or judge a smaller state, where a hard failure would end its run.
            raise ToolExecutionException(
                f"Judgement '{self.name}': the judge failed: {result.error.message if result.error else ''}",
                recoverable=True,
            )
        content = result.output.get("content")
        try:
            parsed = content if isinstance(content, dict) else parse_llm_json_output(str(content))
        except ValueError as e:
            raise ToolExecutionException(
                f"Judgement '{self.name}': the judge did not answer with JSON: {e}", recoverable=True
            ) from e
        if not isinstance(parsed, dict):
            raise ToolExecutionException(
                f"Judgement '{self.name}': the judge answered with a JSON {type(parsed).__name__}, not an object.",
                recoverable=True,
            )
        return {"answers": parsed, "evidence": recorder.evidence}

    def _judge_output(self, runs: list[dict], questions: list[JudgementQuestion]) -> dict[str, Any]:
        verdicts = []
        for question in questions:
            try:
                verdicts.append(self._judge_verdict(question, [run["answers"].get(question.name) for run in runs]))
            except (TypeError, ValueError, AttributeError, LookupError) as e:
                raise ToolExecutionException(
                    f"Judgement '{self.name}': the judge's answer to {question.name!r} is unreadable: {e}",
                    recoverable=True,
                ) from e
        evidence = [item for run in runs for item in run["evidence"]]
        return self._output(verdicts, model=self._judge_model(), evidence=evidence if self.backend == "agent" else None)

    def _judge_verdict(self, question: JudgementQuestion, answers: list[Any]) -> _Verdict:
        if any(not isinstance(answer, dict) for answer in answers):
            raise ValueError("no answer")
        names = ["yes", "no"] if question.type == QuestionType.NOUL else question.option_names
        if self._verbalized():
            answer = answers[0]
            # The answer is what a distribution falls back on, so read it only when one cannot stand on its own.
            if question.type == QuestionType.NOUL:
                probability = (
                    _probability(answer["probability"])
                    if "probability" in answer
                    else float(self._chosen(question, answer.get("answer")) == "yes")
                )
                distribution = {"yes": probability, "no": 1 - probability}
            else:
                raw = answer.get("probabilities")
                anchor = None
                if sum(_weights(names, raw).values()) <= 0 and "answer" in answer:
                    anchor = self._chosen(question, answer["answer"])
                distribution = _distribution(names, raw, anchor)
            rationale = answer.get("rationale")
        else:
            votes = {name: 0.0 for name in names}
            for answer in answers:
                votes[self._chosen(question, answer.get("answer"))] += 1 / len(answers)
            distribution = votes
            top = _top(distribution)
            # The rationale of the first sample that agrees with the verdict, so it explains the answer given.
            rationale = next(
                (
                    a.get("rationale")
                    for a in answers
                    if self._chosen(question, a.get("answer")) == top and a.get("rationale")
                ),
                None,
            )
        return _Verdict(
            question=question, distribution=distribution, rationale=rationale if isinstance(rationale, str) else None
        )

    @staticmethod
    def _chosen(question: JudgementQuestion, answer: Any) -> str:
        """The outcome an answer names, tolerating the case and whitespace an LLM may change."""
        if question.type == QuestionType.NOUL:
            if isinstance(answer, bool):
                return "yes" if answer else "no"
            text = str(answer).strip().lower()
            if text in ("yes", "true", "no", "false"):
                return "yes" if text in ("yes", "true") else "no"
            raise ValueError(f"{answer!r} is not a yes or a no")
        names = question.option_names
        text = str(answer).strip()
        if text in names:
            return text
        for name in names:
            if name.lower() == text.lower():
                return name
        raise ValueError(f"{answer!r} is not one of {', '.join(names)}")

    def _judge_model(self) -> str:
        llm = self.judge if isinstance(self.judge, BaseLLM) else getattr(self.judge, "llm", None)
        return str(getattr(llm, "model", None) or self.judge.name or self.judge.type)

    # -- Output -----------------------------------------------------------------------------------------------

    def _output(
        self,
        verdicts: list[_Verdict],
        model: str,
        usage: dict[str, Any] | None = None,
        evidence: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        answers: dict[str, dict[str, Any]] = {}
        decisions: dict[str, Any] = {}
        rationale: dict[str, str] = {}
        for verdict in verdicts:
            question = verdict.question
            confidence = (
                verdict.confidence
                if verdict.confidence is not None
                else confidence_of(list(verdict.distribution.values()))
            )
            answer: dict[str, Any] = {"type": question.type.value}
            if question.type == QuestionType.NOUL:
                probability = verdict.distribution["yes"]
                decisions[question.name] = probability >= self.noul_threshold
                answer |= {"probability": probability, "decision": decisions[question.name]}
            elif question.type == QuestionType.CHOICE:
                decisions[question.name] = _top(verdict.distribution)
                answer |= {"choice": decisions[question.name], "probabilities": verdict.distribution}
            else:
                level = _top(verdict.distribution)
                decisions[question.name] = level
                answer |= {
                    "level": level,
                    "index": question.option_names.index(level),
                    "score": verdict.score if verdict.score is not None else _weighted_index(verdict.distribution),
                    "probabilities": verdict.distribution,
                }
            answer["confidence"] = confidence
            if verdict.rationale:
                answer["rationale"] = verdict.rationale
                rationale[question.name] = verdict.rationale
            answers[question.name] = answer

        low_confidence = [
            name
            for name, answer in answers.items()
            if self.min_confidence is not None and answer["confidence"] < self.min_confidence
        ]
        confidence = min(answer["confidence"] for answer in answers.values())
        return {
            "content": self._render(answers, low_confidence, model),
            "answers": answers,
            "decisions": decisions,
            "confidence": confidence,
            "needs_review": bool(low_confidence),
            "low_confidence": low_confidence,
            "model": model,
            "backend": self.backend,
            "confidence_source": "model" if self.connection is not None else self.confidence_mode.value,
            "usage": usage,
            "rationale": rationale or None,
            "evidence": evidence,
        }

    def _render(self, answers: dict[str, dict[str, Any]], low_confidence: list[str], model: str) -> str:
        """The answers as a short report, which is what an agent reads when it calls the node as a tool."""
        lines = [f"Judgement by {model}:"]
        for name, answer in answers.items():
            confidence = f"confidence {answer['confidence']:.2f}"
            if answer["type"] == QuestionType.NOUL.value:
                verdict = "yes" if answer["decision"] else "no"
                lines.append(f"- {name}: {verdict} (probability of yes {answer['probability']:.2f}, {confidence})")
            else:
                spread = ", ".join(f"{option} {p:.2f}" for option, p in answer["probabilities"].items())
                if answer["type"] == QuestionType.CHOICE.value:
                    lines.append(f"- {name}: {answer['choice']} ({confidence}; {spread})")
                else:
                    lines.append(f"- {name}: {answer['level']} (score {answer['score']:.2f}, {confidence}; {spread})")
            if answer.get("rationale"):
                lines.append(f"  Rationale: {answer['rationale']}")
        if self.min_confidence is not None:
            lines.append(
                f"Needs review (confidence below {self.min_confidence:.2f}): {', '.join(low_confidence) or 'none'}"
            )
        return "\n".join(lines)


class _EvidenceRecorder(BaseCallbackHandler):
    """Collects the tool calls an agent judge makes, so a verdict says what it was based on.

    Registered on the judge's run alone: the config it returns carries the caller's callbacks plus this one, and
    the judge is an LLM or an agent, never a tool, so only the tools it uses are recorded.
    """

    def __init__(self):
        self.evidence: list[dict[str, Any]] = []

    def config_for(self, config: RunnableConfig) -> RunnableConfig:
        return config.model_copy(update={"callbacks": [*config.callbacks, self]})

    def on_node_end(self, serialized: dict[str, Any], output_data: dict[str, Any], **kwargs: Any) -> None:
        if serialized.get("group") != NodeGroup.TOOLS:
            return
        content = output_data.get("content") if isinstance(output_data, dict) else output_data
        text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, default=str)
        self.evidence.append(
            {
                "tool": serialized.get("name"),
                "output": text[:_EVIDENCE_CHARS] + ("..." if len(text) > _EVIDENCE_CHARS else ""),
            }
        )
