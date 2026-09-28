import asyncio
import json
import time
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from dynamiq.connections import TypeSafe
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.node import ConnectionNode, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.types.cancellation import check_cancellation
from dynamiq.utils.logger import logger
from dynamiq.utils.utils import CHARS_PER_TOKEN

SYSTEM_ONE_PATH = "/v1/systemone"
# The documented context limit for the state and the longest question together; a rough estimate here
# fails early with a clear message, and the API's own 422 stays the backstop.
SYSTEM_ONE_STATE_MAX_TOKENS = 32_000
# The API asks for a backoff on these; the platform's own retry has no way to read Retry-After.
_TRANSIENT_STATUSES = frozenset({408, 429, 500, 502, 503, 504, 529})
_MAX_ATTEMPTS = 3
_MAX_WAIT_SECONDS = 10.0
_ERROR_CHARS = 500


def _tool_error(message: str) -> Exception:
    """Imported here, not at module scope: `dynamiq.nodes.agents` pulls in the tools package, which
    imports the Judgement node, which imports this one."""
    from dynamiq.nodes.agents.exceptions import ToolExecutionException

    return ToolExecutionException(message, recoverable=True)


class SystemOneInputSchema(BaseModel):
    state: Any = Field(default=None, description="The content to judge.")
    questions: dict[str, Any] = Field(default_factory=dict, description="Questions in wire form, keyed by name.")


class SystemOne(ConnectionNode):
    """Asks a TypeSafe System One model typed questions about a state and returns calibrated answers.

    System One answers every question in one call and reports a probability distribution over each
    question's outcomes, calibrated for that purpose rather than verbalized by a general model. The
    node carries what the service needs - a connection, a model and a request timeout - and returns
    the service's answers unread, so the caller decides what they mean.

    It is built to sit in a `Judgement` node's `judge` slot, the way an LLM node sits in an agent's
    `llm`, and is configured the same way any other provider node is.
    """

    group: Literal[NodeGroup.DETECTORS] = NodeGroup.DETECTORS
    name: str | None = "system-one"
    description: str = "Answers typed judgement questions with calibrated probabilities."
    connection: TypeSafe
    model: str = Field(default="jev-latest", description="The System One model.")
    timeout: float = Field(default=30, gt=0, description="Seconds to wait for an answer.")
    input_cost_per_million_tokens: float = Field(
        default=0.042, ge=0, description="What System One charges per million input tokens, for usage tracking."
    )

    input_schema: ClassVar[type[SystemOneInputSchema]] = SystemOneInputSchema

    def execute(self, input_data: SystemOneInputSchema, config: RunnableConfig = None, **kwargs) -> dict[str, Any]:
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)
        data = self.fetch(input_data.state, input_data.questions, config)
        self.run_on_node_execute_run(config.callbacks, usage_data=self.usage_of(data), **kwargs)
        return self.answer_of(data)

    async def execute_async(
        self, input_data: SystemOneInputSchema, config: RunnableConfig = None, **kwargs
    ) -> dict[str, Any]:
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)
        data = await self.fetch_async(input_data.state, input_data.questions, config)
        self.run_on_node_execute_run(config.callbacks, usage_data=self.usage_of(data), **kwargs)
        return self.answer_of(data)

    def answer_of(self, data: dict[str, Any]) -> dict[str, Any]:
        return {
            "model": str(data.get("model") or self.model),
            "answers": data.get("answers"),
            "usage": self.usage_summary(data),
        }

    def fetch(self, state: Any, questions: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
        """Send one request and return the service's body.

        Called directly by a `Judgement` rather than through `run()`, so that a rejected key stays the
        unrecoverable failure it is instead of being flattened into a failed child result.
        """
        requests = self.connection.connect()
        url, headers = self._endpoint()
        payload = self._payload(state, questions)
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

    async def fetch_async(self, state: Any, questions: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
        import httpx

        url, headers = self._endpoint()
        payload = self._payload(state, questions)
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

    def _payload(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, default=str)
        if len(text) / CHARS_PER_TOKEN > SYSTEM_ONE_STATE_MAX_TOKENS:
            raise _tool_error(
                f"System One '{self.name}': the state is about {len(text) // CHARS_PER_TOKEN:,} tokens; System One "
                f"reads at most {SYSTEM_ONE_STATE_MAX_TOKENS:,} with a question. Judge a part of it, or summarize it "
                "first."
            )
        return {
            "model": self.model,
            # What was measured is what goes on the wire. `default=str` renders a datetime, a Decimal
            # or a UUID that the transport's own encoder would refuse with a TypeError - which is
            # neither a requests nor an httpx error, so it would escape the retry loop entirely.
            "state": state if isinstance(state, str) else json.loads(text),
            "questions": questions,
        }

    def _endpoint(self) -> tuple[str, dict[str, str]]:
        url = self.connection.url.rstrip("/") + SYSTEM_ONE_PATH
        return url, {"Authorization": f"Bearer {self.connection.api_key}"}

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

    def _transport_error(self, error: Exception) -> Exception:
        logger.error(f"System One '{self.name}': request failed. Error: {error}")
        return _tool_error(f"System One '{self.name}': the System One request failed ({error}); retry later.")

    def _read_response(self, status: int, text: str) -> dict[str, Any]:
        if status == 200:
            try:
                data = json.loads(text)
            except ValueError:
                data = None
            if not isinstance(data, dict):
                raise _tool_error(f"System One '{self.name}': System One returned a body that is not a JSON object.")
            return data
        detail = self._error_detail(text)
        if status in (401, 403):
            # A credential problem does not go away by asking again.
            raise ValueError(f"System One '{self.name}': System One rejected the API key (HTTP {status}): {detail}")
        if status == 422:
            message = f"System One rejected the request (HTTP 422): {detail}"
        elif status in _TRANSIENT_STATUSES:
            message = f"System One is rate limited or unavailable (HTTP {status}); retry later. {detail}".rstrip()
        else:
            message = f"System One answered HTTP {status}: {detail}"
        logger.error(f"System One '{self.name}': {message}")
        raise _tool_error(f"System One '{self.name}': {message}")

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
        return text.strip()[:_ERROR_CHARS]

    def usage_of(self, data: dict[str, Any]) -> dict[str, Any]:
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

    def usage_summary(self, data: dict[str, Any]) -> dict[str, Any]:
        usage = self.usage_of(data)
        return {
            "input_tokens": usage["prompt_tokens"],
            "output_tokens": usage["completion_tokens"],
            "cost_usd": usage["total_tokens_cost_usd"],
        }
