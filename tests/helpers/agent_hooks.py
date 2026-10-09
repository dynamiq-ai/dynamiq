"""Scripted agents for hook tests: the same script runs in every inference mode."""

import json
import threading
from contextlib import contextmanager
from typing import Any, ClassVar, Literal
from unittest.mock import MagicMock, patch

from pydantic import BaseModel, Field

from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.node import Node
from dynamiq.nodes.types import InferenceMode
from dynamiq.prompts import Message
from dynamiq.runnables import RunnableConfig, RunnableResult, RunnableStatus

ALL_MODES = [
    InferenceMode.STRUCTURED_OUTPUT,
    InferenceMode.FUNCTION_CALLING,
    InferenceMode.DEFAULT,
    InferenceMode.XML,
]


class SearchSchema(BaseModel):
    query: str = Field(default="", description="Search query.")
    mode: str = Field(default="slow", description="Search mode.")


class SearchTool(Node):
    """Records every input it is run with; returns ``result`` (or a canned string)."""

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "search"
    description: str = "Searches the web."
    result: str | None = None
    input_schema: ClassVar[type[SearchSchema]] = SearchSchema

    def execute(self, input_data: SearchSchema, config: RunnableConfig | None = None, **kwargs) -> dict[str, Any]:
        TOOL_CALLS.append((self.name, input_data.model_dump()))
        return {"content": self.result if self.result is not None else f"results for {input_data.query}"}


class EmailSchema(BaseModel):
    to: str = Field(default="", description="Recipient address.")
    body: str = Field(default="", description="Body.")


class EmailTool(Node):
    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "send-email"
    description: str = "Sends an email."
    input_schema: ClassVar[type[EmailSchema]] = EmailSchema

    def execute(self, input_data: EmailSchema, config: RunnableConfig | None = None, **kwargs) -> dict[str, Any]:
        TOOL_CALLS.append((self.name, input_data.model_dump()))
        return {"content": f"sent to {input_data.to}"}


class ExplodingTool(Node):
    """Fails like an HTTP tool: the failure text carries the response body."""

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "http"
    description: str = "Calls an API."
    input_schema: ClassVar[type[SearchSchema]] = SearchSchema
    body: str = "404 Not Found"

    def execute(self, input_data: SearchSchema, config: RunnableConfig | None = None, **kwargs) -> dict[str, Any]:
        TOOL_CALLS.append((self.name, input_data.model_dump()))
        raise ToolExecutionException(self.body, recoverable=True)


TOOL_CALLS: list[tuple[str, dict]] = []
_LOCK = threading.Lock()


def make_llm() -> OpenAI:
    return OpenAI(connection=OpenAIConnection(api_key="test-api-key"), model="gpt-4o", max_tokens=100, temperature=0)


def tool_step(name: str, **args) -> tuple:
    return ("tool", name, args)


def final_step(text: str) -> tuple:
    return ("final", text)


def parallel_step(*calls: tuple[str, dict]) -> tuple:
    """Several tool calls in one reply (native parallel tool calls: FUNCTION_CALLING only)."""
    return ("parallel", list(calls))


def encode(mode: InferenceMode, step: tuple, index: int) -> dict[str, Any]:
    """The raw LLM output that makes the agent take ``step`` in ``mode``."""
    kind = step[0]
    if mode == InferenceMode.FUNCTION_CALLING:
        if kind == "final":
            call = {"name": "provide_final_answer", "arguments": json.dumps({"thought": "done", "answer": step[1]})}
            return {"content": "", "tool_calls": [{"id": f"call_{index}", "type": "function", "function": call}]}
        items = [(step[1], step[2])] if kind == "tool" else step[1]
        return {
            "content": "",
            "tool_calls": [
                {
                    "id": f"call_{index}_{n}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps({"thought": "t", **args})},
                }
                for n, (name, args) in enumerate(items)
            ],
        }
    if kind == "final":
        text = step[1]
        if mode == InferenceMode.STRUCTURED_OUTPUT:
            content = json.dumps({"thought": "done", "action": "finish", "action_input": text})
        elif mode == InferenceMode.DEFAULT:
            content = f"Thought: done\nAnswer: {text}"
        else:
            content = f"<output><thought>done</thought><answer>{text}</answer></output>"
    else:
        _, name, args = step
        if mode == InferenceMode.STRUCTURED_OUTPUT:
            content = json.dumps({"thought": "t", "action": name, "action_input": args})
        elif mode == InferenceMode.DEFAULT:
            content = f"Thought: t\nAction: {name}\nAction Input: {json.dumps(args)}"
        else:
            content = (
                f"<output><thought>t</thought><action>{name}</action>"
                f"<action_input>{json.dumps(args)}</action_input></output>"
            )
    return {"content": content}


class Recorder:
    """What the (scripted) LLM was sent, call by call."""

    def __init__(self):
        self.prompts: list[list[Message]] = []

    @property
    def calls(self) -> int:
        return len(self.prompts)

    def sent_text(self, call: int | None = None) -> str:
        prompts = self.prompts if call is None else [self.prompts[call]]
        return "\n".join(str(getattr(m, "content", "")) for prompt in prompts for m in prompt)


@contextmanager
def scripted(agent: Agent, mode: InferenceMode, steps: list[tuple]):
    """Make ``agent``'s LLM follow ``steps`` in ``mode``. Yields a Recorder of the prompts it was sent."""
    agent.inference_mode = mode
    recorder = Recorder()
    counter = {"n": 0}

    def run(**kwargs):
        with _LOCK:
            index = counter["n"]
            counter["n"] += 1
            recorder.prompts.append(list(kwargs["prompt"].messages))
        step = steps[min(index, len(steps) - 1)]
        result = MagicMock(spec=RunnableResult)
        result.status = RunnableStatus.SUCCESS
        result.output = encode(mode, step, index)
        return result

    with patch.object(agent.llm, "run", side_effect=run):
        yield recorder


def build_agent(hooks: list | None = None, tools: list | None = None, **kwargs) -> Agent:
    return Agent(
        id=kwargs.pop("id", "agent"),
        name=kwargs.pop("name", "assistant"),
        llm=make_llm(),
        role="help the user",
        tools=tools if tools is not None else [SearchTool(), EmailTool()],
        hooks=hooks or [],
        max_loops=kwargs.pop("max_loops", 6),
        **kwargs,
    )


def run_scripted(agent: Agent, mode: InferenceMode, steps: list[tuple], question: str = "hello", **run_kwargs):
    """Run once. Returns (result, recorder)."""
    if (trusted := run_kwargs.pop("trusted", None)) is not None:
        run_kwargs["config"] = RunnableConfig(callbacks=[], trusted_context=trusted)
    with scripted(agent, mode, steps) as recorder:
        result = agent.run(input_data={"input": question, **run_kwargs.pop("input_extra", {})}, **run_kwargs)
    return result, recorder


def observations_in(messages) -> int:
    """How many tool results the conversation already holds (decides which scripted step comes next)."""
    count = 0
    for message in messages:
        content = str(getattr(message, "content", ""))
        role = getattr(getattr(message, "role", None), "value", getattr(message, "role", None))
        if role == "tool" or content.startswith("Observation"):
            count += 1
    return count


@contextmanager
def scripted_stateless(mode: InferenceMode, steps: list[tuple], agents: list[Agent] | None = None):
    """Like ``scripted`` but every ``OpenAI.run`` (any agent instance, e.g. Map clones) picks its step from the
    number of tool results already in its own prompt, so concurrent runs cannot disturb each other."""
    prompts: list[list[Message]] = []
    for agent in agents or []:
        agent.inference_mode = mode

    def run(self, **kwargs):
        messages = list(kwargs["prompt"].messages)
        with _LOCK:
            prompts.append(messages)
        index = observations_in(messages)
        result = MagicMock(spec=RunnableResult)
        result.status = RunnableStatus.SUCCESS
        result.output = encode(mode, steps[min(index, len(steps) - 1)], index)
        return result

    with patch.object(OpenAI, "run", run):
        yield prompts


@contextmanager
def scripted_streaming(agent: Agent, mode: InferenceMode, steps: list[tuple]):
    """Like ``scripted`` but the final reply is also fed, char by char, to the callbacks the agent attached to
    its LLM (as a real streaming LLM would). Yields ``(recorder, streamed)``, ``streamed`` being the
    ``(step, content)`` pairs the agent handed to ``stream_content``."""
    agent.inference_mode = mode
    recorder = Recorder()
    streamed: list[tuple[str, Any]] = []
    counter = {"n": 0}

    def run(**kwargs):
        index = counter["n"]
        counter["n"] += 1
        recorder.prompts.append(list(kwargs["prompt"].messages))
        output = encode(mode, steps[min(index, len(steps) - 1)], index)
        serialized = {"group": "llms", "id": agent.llm.id}
        for callback in kwargs["config"].callbacks:
            if not hasattr(callback, "accumulated_content"):
                continue
            if mode == InferenceMode.FUNCTION_CALLING:
                call = output["tool_calls"][0]
                delta = {
                    "tool_calls": [{"index": 0, "type": "function", "function": {"name": call["function"]["name"]}}]
                }
                callback.on_node_execute_stream(serialized, chunk={"choices": [{"delta": delta}]})
                text = call["function"]["arguments"]
                for char in text:
                    delta = {"tool_calls": [{"index": 0, "type": "function", "function": {"arguments": char}}]}
                    callback.on_node_execute_stream(serialized, chunk={"choices": [{"delta": delta}]})
            else:
                for char in output["content"]:
                    callback.on_node_execute_stream(serialized, chunk={"choices": [{"delta": {"content": char}}]})
            callback.on_node_execute_end(serialized, {})
        result = MagicMock(spec=RunnableResult)
        result.status = RunnableStatus.SUCCESS
        result.output = output
        return result

    def record(self, content, source, step, config=None, **kwargs):
        if self is agent:
            streamed.append((step, content))
        return content

    with patch.object(agent.llm, "run", side_effect=run), patch.object(Agent, "stream_content", record):
        yield recorder, streamed
