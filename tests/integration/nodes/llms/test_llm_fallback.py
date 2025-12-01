import uuid
from typing import Callable

import pytest
from litellm import ModelResponse
from litellm.exceptions import APIConnectionError, BudgetExceededError, RateLimitError, ServiceUnavailableError, Timeout

from dynamiq import Workflow, connections
from dynamiq.flows import Flow
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.llms.base import FallbackConfig, FallbackTrigger
from dynamiq.nodes.node import InputTransformer, NodeDependency, OutputTransformer
from dynamiq.nodes.tools.python import Python
from dynamiq.prompts import Message, Prompt
from dynamiq.runnables import RunnableConfig, RunnableStatus

# Constants
PRIMARY_MODEL = "gpt-4o"
FALLBACK_MODEL = "gpt-4o-mini"
PRIMARY_LLM_NAME = "PrimaryLLM"
FALLBACK_LLM_NAME = "FallbackLLM"
PRIMARY_RESPONSE = "Primary response"
FALLBACK_RESPONSE = "Fallback response"
LLM_PROVIDER = "openai"


@pytest.fixture
def primary_connection():
    return connections.OpenAI(id=str(uuid.uuid4()), api_key="primary_api_key")


@pytest.fixture
def fallback_connection():
    return connections.OpenAI(id=str(uuid.uuid4()), api_key="fallback_api_key")


@pytest.fixture
def prompt():
    return Prompt(messages=[Message(role="user", content="What is LLM?")])


@pytest.fixture
def create_llm_with_fallback(
    primary_connection: connections.OpenAI,
    fallback_connection: connections.OpenAI,
    prompt: Prompt,
) -> Callable[[FallbackConfig], OpenAI]:
    def _create(fallback_config: FallbackConfig) -> OpenAI:
        fallback_llm = OpenAI(
            name=FALLBACK_LLM_NAME,
            model=FALLBACK_MODEL,
            connection=fallback_connection,
            prompt=prompt,
        )
        return OpenAI(
            name=PRIMARY_LLM_NAME,
            model=PRIMARY_MODEL,
            connection=primary_connection,
            prompt=prompt,
            fallback=FallbackConfig(
                llm=fallback_llm,
                enabled=fallback_config.enabled,
                trigger=fallback_config.trigger,
            ),
        )

    return _create


def mock_llm_response(content: str) -> ModelResponse:
    model_r = ModelResponse()
    model_r["choices"][0]["message"]["content"] = content
    return model_r


@pytest.mark.parametrize(
    ("trigger", "exception", "should_fallback"),
    [
        (FallbackTrigger.ANY, ValueError("Some error"), True),
        (FallbackTrigger.ANY, RateLimitError("Rate limit", LLM_PROVIDER, PRIMARY_MODEL), True),
        (FallbackTrigger.ANY, ConnectionError("Connection failed"), True),
        (FallbackTrigger.RATE_LIMIT, RateLimitError("Rate limit", LLM_PROVIDER, PRIMARY_MODEL), True),
        (FallbackTrigger.RATE_LIMIT, BudgetExceededError("Budget exceeded", LLM_PROVIDER, PRIMARY_MODEL), True),
        (FallbackTrigger.RATE_LIMIT, ValueError("rate limit exceeded"), True),
        (FallbackTrigger.RATE_LIMIT, ValueError("Error 429"), True),
        (FallbackTrigger.RATE_LIMIT, ValueError("quota exceeded"), True),
        (FallbackTrigger.RATE_LIMIT, ConnectionError("Connection failed"), False),
        (FallbackTrigger.CONNECTION, APIConnectionError("API connection error", LLM_PROVIDER, PRIMARY_MODEL), True),
        (FallbackTrigger.CONNECTION, Timeout("Timeout error", LLM_PROVIDER, PRIMARY_MODEL), True),
        (FallbackTrigger.CONNECTION, ServiceUnavailableError("Service unavailable", LLM_PROVIDER, PRIMARY_MODEL), True),
        (FallbackTrigger.CONNECTION, ConnectionError("Connection failed"), True),
        (FallbackTrigger.CONNECTION, TimeoutError("Timeout"), True),
        (FallbackTrigger.CONNECTION, RateLimitError("Rate limit", LLM_PROVIDER, PRIMARY_MODEL), False),
    ],
)
def test_should_trigger_fallback(create_llm_with_fallback, trigger, exception, should_fallback):
    llm = create_llm_with_fallback(FallbackConfig(enabled=True, trigger=trigger))
    assert llm._should_trigger_fallback(exception) == should_fallback


@pytest.mark.parametrize(
    ("enabled", "has_fallback_llm"),
    [
        (False, True),
        (True, False),
        (False, False),
    ],
)
def test_should_trigger_fallback_disabled_or_no_llm(
    primary_connection, fallback_connection, prompt, enabled, has_fallback_llm
):
    fallback_llm = (
        OpenAI(name=FALLBACK_LLM_NAME, model=FALLBACK_MODEL, connection=fallback_connection, prompt=prompt)
        if has_fallback_llm
        else None
    )
    llm = OpenAI(
        name=PRIMARY_LLM_NAME,
        model=PRIMARY_MODEL,
        connection=primary_connection,
        prompt=prompt,
        fallback=FallbackConfig(llm=fallback_llm, enabled=enabled, trigger=FallbackTrigger.ANY),
    )
    assert llm._should_trigger_fallback(ValueError("Some error")) is False


def test_fallback_success_when_primary_fails(mocker, create_llm_with_fallback):
    llm = create_llm_with_fallback(FallbackConfig(enabled=True, trigger=FallbackTrigger.ANY))

    call_count = 0

    def mock_completion(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RateLimitError("Rate limit exceeded", LLM_PROVIDER, PRIMARY_MODEL)
        return mock_llm_response(FALLBACK_RESPONSE)

    mocker.patch("dynamiq.nodes.llms.base.BaseLLM._completion", side_effect=mock_completion)

    wf = Workflow(id=str(uuid.uuid4()), flow=Flow(nodes=[llm]))
    result = wf.run(input_data={}, config=RunnableConfig())

    assert result.status == RunnableStatus.SUCCESS
    assert result.output[llm.id]["output"]["content"] == FALLBACK_RESPONSE
    assert call_count == 2


def test_fallback_not_triggered_when_primary_succeeds(mocker, create_llm_with_fallback):
    llm = create_llm_with_fallback(FallbackConfig(enabled=True, trigger=FallbackTrigger.ANY))

    mocker.patch("dynamiq.nodes.llms.base.BaseLLM._completion", return_value=mock_llm_response(PRIMARY_RESPONSE))

    wf = Workflow(id=str(uuid.uuid4()), flow=Flow(nodes=[llm]))
    result = wf.run(input_data={}, config=RunnableConfig())

    assert result.status == RunnableStatus.SUCCESS
    assert result.output[llm.id]["output"]["content"] == PRIMARY_RESPONSE


def test_both_llms_fail_raises_primary_error(mocker, create_llm_with_fallback):
    llm = create_llm_with_fallback(FallbackConfig(enabled=True, trigger=FallbackTrigger.ANY))

    call_count = 0

    def mock_completion(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RateLimitError("Primary rate limit", LLM_PROVIDER, PRIMARY_MODEL)
        raise RateLimitError("Fallback rate limit", LLM_PROVIDER, FALLBACK_MODEL)

    mocker.patch("dynamiq.nodes.llms.base.BaseLLM._completion", side_effect=mock_completion)

    wf = Workflow(id=str(uuid.uuid4()), flow=Flow(nodes=[llm]))
    result = wf.run(input_data={}, config=RunnableConfig())

    assert result.output[llm.id]["status"] == RunnableStatus.FAILURE.value
    assert call_count == 2


def test_no_fallback_config_raises_immediately(mocker, primary_connection, prompt):
    llm = OpenAI(name=PRIMARY_LLM_NAME, model=PRIMARY_MODEL, connection=primary_connection, prompt=prompt)

    mocker.patch(
        "dynamiq.nodes.llms.base.BaseLLM._completion",
        side_effect=RateLimitError("Rate limit", LLM_PROVIDER, PRIMARY_MODEL),
    )

    wf = Workflow(id=str(uuid.uuid4()), flow=Flow(nodes=[llm]))
    result = wf.run(input_data={}, config=RunnableConfig())

    assert result.output[llm.id]["status"] == RunnableStatus.FAILURE.value


def test_fallback_applies_primary_output_transformer(mocker, primary_connection, fallback_connection, prompt):
    fallback_llm = OpenAI(
        name=FALLBACK_LLM_NAME,
        model=FALLBACK_MODEL,
        connection=fallback_connection,
        prompt=prompt,
    )

    llm = OpenAI(
        name=PRIMARY_LLM_NAME,
        model=PRIMARY_MODEL,
        connection=primary_connection,
        prompt=prompt,
        output_transformer=OutputTransformer(selector={"answer": "$.content"}),
        fallback=FallbackConfig(llm=fallback_llm, enabled=True, trigger=FallbackTrigger.ANY),
    )

    call_count = 0

    def mock_completion(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RateLimitError("Rate limit exceeded", LLM_PROVIDER, PRIMARY_MODEL)
        return mock_llm_response(FALLBACK_RESPONSE)

    mocker.patch("dynamiq.nodes.llms.base.BaseLLM._completion", side_effect=mock_completion)

    wf = Workflow(id=str(uuid.uuid4()), flow=Flow(nodes=[llm]))
    result = wf.run(input_data={}, config=RunnableConfig())

    assert result.status == RunnableStatus.SUCCESS
    assert result.output[llm.id]["output"]["answer"] == FALLBACK_RESPONSE
    assert "content" not in result.output[llm.id]["output"]


def test_fallback_uses_primary_input_transformer(mocker, primary_connection, fallback_connection):
    fallback_llm = OpenAI(
        name=FALLBACK_LLM_NAME,
        model=FALLBACK_MODEL,
        connection=fallback_connection,
        prompt=Prompt(messages=[Message(role="user", content="Transformed question: {{transformed_question}}")]),
    )

    llm = OpenAI(
        name=PRIMARY_LLM_NAME,
        model=PRIMARY_MODEL,
        connection=primary_connection,
        prompt=Prompt(messages=[Message(role="user", content="Transformed question: {{transformed_question}}")]),
        input_transformer=InputTransformer(selector={"transformed_question": "$.question"}),
        fallback=FallbackConfig(llm=fallback_llm, enabled=True, trigger=FallbackTrigger.ANY),
    )

    call_count = 0
    captured_inputs = []

    def mock_completion(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        messages = kwargs.get("messages", [])
        captured_inputs.append(messages)
        if call_count == 1:
            raise RateLimitError("Rate limit exceeded", LLM_PROVIDER, PRIMARY_MODEL)
        return mock_llm_response(FALLBACK_RESPONSE)

    mocker.patch("dynamiq.nodes.llms.base.BaseLLM._completion", side_effect=mock_completion)

    wf = Workflow(id=str(uuid.uuid4()), flow=Flow(nodes=[llm]))
    question = "What is AI?"
    result = wf.run(input_data={"question": question}, config=RunnableConfig())

    assert result.status == RunnableStatus.SUCCESS
    assert len(captured_inputs) == 2
    fallback_messages = captured_inputs[1]
    assert any(question in str(msg) for msg in fallback_messages)


def test_fallback_not_triggered_on_skip(mocker, primary_connection, fallback_connection, prompt):
    """Test that fallback is NOT triggered when primary LLM is skipped (e.g., dependency failure)."""
    failing_node = Python(
        name="FailingNode",
        code="def run(input_data): raise ValueError('Dependency failed')",
    )

    fallback_llm = OpenAI(
        name=FALLBACK_LLM_NAME,
        model=FALLBACK_MODEL,
        connection=fallback_connection,
        prompt=prompt,
    )

    llm = OpenAI(
        name=PRIMARY_LLM_NAME,
        model=PRIMARY_MODEL,
        connection=primary_connection,
        prompt=prompt,
        depends=[NodeDependency(node=failing_node)],
        fallback=FallbackConfig(llm=fallback_llm, enabled=True, trigger=FallbackTrigger.ANY),
    )

    mock_completion = mocker.patch("dynamiq.nodes.llms.base.BaseLLM._completion")

    wf = Workflow(id=str(uuid.uuid4()), flow=Flow(nodes=[failing_node, llm]))
    result = wf.run(input_data={}, config=RunnableConfig())

    assert result.output[llm.id]["status"] == RunnableStatus.SKIP.value
    mock_completion.assert_not_called()
