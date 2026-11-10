from uuid import uuid4

import pytest

from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.callbacks.tracing import PROMPT_MESSAGES_BASE_KEY

SYSTEM = {"role": "system", "content": "You are an agent with a long system prompt."}
USER = {"role": "user", "content": "Question"}
STEP_1 = {"role": "assistant", "content": "Thought 1"}
OBSERVATION_1 = {"role": "user", "content": "Observation 1"}
STEP_2 = {"role": "assistant", "content": "Thought 2"}
OBSERVATION_2 = {"role": "user", "content": "Observation 2"}


def _llm_node(node_id="llm-1"):
    return {"id": node_id, "name": "LLM", "group": "llms", "prompt": {"messages": []}}


def _run_llm(handler, parent_run_id, messages, node_id="llm-1", run_id=None):
    run_id = run_id or uuid4()
    node = _llm_node(node_id)
    handler.on_node_start(node, {}, run_id=run_id, parent_run_id=parent_run_id)
    handler.on_node_execute_run(node, run_id=run_id, prompt_messages=messages)
    return handler.runs[run_id]


def _full_messages(runs, run):
    prompt = run.metadata["node"]["prompt"]
    base = prompt.get(PROMPT_MESSAGES_BASE_KEY)
    if base is None:
        return prompt["messages"]
    return _full_messages(runs, runs[base["run_id"]])[: base["count"]] + prompt["messages"]


@pytest.fixture
def handler():
    return TracingCallbackHandler(compact_llm_prompts_enabled=True)


def test_iterations_store_only_new_messages(handler):
    agent_run_id = uuid4()
    prompts = [
        [SYSTEM, USER],
        [SYSTEM, USER, STEP_1, OBSERVATION_1],
        [SYSTEM, USER, STEP_1, OBSERVATION_1, STEP_2, OBSERVATION_2],
    ]

    runs = [_run_llm(handler, agent_run_id, prompt) for prompt in prompts]

    assert runs[0].metadata["node"]["prompt"] == {"messages": [SYSTEM, USER], PROMPT_MESSAGES_BASE_KEY: None}
    assert runs[1].metadata["node"]["prompt"] == {
        "messages": [STEP_1, OBSERVATION_1],
        PROMPT_MESSAGES_BASE_KEY: {"run_id": runs[0].id, "count": 2},
    }
    assert runs[2].metadata["node"]["prompt"] == {
        "messages": [STEP_2, OBSERVATION_2],
        PROMPT_MESSAGES_BASE_KEY: {"run_id": runs[1].id, "count": 4},
    }
    for run, prompt in zip(runs, prompts):
        assert _full_messages(handler.runs, run) == prompt


def test_rewritten_history_keeps_only_the_shared_start(handler):
    agent_run_id = uuid4()
    summary = {"role": "user", "content": "Summary of earlier steps"}
    first = _run_llm(handler, agent_run_id, [SYSTEM, USER, STEP_1, OBSERVATION_1])

    second = _run_llm(handler, agent_run_id, [SYSTEM, summary])

    assert second.metadata["node"]["prompt"] == {
        "messages": [summary],
        PROMPT_MESSAGES_BASE_KEY: {"run_id": first.id, "count": 1},
    }
    assert _full_messages(handler.runs, second) == [SYSTEM, summary]


def test_prompt_without_shared_start_is_stored_whole(handler):
    agent_run_id = uuid4()
    _run_llm(handler, agent_run_id, [SYSTEM, USER])

    run = _run_llm(handler, agent_run_id, [USER])

    assert run.metadata["node"]["prompt"] == {"messages": [USER], PROMPT_MESSAGES_BASE_KEY: None}


def test_runs_of_other_parents_and_llms_are_not_bases(handler):
    _run_llm(handler, uuid4(), [SYSTEM, USER])
    agent_run_id = uuid4()
    _run_llm(handler, agent_run_id, [SYSTEM, USER], node_id="llm-2")

    run = _run_llm(handler, agent_run_id, [SYSTEM, USER, STEP_1])

    assert run.metadata["node"]["prompt"] == {"messages": [SYSTEM, USER, STEP_1], PROMPT_MESSAGES_BASE_KEY: None}


def test_retried_run_is_not_its_own_base(handler):
    agent_run_id = uuid4()
    first = _run_llm(handler, agent_run_id, [SYSTEM, USER])
    second = _run_llm(handler, agent_run_id, [SYSTEM, USER, STEP_1])

    handler.on_node_execute_run(_llm_node(), run_id=second.id, prompt_messages=[SYSTEM, USER, STEP_1])

    assert second.metadata["node"]["prompt"] == {
        "messages": [STEP_1],
        PROMPT_MESSAGES_BASE_KEY: {"run_id": first.id, "count": 2},
    }


def test_retry_between_concurrent_runs_keeps_what_was_stored(handler):
    agent_run_id = uuid4()
    other_user = {"role": "user", "content": "Other question"}
    first = _run_llm(handler, agent_run_id, [SYSTEM, USER])
    second = _run_llm(handler, agent_run_id, [SYSTEM, other_user])

    handler.on_node_execute_run(_llm_node(), run_id=first.id, prompt_messages=[SYSTEM, USER])

    assert first.metadata["node"]["prompt"] == {"messages": [SYSTEM, USER], PROMPT_MESSAGES_BASE_KEY: None}
    assert _full_messages(handler.runs, second) == [SYSTEM, other_user]


def test_compaction_is_off_by_default():
    handler = TracingCallbackHandler()
    agent_run_id = uuid4()
    _run_llm(handler, agent_run_id, [SYSTEM, USER])

    run = _run_llm(handler, agent_run_id, [SYSTEM, USER, STEP_1])

    assert run.metadata["node"]["prompt"] == {"messages": [SYSTEM, USER, STEP_1]}
