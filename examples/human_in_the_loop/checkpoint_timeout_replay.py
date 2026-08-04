"""HITL input timeout -> checkpoint -> resume, with the agent's prompt printed on both sides.

Scenario (two human-feedback requests, one interruption):

    run 1, loop 1  agent asks Q1  -> an answer is already queued -> answered normally
    run 1, loop 2  agent asks Q2  -> nobody answers -> input-streaming timeout
                                  -> checkpoint saved, flow run FAILS
    -- pause: the process could exit here; the checkpoint is the whole state --
    run 2, loop 2  the pending tool call is REPLAYED from the checkpoint (no LLM call)
                   -> the queued answer for Q2 is consumed -> agent finishes

What to look for in the output: the conversation printed after the resume is the
same conversation that would exist had nobody ever walked away - the Q2 answer
comes back as a `role: tool` message carrying the SAME tool_call_id the assistant
message emitted before the pause. Losing that id is what previously turned the
resumed history into an orphaned tool_call plus a stray `role: user` observation.

The LLM is scripted, so this runs offline with no API key and no network. Every
other moving part (tool, agent, flow, checkpoint backend, resume) is real.
"""

from queue import Queue
from unittest.mock import patch

from litellm.types.utils import ModelResponse

from dynamiq import flows
from dynamiq.checkpoints.backends.in_memory import InMemory
from dynamiq.checkpoints.config import CheckpointConfig
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.tools.human_feedback import (
    HFStreamingInputEventMessage,
    HFStreamingInputEventMessageData,
    HumanFeedbackAction,
    HumanFeedbackTool,
)
from dynamiq.nodes.types import InferenceMode
from dynamiq.types.feedback import FeedbackMethod
from dynamiq.types.streaming import StreamingConfig

FLOW_ID = "hitl-checkpoint-flow"
AGENT_ID = "hitl-checkpoint-agent"
TOOL_ID = "human-input-tool"
TOOL_NAME = "human-input"

# Short enough to keep the demo snappy; in production this is how long a user has to reply.
INPUT_TIMEOUT_SECONDS = 2.0

QUESTION_1 = "Do you want fiction or non-fiction?"
QUESTION_2 = "Should I limit it to books published after 2020?"
ANSWER_1 = "Fiction, please."
ANSWER_2 = "Yes, after 2020."


def tool_call_response(call_id: str, question: str) -> ModelResponse:
    """A scripted assistant turn that calls the human-feedback tool."""
    return ModelResponse(
        choices=[
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": TOOL_NAME,
                                "arguments": (
                                    '{"thought": "I need the user to clarify before answering.", '
                                    f'"action": "ask", "input": "{question}"}}'
                                ),
                            },
                        }
                    ],
                }
            }
        ]
    )


def final_answer_response(answer: str) -> ModelResponse:
    """A scripted assistant turn that finishes the run."""
    return ModelResponse(
        choices=[
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_final",
                            "type": "function",
                            "function": {
                                "name": "provide_final_answer",
                                "arguments": f'{{"thought": "I have everything I need.", "answer": "{answer}"}}',
                            },
                        }
                    ],
                }
            }
        ]
    )


class ScriptedLLM:
    """Replays a fixed list of assistant turns, one per LLM call.

    The tool call for Q2 is scripted exactly once. If the resumed run had to ask
    the LLM again instead of replaying the checkpointed call, the script would
    hand it the final answer and the second question would never be asked - so a
    correct replay is observable in the output rather than merely asserted.
    """

    def __init__(self, responses: list[ModelResponse]):
        self.responses = responses
        self.calls = 0

    def __call__(self, *args, **kwargs) -> ModelResponse:
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return response


def queue_answer(queue: Queue, answer: str) -> None:
    """Put a human reply on the tool's input queue, as a chat UI would."""
    queue.put(
        HFStreamingInputEventMessage(
            entity_id=TOOL_ID,
            data=HFStreamingInputEventMessageData(content=answer),
        ).model_dump_json()
    )


def print_history(messages, title: str) -> None:
    """Print the agent's conversation the way the provider will receive it."""
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
    for i, msg in enumerate(messages):
        role = msg.role.value if hasattr(msg.role, "value") else msg.role
        content = (msg.content or "").replace("\n", " ").strip()
        line = f"[{i}] {role:<9} | {content[:60]}"
        if msg.tool_calls:
            ids = ", ".join(tc["id"] for tc in msg.tool_calls)
            names = ", ".join(tc["function"]["name"] for tc in msg.tool_calls)
            line += f"\n            tool_calls -> {names} (id: {ids})"
        if msg.tool_call_id:
            line += f"\n            answers tool_call id: {msg.tool_call_id}"
        print(line)


def build_flow(queue: Queue, backend: InMemory) -> tuple[flows.Flow, Agent]:
    tool = HumanFeedbackTool(
        id=TOOL_ID,
        name=TOOL_NAME,
        action=HumanFeedbackAction.ASK,
        input_method=FeedbackMethod.STREAM,
        output_method=FeedbackMethod.STREAM,
        streaming=StreamingConfig(enabled=True, input_queue=queue, timeout=INPUT_TIMEOUT_SECONDS),
    )
    agent = Agent(
        id=AGENT_ID,
        name="Book Recommender",
        llm=OpenAI(connection=OpenAIConnection(api_key="not-used-scripted-llm"), model="gpt-4o"),
        tools=[tool],
        role="Ask the user clarifying questions before recommending a book.",
        inference_mode=InferenceMode.FUNCTION_CALLING,
        max_loops=4,
    )
    checkpoint = CheckpointConfig(
        enabled=True,
        backend=backend,
        checkpoint_after_node_enabled=False,
        checkpoint_on_failure_enabled=False,
        checkpoint_on_cancel_enabled=False,
        checkpoint_mid_agent_loop_enabled=False,
        # The trigger this demo is about: persist state when a human doesn't reply in time.
        checkpoint_on_input_timeout_enabled=True,
    )
    return flows.Flow(id=FLOW_ID, nodes=[agent], checkpoint=checkpoint), agent


def main() -> None:
    queue = Queue()
    backend = InMemory()
    flow, agent = build_flow(queue, backend)

    llm = ScriptedLLM(
        [
            tool_call_response("call_q1", QUESTION_1),
            tool_call_response("call_q2", QUESTION_2),
            final_answer_response("Try 'The Ministry for the Future' (2020)."),
        ]
    )

    with patch("dynamiq.nodes.llms.base.BaseLLM._completion", side_effect=llm):
        # ---- run 1: Q1 is answered, Q2 is not -------------------------------
        print(f"Queuing an answer for Q1 only. Q2 will wait {INPUT_TIMEOUT_SECONDS}s and time out.")
        queue_answer(queue, ANSWER_1)

        first = flow.run_sync(input_data={"input": "Recommend me a book."})
        print(f"\nrun 1 status: {first.status.value}  (LLM calls so far: {llm.calls})")

        print_history(agent._prompt.messages, "BEFORE THE PAUSE - live agent history at the moment of timeout")

        # ---- the pause: everything needed to continue is in the checkpoint --
        saved = backend.get_latest_by_flow(FLOW_ID)
        agent_iteration = saved.node_states[AGENT_ID].internal_state["iteration"]
        iteration = agent_iteration["iteration_data"]
        print(f"\n{'=' * 78}\nCHECKPOINT {saved.id}\n{'=' * 78}")
        print(f"  completed loops        : {agent_iteration['completed_iterations']}")
        print(f"  pending action         : {iteration['pending_action']}")
        print(f"  pending action input   : {iteration['pending_action_input']}")
        print(f"  serialized messages    : {len(iteration['prompt_messages'])}")
        # The tool_call id is not stored as its own field - it is already in the last
        # assistant message, and that is where the resumed run reads it back from.
        unanswered = [tc["id"] for m in iteration["prompt_messages"] if m.get("tool_calls") for tc in m["tool_calls"]]
        print(f"  tool_call id in history: {unanswered[-1]}")

        # ---- run 2: the user finally answers, the tool call is replayed -----
        print(f"\nThe user comes back and answers Q2. Resuming from {saved.id} ...")
        queue_answer(queue, ANSWER_2)

        second = flow.run_sync(input_data={"input": "Recommend me a book."}, resume_from=saved.id)
        print(f"\nrun 2 status: {second.status.value}  (LLM calls total: {llm.calls})")

        print_history(agent._prompt.messages, "AFTER THE PAUSE - history rebuilt from the checkpoint")

        print(f"\nfinal answer: {second.output[AGENT_ID]['output']['content']}")

    # ---- the point of the demo -------------------------------------------
    messages = agent._prompt.messages
    q2_call = next(m for m in messages if m.tool_calls and m.tool_calls[0]["id"] == "call_q2")
    q2_reply = next((m for m in messages if m.tool_call_id == "call_q2"), None)
    print(f"\n{'=' * 78}\nCONSISTENCY CHECK\n{'=' * 78}")
    print(f"  Q2 asked in     : {q2_call.role.value} message with tool_call id 'call_q2'")
    if q2_reply is None:
        print("  Q2 answered in  : NOTHING - the tool_call is orphaned (history diverged from an " "uninterrupted run)")
    else:
        print(f"  Q2 answered in  : {q2_reply.role.value} message, tool_call_id={q2_reply.tool_call_id}")
        print(f"  answer content  : {q2_reply.content!r}")
        print("  -> identical in shape to a run that was never interrupted")


if __name__ == "__main__":
    main()
