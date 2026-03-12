from dynamiq.callbacks import BaseCallbackHandler
from dynamiq.connections import Anthropic as AnthropicConnection
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms import Anthropic
from dynamiq.runnables import RunnableConfig


class UsageCaptureCallback(BaseCallbackHandler):
    """Capture usage payloads emitted during node execution."""

    def __init__(self):
        self.usage_payloads: list[dict] = []

    def on_node_execute_run(self, serialized: dict, **kwargs):
        if usage_data := kwargs.get("usage_data"):
            self.usage_payloads.append(usage_data)


def build_long_agent_role(repetitions: int = 80) -> str:
    base_block = (
        "You are a concise technical assistant. "
        "Prefer factual, structured answers, highlight assumptions, and avoid speculation.\n"
    )
    return "System directives:\n" + "".join(base_block for _ in range(repetitions))


def summarize_cache_usage(usage_payloads: list[dict]) -> dict:
    cache_creation_input_tokens = sum((item.get("cache_creation_input_tokens") or 0) for item in usage_payloads)
    cache_read_input_tokens = sum((item.get("cache_read_input_tokens") or 0) for item in usage_payloads)
    prompt_tokens = sum((item.get("prompt_tokens") or 0) for item in usage_payloads)
    return {
        "prompt_tokens": prompt_tokens,
        "cache_creation_input_tokens": cache_creation_input_tokens,
        "cache_read_input_tokens": cache_read_input_tokens,
    }


def run_agent_with_anthropic_prompt_caching(user_input: str):
    llm = Anthropic(
        model="claude-sonnet-4-6",
        connection=AnthropicConnection(),
    )
    agent = Agent(name="Agent", llm=llm, role=build_long_agent_role())

    usage_capture = UsageCaptureCallback()
    config = RunnableConfig(callbacks=[usage_capture])

    first_response = agent.run(input_data={"input": user_input}, config=config)
    first_usage = summarize_cache_usage(usage_capture.usage_payloads)

    usage_capture.usage_payloads.clear()
    second_response = agent.run(input_data={"input": user_input}, config=config)
    second_usage = summarize_cache_usage(usage_capture.usage_payloads)

    print("First response:", first_response.output.get("content", "")[:200])
    print("Second response:", second_response.output.get("content", "")[:200])
    print("First run usage:", first_usage)
    print("Second run usage:", second_usage)


if __name__ == "__main__":
    run_agent_with_anthropic_prompt_caching(user_input="Explain prompt caching in one short paragraph.")
