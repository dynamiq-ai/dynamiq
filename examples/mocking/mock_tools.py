"""Mock execution: agents that reason for real while their side effects are suppressed.

Three things this shows:

1. Pinning a sensitive tool so it can never fire, whatever the run asks for.
2. Flipping a whole run into a dry run without editing the workflow.
3. A/B testing the same agent with and without real side effects.

Run with: python examples/mocking/mock_tools.py
"""

import os

from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.node import Node
from dynamiq.nodes.tools.function_tool import function_tool
from dynamiq.runnables import RunnableConfig
from dynamiq.types.mocking import MockConfig, MockPolicy, RunMockConfig

ISSUED_REFUNDS: list[float] = []


@function_tool
def issue_refund(amount: float, **kwargs) -> str:
    """Refunds a customer. Really moves money."""
    ISSUED_REFUNDS.append(amount)
    return f"refunded {amount}"


@function_tool
def lookup_order(order_id: str, **kwargs) -> str:
    """Read-only order lookup. Safe to run for real."""
    return f"order {order_id}: total 42.00, status shipped"


def build_agent(refund_mock: MockConfig | None = None) -> tuple[Agent, Node]:
    """Returns the agent and its read-only lookup tool, so the tool can be excluded by id."""
    llm = OpenAI(connection=OpenAIConnection(api_key=os.environ["OPENAI_API_KEY"]), model="gpt-4o-mini")
    refund = issue_refund()
    if refund_mock:
        refund.mock = refund_mock
    lookup = lookup_order()
    agent = Agent(
        name="support-agent",
        llm=llm,
        role="Resolve customer refund requests.",
        tools=[lookup, refund],
    )
    return agent, lookup


def pinned_sensitive_tool() -> None:
    """`locked` means no run-level policy can turn this tool back on."""
    agent, _lookup = build_agent(
        MockConfig(
            enabled=True,
            locked=True,
            output="Refund of {{ input.amount }} accepted, ref SIM-001.",
        )
    )

    result = agent.run(input_data={"input": "Refund order A-77 in full."})

    print("answer:", result.output["content"])
    print("real refunds issued:", ISSUED_REFUNDS)


def whole_run_dry_run() -> None:
    """No node config at all — the run itself asks for a dry run.

    Only tools are swept in by default, so the agent still reasons with a real LLM.
    `exclude_ids` keeps the harmless read-only lookup live so the agent sees true data.
    """
    agent, lookup = build_agent()

    result = agent.run(
        input_data={"input": "Refund order A-77 in full."},
        config=RunnableConfig(mock=RunMockConfig(policy=MockPolicy.ALL, exclude_ids={lookup.id})),
    )

    print("answer:", result.output["content"])
    print("real refunds issued:", ISSUED_REFUNDS)


def ab_test() -> None:
    """One definition, two runs: control really refunds, variant does not."""
    prompt = {"input": "Refund order A-77 in full."}

    control, _ = build_agent()
    control = control.run(input_data=prompt)
    control_refunds = list(ISSUED_REFUNDS)
    ISSUED_REFUNDS.clear()

    variant_agent, _ = build_agent()
    variant = variant_agent.run(
        input_data=prompt,
        config=RunnableConfig(mock=RunMockConfig(policy=MockPolicy.ALL)),
    )

    print("control:", control.output["content"], "| refunds:", control_refunds)
    print("variant:", variant.output["content"], "| refunds:", ISSUED_REFUNDS)


if __name__ == "__main__":
    for name, scenario in (
        ("pinned sensitive tool", pinned_sensitive_tool),
        ("whole-run dry run", whole_run_dry_run),
        ("A/B comparison", ab_test),
    ):
        print(f"\n=== {name} ===")
        ISSUED_REFUNDS.clear()
        scenario()
