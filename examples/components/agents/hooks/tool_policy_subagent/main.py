"""Tool policy example (YAML DAG + tracing): per-user rules, a call budget and a sub-agent as a tool.

Expected: a viewer's delete call and, separately, a viewer's Researcher call are blocked (the sub-agent never
starts); an analyst may research but not delete and needs a human's approval for a refund (a scripted human
approves one and declines the other); an admin may do all of it; the third web-search of a run is refused.
Blocks show up as hook events (name, point, decision) in the trace, never as failed tool calls.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from _common import (  # noqa: E402
    is_offline,
    llm_session,
    load_agent,
    parse_args,
    run_scenario,
    send_to_ui,
    step_final,
    step_tool,
)

DELETE_AND_RESEARCH = "Delete customer 42, then research the history of the Apache license."
RESEARCH = "Research the history of the Apache license and summarize it in two sentences."
BUDGET = "Search the web for three different things: Python, Rust and Go."

DELETE_THEN_RESEARCH = [
    step_tool("delete-customer", customer_id="42"),
    step_tool("Researcher", input="History of the Apache license"),
    step_final("Done. See the tool observations for what was allowed."),
]
RESEARCH_ONLY = [
    step_tool("Researcher", input="History of the Apache license"),
    step_final("Done. See the tool observation for what was allowed."),
]
SEARCH_THREE = [
    step_tool("web-search", query="Python"),
    step_tool("web-search", query="Rust"),
    step_tool("web-search", query="Go"),
    step_final("Searched Python and Rust; the budget stopped the third search."),
]
REFUND = "Refund customer 42 by 20 dollars."
REFUND_SCRIPT = [
    step_tool("refund-customer", customer_id="42", amount="20"),
    step_final("Done. See the tool observation for the outcome of the refund."),
]
# (title, question, metadata, parent script, answers of the scripted human to approval questions)
SCENARIOS = [
    (
        "viewer: delete is refused",
        "Delete customer 42.",
        {"role": "viewer"},
        DELETE_THEN_RESEARCH[:1] + [step_final("An admin must do that.")],
        [],
    ),
    ("viewer: the sub-agent call is refused", RESEARCH, {"role": "viewer"}, RESEARCH_ONLY, []),
    ("analyst: may research, may not delete", DELETE_AND_RESEARCH, {"role": "analyst"}, DELETE_THEN_RESEARCH, []),
    ("analyst: a refund needs approval, the human approves", REFUND, {"role": "analyst"}, REFUND_SCRIPT, [""]),
    ("analyst: a refund needs approval, the human declines", REFUND, {"role": "analyst"}, REFUND_SCRIPT, ["too large"]),
    ("admin: may do both, no approval needed", DELETE_AND_RESEARCH, {"role": "admin"}, DELETE_THEN_RESEARCH, []),
    ("admin: a refund needs no approval", REFUND, {"role": "admin"}, REFUND_SCRIPT, []),
    ("admin: the third search is over budget", BUDGET, {"role": "admin"}, SEARCH_THREE, []),
]
CHILD_SCRIPT = [step_final("The Apache License 2.0 was published in 2004 by the ASF.")]


def main():
    args = parse_args(__doc__)
    ui = send_to_ui(args)
    workflow, agent = load_agent(os.path.join(os.path.dirname(__file__), "dag.yaml"))
    print(f"Loaded '{agent.name}' with hooks: {[h.display_name for h in agent.hooks]}")
    offline = is_offline(args)
    if offline:
        print("(offline: scripted LLM)")

    runs = [("Your question", args.question, {"role": "viewer"}, RESEARCH_ONLY, [])] if args.question else SCENARIOS
    for title, question, metadata, parent_script, answers in runs:
        data = {"input": question, "user_id": "demo-user", "metadata": metadata}
        with llm_session(offline, {"research sub-agent": CHILD_SCRIPT, "*": parent_script}, answers):
            run_scenario(workflow, agent, title, data, ui=ui)


if __name__ == "__main__":
    main()
