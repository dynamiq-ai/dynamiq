"""Hook options example (YAML DAG + tracing): `inherit`, PII `scope: request`, `live_stream`, `from_context`.

Expected: the model never sees the user's email (it gets <EMAIL_1>) although the history keeps the raw text; the
CRM tool is called with the tenant from the caller's metadata whatever the model sent; a value the Researcher
sub-agent finds is masked too (<EMAIL_2>, the numbering continues because the run state is shared); the crm-lookup
budget of 2 counts the parent's and the sub-agent's calls together.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from _common import (  # noqa: E402
    is_offline,
    llm_session,
    load_agent,
    parse_args,
    report_leaks,
    run_scenario,
    send_to_ui,
    step_final,
    step_tool,
)

SCENARIOS = [
    (
        "scope request + from_context: tenant comes from the caller",
        "Find ann@leak.example in the CRM and email her that we looked.",
        {"tenant": "acme"},
        {
            "*": [
                step_tool("crm-lookup", query="<EMAIL_1>", tenant="spoofed-tenant"),
                step_tool("send-email", to="<EMAIL_1>", subject="Looked", body="We looked you up."),
                step_final("Done: I looked up <EMAIL_1> and emailed her."),
            ]
        },
    ),
    (
        "inherit: the sub-agent shares the masking and the budget",
        "Ask the Researcher who handles ann@leak.example, then look up the customer twice.",
        {"tenant": "acme"},
        {
            "research sub-agent": [
                step_tool("crm-lookup", query="ann", tenant="x"),
                step_final("The account owner is zq9boss@leak.example."),
            ],
            "*": [
                step_tool("Researcher", input="Who handles <EMAIL_1>?"),
                step_tool("crm-lookup", query="owner", tenant="x"),
                step_tool("crm-lookup", query="owner again", tenant="x"),
                step_final("Researched; the lookups after the shared budget of 2 were refused."),
            ],
        },
    ),
]


def main():
    args = parse_args(__doc__)
    ui = send_to_ui(args)
    workflow, agent = load_agent(os.path.join(os.path.dirname(__file__), "dag.yaml"))
    print(f"Loaded '{agent.name}' with hooks: {[h.display_name for h in agent.hooks]}")
    offline = is_offline(args)
    if offline:
        print("(offline: scripted LLM)")
    scenarios = [("Your question", args.question, {"tenant": "acme"}, {})] if args.question else SCENARIOS

    seen_all: list[str] = []
    for title, question, metadata, script in scenarios:
        data = {"input": question, "user_id": "demo-user", "metadata": metadata}
        with llm_session(offline, script) as seen:
            run_scenario(workflow, agent, title, data, ui=ui)
        seen_all += seen
    if offline:
        report_leaks(seen_all, ["ann@leak.example", "zq9boss@leak.example"])


if __name__ == "__main__":
    main()
