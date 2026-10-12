"""PII mask / restore example (YAML DAG + tracing).

Expected: the model never sees the user's email or phone (only <EMAIL_1> / <PHONE_1>); `send-email` receives the
real address; the account manager's email from a tool result stays masked; the final answer shows the user's own
address again. The trace holds hook events (name, point, decision, sizes) and never a value.
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

QUESTION = (
    "Look up customer 42, then email ann.lee@example.com a short summary and tell her we will call "
    "+1 415 555 0100. Also cc her account manager."
)

OFFLINE_SCRIPT = {
    "*": [
        step_tool("lookup-customer", customer_id="42"),
        step_tool("send-email", to="<EMAIL_1>", subject="Your account", body="Hi Ann, we will call <PHONE_1>."),
        step_final("Done: I emailed <EMAIL_1>, will call <PHONE_1>, and cc'd the manager (<EMAIL_2>)."),
    ]
}


def main():
    args = parse_args(__doc__)
    ui = send_to_ui(args)
    workflow, agent = load_agent(os.path.join(os.path.dirname(__file__), "dag.yaml"))
    print(f"Loaded '{agent.name}' with hooks: {[h.display_name for h in agent.hooks]}")
    question = args.question or QUESTION

    offline = is_offline(args)
    if offline:
        print("(offline: scripted LLM)")
    with llm_session(offline, OFFLINE_SCRIPT) as seen:
        run_scenario(workflow, agent, "PII round trip", {"input": question}, ui=ui)
    if offline:
        report_leaks(seen, ["ann.lee@example.com", "+1 415 555 0100", "boss@example.com"])


if __name__ == "__main__":
    main()
