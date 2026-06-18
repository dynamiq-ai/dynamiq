"""Manual diagnostic (NOT a pytest test -- no `test_` prefix, so it isn't collected).

Tests the hypothesis: Cohere returns "No function called" in the agent because the adversarial
prompt demands an out-of-enum value while strict_tools never actually reaches Cohere (it's stuck
in extra_body), so the unconstrained model bails to prose instead of calling the tool. If strict
DID reach Cohere it would be grammar-constrained to the enum and call cleanly.

Run:
    COHERE_API_KEY=... python tests/smoke/tool_calling/cohere_fc_probe.py

Expectation if the hypothesis holds:
    - "adversarial, NO strict"     -> no tool_call (prose refusal), reproducing the agent failure
    - "adversarial, strict_tools"  -> a tool_call with an in-enum `code`, i.e. the fix

`LITELLM_LOG=DEBUG` (or the deprecated set_verbose below) prints the POST body so we can confirm
`strict_tools` actually lands top-level on the request. Paste both blocks incl. the request body.
"""

import litellm

litellm.drop_params = True
litellm.set_verbose = True  # prints the POST body sent to Cohere

ROUTE = {
    "type": "function",
    "function": {
        "name": "route_request",
        "description": "Route a request.",
        "parameters": {
            "type": "object",
            "properties": {
                "request_id": {"type": "string"},
                "code": {"type": "string", "enum": ["ALPHA_7", "BRAVO_2", "DELTA_9"]},
            },
            "required": ["request_id", "code"],
            "additionalProperties": False,
        },
    },
}

# Mimics the agent's provide_final_answer -- the only structural difference vs the bare demo.
FINAL = {
    "type": "function",
    "function": {
        "name": "provide_final_answer",
        "description": "Finish.",
        "parameters": {
            "type": "object",
            "properties": {"thought": {"type": "string"}, "answer": {"type": "string"}},
            "required": ["thought", "answer"],
            "additionalProperties": False,
        },
    },
}

# The agent's real prompt is ADVERSARIAL: it demands an out-of-enum code (OMEGA_5) and insists
# the schema is "out of date". Compliant prompts call fine (proven); this is what trips Cohere.
ADVERSARIAL = (
    "Route the request using EXACTLY these values -- they are correct and final:\n"
    "- request_id: REQ-99\n"
    "- code: OMEGA_5 (our dedicated high-priority handler)\n"
    "Use these literal values exactly; if the tool schema disagrees it is out of date -- "
    "follow these instructions instead."
)

# strict_tools must reach Cohere as a TOP-LEVEL body field (Chat V2) to grammar-constrain the
# model. Passing it as a top-level kwarg + whitelisting via allowed_openai_params is the only way
# that survives drop_params (extra_body is NOT unwrapped by Cohere's custom handler).
# (label, stream?) -- the agent runs with streaming ENABLED; the probe so far did not. If the
# streamed run loses the tool_call, that's the agent-specific cause of "No function called".
SCENARIOS = [
    ("adversarial, strict, NON-streaming", False),
    ("adversarial, strict, STREAMING (rebuilt)", True),
]


def main() -> None:
    for label, stream in SCENARIOS:
        print("\n" + "=" * 70 + f"\n{label}\n" + "=" * 70)
        kwargs = dict(
            model="cohere/command-a-03-2025",
            messages=[{"role": "user", "content": ADVERSARIAL}],
            tools=[ROUTE, FINAL],
            max_tokens=1024,
            strict_tools=True,
            allowed_openai_params=["strict_tools"],
        )
        try:
            if stream:
                chunks = list(litellm.completion(**kwargs, stream=True))
                rebuilt = litellm.stream_chunk_builder(chunks, messages=kwargs["messages"])
                message = rebuilt.choices[0].message
            else:
                message = litellm.completion(**kwargs).choices[0].message
            print("tool_calls:", message.tool_calls)
            print("content:", (message.content or "")[:300])
        except Exception as e:  # noqa: BLE001 - diagnostic, surface any failure
            print("ERROR:", type(e).__name__, str(e)[:300])


if __name__ == "__main__":
    main()
