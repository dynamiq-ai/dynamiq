"""End-to-end smoke test for the Stagehand tool on the v3 SDK.

Requires outbound access to api.browserbase.com / api.stagehand.browserbase.com (and api.steel.dev
for the Steel path). Keys come from environment variables only:

    BROWSERBASE_API_KEY, BROWSERBASE_PROJECT_ID, ANTHROPIC_API_KEY, OPENAI_API_KEY, STEEL_API_KEY

Run all paths, or a subset:

    python e2e_smoke.py                 # all
    python e2e_smoke.py bb-provider     # Browserbase + Anthropic provider key
    python e2e_smoke.py bb-openai       # Browserbase + OpenAI provider key
    python e2e_smoke.py bb-gateway      # Browserbase Model Gateway (no model key)
    python e2e_smoke.py steel           # Steel cloud via the bundled local Stagehand server

Steel path note: leave ANTHROPIC_BASE_URL unset (the bundled Stagehand server then defaults to
https://api.anthropic.com/v1); if your environment sets it, the value must include the /v1 suffix
or anthropic/* model calls 404.

Model support is NOT the same on the two Browserbase paths. With your own provider key any model
that provider serves works; the Model Gateway serves a much narrower set. Measured: gpt-4.1 and
claude-sonnet-4-6 work on the gateway, while gpt-4o is rejected (400 unsupported) and gpt-5 fails
(422, no deployment). Keep the gateway target on a model known to be served there.
"""

import os
import sys
import traceback

from dynamiq.connections import Browserbase as BrowserbaseConnection
from dynamiq.connections import SteelBrowser
from dynamiq.nodes.tools.stagehand import Stagehand, StagehandInputSchema

# Deliberately NOT claude-sonnet-5. Its observe fails on the HOSTED Browserbase v3 server, not just
# the bundled local one: 0/7 on api.stagehand.browserbase.com, always a bare 502. act, extract and
# go_back are fine — observe alone is broken. The Model Gateway refuses sonnet-5 outright (400,
# unsupported model), so no key arrangement makes it usable there. Revisit once Browserbase ships a
# fix; `fallback_model_name` also covers it now, at the cost of a failed round-trip per observe.
MODEL = "anthropic/claude-sonnet-4-6"
# The shipped examples drive Stagehand with gpt-4o, so keep that exact model covered.
OPENAI_MODEL = "openai/gpt-4o"
# gpt-4o is not served by the gateway (see the module docstring), so that target needs its own.
GATEWAY_MODEL = "openai/gpt-4.1"
RESULTS = {}

ACTIONS = [
    ("goto", {"action_type": "goto", "url": "https://example.com", "brief": "open example.com"}),
    (
        "extract",
        {"action_type": "extract", "instruction": "Extract the page heading and first paragraph", "brief": "extract"},
    ),
    ("observe", {"action_type": "observe", "instruction": "Find all links on the page", "brief": "observe links"}),
    ("act", {"action_type": "act", "instruction": "Click the 'More information...' link", "brief": "click link"}),
    ("go_back", {"action_type": "go_back", "instruction": "go back", "brief": "go back"}),
]


def run(tool, label, actions):
    try:
        for name, payload in actions:
            out = tool.execute(StagehandInputSchema(**payload))
            extras = [k for k in ("screenshot", "live_view_url", "files") if k in out]
            print(f"  [{label}] {name}: OK {extras} -> {str(out.get('content'))[:120]}")
        RESULTS[label] = "PASS"
    except Exception as exc:
        RESULTS[label] = f"FAIL: {exc}"
        traceback.print_exc()
    finally:
        try:
            tool.close()
        except Exception as exc:
            print(f"  [{label}] close failed: {exc}")


targets = sys.argv[1:] or ["bb-provider", "bb-openai", "bb-gateway", "steel"]

if "bb-provider" in targets:
    print("== Browserbase + Anthropic provider key ==")
    run(
        Stagehand(
            connection=BrowserbaseConnection(model_api_key=os.environ["ANTHROPIC_API_KEY"]),
            model_name=MODEL,
            is_return_screenshot_bytes_enabled=True,
            is_return_live_view_url_enabled=True,
        ),
        "bb-provider",
        ACTIONS,
    )

if "bb-openai" in targets:
    print("== Browserbase + OpenAI provider key ==")
    run(
        Stagehand(
            connection=BrowserbaseConnection(model_api_key=os.environ["OPENAI_API_KEY"]),
            model_name=OPENAI_MODEL,
        ),
        "bb-openai",
        ACTIONS,
    )

if "bb-gateway" in targets:
    print("== Browserbase Model Gateway (no model key) ==")
    run(
        Stagehand(
            connection=BrowserbaseConnection(model_api_key=None),
            model_name=GATEWAY_MODEL,
        ),
        "bb-gateway",
        ACTIONS[:2],
    )

if "steel" in targets:
    print("== Steel cloud (bundled local Stagehand server) ==")
    run(
        Stagehand(
            connection=SteelBrowser(model_api_key=os.environ["ANTHROPIC_API_KEY"]),
            model_name=MODEL,
        ),
        "steel",
        ACTIONS[:2],
    )

print("\n== SUMMARY ==")
for label, verdict in RESULTS.items():
    print(f"  {label}: {verdict}")
sys.exit(1 if any(v != "PASS" for v in RESULTS.values()) else 0)
