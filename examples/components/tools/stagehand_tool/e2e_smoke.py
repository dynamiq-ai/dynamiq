"""End-to-end smoke test for the Stagehand tool on the v3 SDK.

Requires outbound access to api.browserbase.com / api.stagehand.browserbase.com (and api.steel.dev
for the Steel path). Keys come from environment variables only:

    BROWSERBASE_API_KEY, BROWSERBASE_PROJECT_ID, ANTHROPIC_API_KEY, STEEL_API_KEY

Run all paths, or a subset:

    python e2e_smoke.py                 # all
    python e2e_smoke.py bb-provider     # Browserbase + provider model key
    python e2e_smoke.py bb-gateway      # Browserbase Model Gateway (no model key)
    python e2e_smoke.py steel           # Steel cloud via the bundled local Stagehand server

Steel path note: leave ANTHROPIC_BASE_URL unset (the bundled Stagehand server then defaults to
https://api.anthropic.com/v1); if your environment sets it, the value must include the /v1 suffix
or anthropic/* model calls 404.
"""

import os
import sys
import traceback

from dynamiq.connections import Browserbase as BrowserbaseConnection
from dynamiq.connections import SteelBrowser
from dynamiq.nodes.tools.stagehand import Stagehand, StagehandInputSchema

# claude-sonnet-5 is NOT usable here: its observe fails 100% of the time server-side (bare 502),
# and the Browserbase Model Gateway rejects it outright (400, unsupported model).
MODEL = "anthropic/claude-sonnet-4-6"
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


targets = sys.argv[1:] or ["bb-provider", "bb-gateway", "steel"]

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

if "bb-gateway" in targets:
    print("== Browserbase Model Gateway (no model key) ==")
    run(
        Stagehand(
            connection=BrowserbaseConnection(model_api_key=None),
            model_name=MODEL,
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
