"""Idempotent: create (or find) a PUBLIC-network AgentCore code interpreter and wait for READY.

The built-in ``aws.codeinterpreter.v1`` runs in SANDBOX network mode with no DNS/egress, so
``pip install`` can never work there. A custom interpreter with ``networkMode=PUBLIC`` is the
only way to exercise the package-installation branch.

Prints the interpreter id on the last line. Teardown: 11_teardown_interpreter.py
"""

import os
import sys
import time

import boto3
from botocore.exceptions import ClientError

NAME = os.getenv("AGENTCORE_E2E_INTERPRETER_NAME", "dynamiq_e2e_public")
REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
PROFILE = os.getenv("AWS_DEFAULT_PROFILE")

session = boto3.Session(profile_name=PROFILE) if PROFILE else boto3.Session()
ctl = session.client("bedrock-agentcore-control", region_name=REGION)


def find() -> str | None:
    paginator_items = ctl.list_code_interpreters().get("codeInterpreterSummaries", [])
    for item in paginator_items:
        if item.get("name") == NAME:
            return item["codeInterpreterId"]
    return None


interpreter_id = find()
if interpreter_id:
    print(f"reusing existing interpreter {interpreter_id}", file=sys.stderr)
else:
    try:
        resp = ctl.create_code_interpreter(
            name=NAME,
            description="Ephemeral fixture for dynamiq AgentCore e2e tests (PUBLIC egress).",
            networkConfiguration={"networkMode": "PUBLIC"},
        )
    except ClientError as e:
        print(f"create failed: {e.response['Error']['Code']}: {e.response['Error']['Message']}", file=sys.stderr)
        raise
    interpreter_id = resp["codeInterpreterId"]
    print(f"created interpreter {interpreter_id}", file=sys.stderr)

deadline = time.time() + 300
while time.time() < deadline:
    detail = ctl.get_code_interpreter(codeInterpreterId=interpreter_id)
    status = detail.get("status")
    if status == "READY":
        print(f"READY network={detail.get('networkConfiguration')}", file=sys.stderr)
        break
    if status in ("CREATE_FAILED", "DELETE_FAILED", "DELETING"):
        raise SystemExit(f"interpreter entered terminal state {status}: {detail}")
    print(f"  status={status}, waiting…", file=sys.stderr)
    time.sleep(5)
else:
    raise SystemExit("timed out waiting for interpreter to become READY")

print(interpreter_id)
