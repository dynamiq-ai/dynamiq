"""Teardown for the AgentCore e2e fixtures. Requires --yes.

Order: stop live sessions (children) before deleting the interpreter (parent).
Best-effort per resource: one failure must not abort the rest.

Usage:
    python 11_teardown_interpreter.py --yes            # fixture interpreter + its sessions
    python 11_teardown_interpreter.py --yes --builtin  # also stop sessions on aws.codeinterpreter.v1
"""

import os
import sys

import boto3
from botocore.exceptions import ClientError

if "--yes" not in sys.argv:
    raise SystemExit("refusing to delete anything without --yes")

NAME = os.getenv("AGENTCORE_E2E_INTERPRETER_NAME", "dynamiq_e2e_public")
REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
PROFILE = os.getenv("AWS_DEFAULT_PROFILE")
BUILTIN = "aws.codeinterpreter.v1"

session = boto3.Session(profile_name=PROFILE) if PROFILE else boto3.Session()
ctl = session.client("bedrock-agentcore-control", region_name=REGION)
dp = session.client("bedrock-agentcore", region_name=REGION)

deleted, skipped = [], []


def stop_sessions(identifier: str) -> None:
    try:
        resp = dp.list_code_interpreter_sessions(codeInterpreterIdentifier=identifier, status="READY")
    except ClientError as e:
        skipped.append(f"list sessions {identifier}: {e.response['Error']['Code']}")
        return
    for item in resp.get("items", []):
        session_id = item["sessionId"]
        try:
            dp.stop_code_interpreter_session(codeInterpreterIdentifier=identifier, sessionId=session_id)
            deleted.append(f"session {identifier}/{session_id}")
        except ClientError as e:
            skipped.append(f"session {session_id}: {e.response['Error']['Code']}")


target = None
for item in ctl.list_code_interpreters().get("codeInterpreterSummaries", []):
    if item.get("name") == NAME:
        target = item["codeInterpreterId"]

if target:
    stop_sessions(target)
    try:
        ctl.delete_code_interpreter(codeInterpreterId=target)
        deleted.append(f"interpreter {target}")
    except ClientError as e:
        skipped.append(f"interpreter {target}: {e.response['Error']['Code']}")
else:
    skipped.append(f"interpreter named {NAME}: not found")

if "--builtin" in sys.argv:
    stop_sessions(BUILTIN)

print("DELETED:")
for line in deleted or ["  (nothing)"]:
    print(f"  {line}")
print("SKIPPED:")
for line in skipped or ["  (nothing)"]:
    print(f"  {line}")
