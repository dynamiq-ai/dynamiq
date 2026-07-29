"""Integration tests for BedrockAgentCoreInterpreterTool (live AWS AgentCore).

Requires AWS credentials with ``bedrock-agentcore:*CodeInterpreterSession*`` /
``bedrock-agentcore:InvokeCodeInterpreter`` permissions and an explicit
``BEDROCK_AGENTCORE_TESTS=1`` opt-in, since AWS credentials alone do not imply
AgentCore is enabled in the account/region.

``BEDROCK_AGENTCORE_INTERPRETER_ID`` is optional and only needed for the package
installation test: see :func:`egress_agentcore_tool` for why.
"""

import io
import os

import pytest

from dynamiq.connections import AWS as AWSConnection

# A custom interpreter created with networkMode=PUBLIC. The built-in interpreter cannot
# install packages (see egress_agentcore_tool), so that one test needs its own fixture.
EGRESS_INTERPRETER_ID_ENV = "BEDROCK_AGENTCORE_INTERPRETER_ID"


def _require_agentcore() -> None:
    if not os.getenv("BEDROCK_AGENTCORE_TESTS"):
        pytest.skip("BEDROCK_AGENTCORE_TESTS is not set")
    if not os.getenv("AWS_DEFAULT_REGION"):
        pytest.skip("AWS_DEFAULT_REGION is not set")


def _build_tool(**kwargs):
    from dynamiq.nodes.tools.bedrock_agentcore_sandbox import BedrockAgentCoreInterpreterTool

    return BedrockAgentCoreInterpreterTool(
        connection=AWSConnection(),
        persistent_sandbox=True,
        is_optimized_for_agents=False,
        **kwargs,
    )


@pytest.fixture(scope="module")
def agentcore_tool():
    """Tool on the built-in ``aws.codeinterpreter.v1`` interpreter."""
    _require_agentcore()

    tool = _build_tool()
    yield tool
    tool.close()


@pytest.fixture(scope="module")
def egress_agentcore_tool():
    """Tool on an interpreter that has outbound internet access.

    The built-in ``aws.codeinterpreter.v1`` runs in AgentCore's ``SANDBOX`` network mode:
    it has no DNS and no egress, so ``pip install`` can never resolve an index there. Only
    a custom interpreter created with ``networkConfiguration={"networkMode": "PUBLIC"}``
    can install packages, hence the separate opt-in variable — skipping here reports an
    environment gap rather than failing as if the tool were broken.

    ``scripts/agentcore-e2e/10_provision_interpreter.py`` creates a suitable interpreter and
    prints its id; ``11_teardown_interpreter.py --yes`` removes it again.
    """
    _require_agentcore()

    identifier = os.getenv(EGRESS_INTERPRETER_ID_ENV)
    if not identifier:
        pytest.skip(
            f"{EGRESS_INTERPRETER_ID_ENV} is not set; the built-in interpreter has no network "
            "egress, so package installation cannot be exercised"
        )

    tool = _build_tool(code_interpreter_identifier=identifier)
    yield tool
    tool.close()


@pytest.mark.integration
def test_python_execution(agentcore_tool):
    from dynamiq.nodes.tools.code_interpreter import CodeInterpreterInputSchema

    result = agentcore_tool.execute(CodeInterpreterInputSchema(python="print(21 * 2)"))

    assert result["content"]["code_execution"].strip() == "42"


@pytest.mark.integration
def test_python_state_persists_between_executions(agentcore_tool):
    from dynamiq.nodes.tools.code_interpreter import CodeInterpreterInputSchema

    agentcore_tool.execute(CodeInterpreterInputSchema(python="state_marker = 'alive'\nprint(state_marker)"))
    result = agentcore_tool.execute(CodeInterpreterInputSchema(python="print(state_marker)"))

    assert result["content"]["code_execution"].strip() == "alive"


@pytest.mark.integration
def test_shell_command_execution(agentcore_tool):
    from dynamiq.nodes.tools.code_interpreter import CodeInterpreterInputSchema

    result = agentcore_tool.execute(CodeInterpreterInputSchema(shell_command="echo hello-agentcore"))

    assert "hello-agentcore" in result["content"]["shell_command_execution"]


@pytest.mark.integration
def test_file_upload_and_read_back(agentcore_tool):
    from dynamiq.nodes.tools.code_interpreter import CodeInterpreterInputSchema

    test_file = io.BytesIO(b"name,score\nAlice,95\n")
    test_file.name = "scores.csv"

    result = agentcore_tool.execute(
        CodeInterpreterInputSchema(
            files=[test_file],
            python="with open('./input/scores.csv') as f:\n    print(f.read().strip())",
        )
    )

    assert "Alice,95" in result["content"]["code_execution"]


@pytest.mark.integration
def test_output_file_collection(agentcore_tool):
    import base64

    from dynamiq.nodes.tools.code_interpreter import CodeInterpreterInputSchema

    result = agentcore_tool.execute(
        CodeInterpreterInputSchema(
            python="with open('./output/result.txt', 'w') as f:\n    f.write('generated')\nprint('saved')"
        )
    )

    files = result["content"].get("files", {})
    content_b64 = next((v for k, v in files.items() if "result.txt" in k), None)
    assert content_b64 is not None, "output file was not collected"
    assert not content_b64.startswith("Error:"), content_b64
    assert base64.b64decode(content_b64) == b"generated"


@pytest.mark.integration
def test_package_installation(egress_agentcore_tool):
    from dynamiq.nodes.tools.code_interpreter import CodeInterpreterInputSchema

    result = egress_agentcore_tool.execute(
        CodeInterpreterInputSchema(packages="pyjokes", python="import pyjokes\nprint('installed')")
    )

    assert "installed" in result["content"]["code_execution"]


@pytest.mark.integration
def test_package_installation_without_egress_fails_cleanly(agentcore_tool):
    """A pip failure must surface as a recoverable tool error rather than hanging or leaking.

    Asserted conditionally on purpose: the built-in interpreter's ``SANDBOX`` network mode is
    AWS's choice, not ours, so if it ever gains egress this skips instead of turning red.
    """
    from dynamiq.nodes.agents.exceptions import ToolExecutionException
    from dynamiq.nodes.tools.code_interpreter import CodeInterpreterInputSchema

    try:
        agentcore_tool.execute(CodeInterpreterInputSchema(packages="pyjokes", python="print('installed')"))
    except ToolExecutionException as exc:
        assert exc.recoverable is True
        assert "package installation" in str(exc).lower()
    else:
        pytest.skip("built-in interpreter now has network egress; no failure path to assert")
