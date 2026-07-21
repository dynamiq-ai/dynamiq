"""Integration tests for BedrockAgentCoreInterpreterTool (live AWS AgentCore).

Requires AWS credentials with ``bedrock-agentcore:*CodeInterpreterSession*`` /
``bedrock-agentcore:InvokeCodeInterpreter`` permissions and an explicit
``BEDROCK_AGENTCORE_TESTS=1`` opt-in, since AWS credentials alone do not imply
AgentCore is enabled in the account/region.
"""

import io
import os

import pytest

from dynamiq.connections import AWS as AWSConnection


@pytest.fixture(scope="module")
def agentcore_tool():
    if not os.getenv("BEDROCK_AGENTCORE_TESTS"):
        pytest.skip("BEDROCK_AGENTCORE_TESTS is not set")
    if not os.getenv("AWS_DEFAULT_REGION"):
        pytest.skip("AWS_DEFAULT_REGION is not set")

    from dynamiq.nodes.tools.bedrock_agentcore_sandbox import BedrockAgentCoreInterpreterTool

    tool = BedrockAgentCoreInterpreterTool(
        connection=AWSConnection(),
        persistent_sandbox=True,
        is_optimized_for_agents=False,
    )
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
    from dynamiq.nodes.tools.code_interpreter import CodeInterpreterInputSchema

    result = agentcore_tool.execute(
        CodeInterpreterInputSchema(
            python="with open('./output/result.txt', 'w') as f:\n    f.write('generated')\nprint('saved')"
        )
    )

    files = result["content"].get("files", {})
    assert any("result.txt" in path for path in files)


@pytest.mark.integration
def test_package_installation(agentcore_tool):
    from dynamiq.nodes.tools.code_interpreter import CodeInterpreterInputSchema

    result = agentcore_tool.execute(
        CodeInterpreterInputSchema(packages="pyjokes", python="import pyjokes\nprint('installed')")
    )

    assert "installed" in result["content"]["code_execution"]
