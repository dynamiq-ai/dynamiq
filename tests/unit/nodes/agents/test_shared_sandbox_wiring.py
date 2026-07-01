def test_cleanup_skips_shared_sandbox():
    from unittest.mock import MagicMock

    from dynamiq.nodes.tools.agent_tool import SubAgentTool

    agent = MagicMock()
    agent._sandbox_is_shared = True
    SubAgentTool.cleanup_factory_agent(agent)
    agent.sandbox_backend.close.assert_not_called()


def test_cleanup_kills_owned_sandbox():
    from unittest.mock import MagicMock

    from dynamiq.nodes.tools.agent_tool import SubAgentTool

    agent = MagicMock()
    agent._sandbox_is_shared = False
    SubAgentTool.cleanup_factory_agent(agent)
    agent.sandbox_backend.close.assert_called_once_with(kill=True)
