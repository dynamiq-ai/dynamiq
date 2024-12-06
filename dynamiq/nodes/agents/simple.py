from dynamiq.nodes.agents.base import Agent


class SimpleAgent(Agent):
    """Agent that uses the Simple strategy for processing tasks."""

    name: str = "Agent Simple"
