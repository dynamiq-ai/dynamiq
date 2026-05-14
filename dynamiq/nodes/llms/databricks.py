from dynamiq.connections import Databricks as DatabricksConnection
from dynamiq.nodes.llms.base import BaseLLM
from dynamiq.nodes.llms.openai import ReasoningEffort


class Databricks(BaseLLM):
    """Databricks LLM node.

    This class provides an implementation for the Databricks Language Model node.

    Attributes:
        connection (DatabricksConnection): The connection to use for the Databricks LLM.
        MODEL_PREFIX (str): The prefix for the Databricks model name.
        reasoning_effort (ReasoningEffort | None): Controls the depth and complexity of reasoning
        performed by the model.
    """

    connection: DatabricksConnection
    MODEL_PREFIX = "databricks/"
    reasoning_effort: ReasoningEffort | None = ReasoningEffort.MEDIUM

    def __init__(self, **kwargs):
        """Initialize the Databricks LLM node.

        Args:
            **kwargs: Additional keyword arguments.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = DatabricksConnection()
        super().__init__(**kwargs)
