"""Set CONTEXT_DEV_API_KEY, then run this example from the repository root."""

from dynamiq.connections import Context
from dynamiq.nodes.tools import ContextSearchTool
from dynamiq.runnables import RunnableStatus
from dynamiq.utils.logger import logger

if __name__ == "__main__":
    tool = ContextSearchTool(connection=Context(), max_results=5, is_optimized_for_agents=True)
    result = tool.run(input_data={"query": "Python standard library documentation"})
    if result.status != RunnableStatus.SUCCESS:
        raise RuntimeError(result.error.message)
    logger.info(result.output["content"])
