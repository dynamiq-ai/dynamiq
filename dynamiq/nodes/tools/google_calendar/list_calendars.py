from typing import Any, ClassVar, Literal

from pydantic import BaseModel

from dynamiq.connections import GoogleOAuth2
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ensure_config
from dynamiq.nodes.tools.google_calendar.google_calendar_base import GoogleCalendarBase
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

DESCRIPTION_LIST_CALENDARS = """## List Calendars Tool
Retrieves the list of calendars available in the user's Google Calendar account.

### Parameters
No input parameters required
"""


class ListCalendars(GoogleCalendarBase):
    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "list-calendars"
    description: str = DESCRIPTION_LIST_CALENDARS
    input_schema: ClassVar[type[BaseModel]] = BaseModel
    connection: GoogleOAuth2 = GoogleOAuth2()

    def execute(self, input_data: BaseModel, config: RunnableConfig | None = None, **kwargs) -> dict[str, Any]:
        """
        Lists the calendars available to the authenticated user.

        Args:
            input_data (BaseModel): No input parameters required.
            config (RunnableConfig | None): Optional runtime configuration.
            **kwargs: Additional arguments.

        Returns:
            dict[str, Any]: A list of calendars.

        Raises:
            ToolExecutionException: If fetching calendars fails.
        """
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        try:
            calendar_list = self.client.calendarList().list().execute()
            result = calendar_list.get("items", [])

            logger.info(f"Tool {self.name} - {self.id}: finished with result:\n{str(result)[:200]}...")
            return {"content": result}

        except Exception as e:
            raise ToolExecutionException(str(e), recoverable=True)

    def close(self):
        self.client.close()
