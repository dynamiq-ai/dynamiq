from enum import Enum
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field
from stagehand import StagehandConfig
from stagehand.sync import Stagehand

from dynamiq.connections import Stagehand as StagehandConnection
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ConnectionNode
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

DESCRIPTION_STAGEHAND = """## Stagehand Tool
### Description
A headless browser automation and observation tool designed to navigate, interact with,
and extract structured data from web pages using natural language instructions.

### Capabilities
- Navigate to a given web page URL.
- Observe and list candidate interactive elements on the page.
- Extract structured information using natural language prompts.
- Perform actions like clicks, selections, and form filling.

### Parameters
- `action_type`: Must be one of the following:
  - `goto`: Navigate to the specified URL.
  - `observe`: Return a list of candidate DOM elements based on the instruction.
  - `extract`: Extract structured data as described in the instruction.
  - `act`: Perform a user-specified action (e.g., click a button, select an option).
- `instruction`: A natural language prompt specifying the action to perform. Required for all actions except `goto`.
- `url`: The web address to navigate to. Required only when `action_type` is `goto`.

### Usage Examples
1. Navigate to a web page:
   ```json
   {
     "action_type": "goto",
     "url": "https://example.com"
   }
    ````

2. Observe candidate elements:

   ```json
   {
     "action_type": "observe",
     "instruction": "Find all clickable links on the homepage"
   }
   ```

3. Extract data:

   ```json
   {
     "action_type": "extract",
     "instruction": "Get the list of product names and prices"
   }
   ```

4. Perform an action:

   ```json
   {
     "action_type": "act",
     "instruction": "Click the 'Add to Cart' button for the first product"
   }
   ```

### Tips

- Use clear, specific instructions for better extraction and action reliability.
- Break down complex tasks into sequential steps.
- `observe` is useful for debugging or when planning a follow-up action.
"""


class StagehandActionType(str, Enum):
    ACT = "act"
    EXTRACT = "extract"
    OBSERVE = "observe"
    GOTO = "goto"


class AvailableModel(str, Enum):
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    CLAUDE_3_5_SONNET_LATEST = "claude-3-5-sonnet-latest"
    CLAUDE_3_7_SONNET_LATEST = "claude-3-7-sonnet-latest"


class StagehandInputSchema(BaseModel):
    action_type: StagehandActionType = Field(
        ...,
        description="""
    Specifies the type of interaction with the web page:

    - `act`: Perform an action on the current page (e.g., click, fill, select).
    - `extract`: Retrieve structured data from the page based on the given instruction.
    - `observe`: Return a list of candidate DOM elements for potential interaction.
    - `goto`: Navigate to the specified URL.
    """,
    )

    instruction: str | None = Field(
        None,
        description="""
    A natural language instruction describing the action, data to extract, or elements to observe.
    Required for all action types except `goto`.
    """,
    )

    url: str | None = Field(
        None,
        description="""
    The target URL to navigate to. Required only when `action_type` is `goto`.
    """,
    )


class StagehandTool(ConnectionNode):
    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "Stagehand Tool"
    description: str = DESCRIPTION_STAGEHAND
    connection: StagehandConnection = StagehandConnection()
    model_name: str = Field(AvailableModel.GPT_4O, description="Name of the model to use")
    extra_config: dict[str, Any] = Field(
        default_factory=dict, description="Additional options to pass into StagehandConfig"
    )

    input_schema: ClassVar[type[StagehandInputSchema]] = StagehandInputSchema
    session: Stagehand | None = None

    def start_session(self):
        """
        Initializes a Stagehand session.

        Raises:
            ToolExecutionException: If a required API key is missing or the model is unknown.
        """
        if self.model_name in {AvailableModel.GPT_4O, AvailableModel.GPT_4O_MINI}:
            api_key = self.connection.openai_api_key
            if not api_key:
                raise ToolExecutionException("Missing OpenAI API key for selected model", recoverable=False)
        elif self.model_name in {AvailableModel.CLAUDE_3_5_SONNET_LATEST, AvailableModel.CLAUDE_3_7_SONNET_LATEST}:
            api_key = self.connection.anthropic_api_key
            if not api_key:
                raise ToolExecutionException("Missing Anthropic API key for selected model", recoverable=False)
        else:
            raise ToolExecutionException(f"Unknown model selected: {self.model_name}", recoverable=False)

        config = StagehandConfig(
            env="BROWSERBASE",
            api_key=self.connection.browserbase_api,
            project_id=self.connection.browserbase_project_id,
            model_name=self.model_name,
            **self.extra_config,
        )

        stagehand = Stagehand(config=config, server_url=self.connection.server_url, model_api_key=api_key)
        stagehand.init()
        self.session = stagehand

    def execute(
        self, input_data: StagehandInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        """
        Executes a web automation task using the Stagehand session based on the input action type.

        Args:
            input_data (StagehandInputSchema): The schema describing the action to perform.
            config (RunnableConfig, optional): Execution config if applicable.
            **kwargs: Additional arguments.

        Returns:
            dict[str, Any]: The result of the action.

        Raises:
            ToolExecutionException: If the input is invalid or execution fails.
        """
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")
        if self.session is None:
            self.start_session()

        try:
            if input_data.action_type == StagehandActionType.EXTRACT:
                result = self.session.page.extract(input_data.instruction)
                result = result.model_dump()
            elif input_data.action_type == StagehandActionType.OBSERVE:
                result = self.session.page.observe(input_data.instruction)
                result = [el.model_dump() for el in result]
            elif input_data.action_type == StagehandActionType.ACT:
                result = self.session.page.act(input_data.instruction)
                result = result.model_dump()
            elif input_data.action_type == StagehandActionType.GOTO:
                if input_data.url is None:
                    raise ToolExecutionException(
                        "Missing required URL for 'navigate' action. Please provide a valid URL.", recoverable=True
                    )
                self.session.page.goto(input_data.url)
                result = "Navigated to " + input_data.url
            else:
                raise ToolExecutionException(f"Invalid action type: {input_data.action_type}", recoverable=True)

        except Exception as e:
            raise ToolExecutionException(f"Error message: {e}", recoverable=True)

        logger.info(f"Tool {self.name} - {self.id}: finished with result:\n{str(result)[:200]}...")

        return {"content": result}

    def __del__(self):
        """
        Cleans up the Stagehand session.
        """
        if self.session is not None:
            self.session.close()
