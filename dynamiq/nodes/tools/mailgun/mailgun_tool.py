from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from dynamiq.connections import HttpApiKey
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ConnectionNode, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger


class MailGunInputSchema(BaseModel):
    from_email: str = Field(
        description="Email address for From header (str), allowed domains: yourdomain.com",
    )
    to_emails: str | list[str] = Field(
        description="Email address(es) of the recipient(s) (str or list[str])",
    )
    subject: str = Field(
        description="Message subject (str)",
    )

    text: str | None = Field(default=None, description="Body of the message (text version) (str)")
    html: str | None = Field(default=None, description="Body of the message (HTML version) (str)")
    cc: str | list[str] | None = Field(default=None, description="CC recipients (str or list[str])")
    bcc: str | list[str] | None = Field(default=None, description="BCC recipients (str or list[str])")


class MailGunTool(ConnectionNode):
    """
    A tool for sending emails using the Mailgun API.

    Attributes:
        name (str): Name of the tool
        description (str): Description of the tool
        group (Literal[NodeGroup.TOOLS]): The group the node belongs to
        connection (HttpApiKey): The Mailgun API connection
        domain_name (str): Mailgun domain name
        success_codes (list[int]): Expected successful response codes
        timeout (float): Request timeout in seconds
    """

    name: str = "Mailgun Email Sender"
    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    connection: HttpApiKey
    domain_name: str
    success_codes: list[int] = [200]
    timeout: float = 30
    input_schema: ClassVar[type[MailGunInputSchema]] = MailGunInputSchema
    description: str = ""

    def __init__(self, **data):
        super().__init__(**data)
        self.description = self._generate_description()

    def _generate_description(self) -> str:
        """
        Generates a detailed description of the tool based on the input schema.

        Returns:
            str: A formatted description of the tool and its capabilities
        """
        schema_fields: dict[str, Any] = self.input_schema.model_fields
        logger.debug(f"Tool {self.name} - Generating description from schema fields")

        desc: list[str] = [
            "Mailgun Email Sending Tool for sending simple emails with text or HTML content.\n",
            "Required Parameters:",
        ]

        optional_fields: list[str] = [
            name for name, field in schema_fields.items() if field.default is not None or field.is_required is False
        ]

        if optional_fields:
            for field_name in sorted(optional_fields):
                field = schema_fields[field_name]
                desc.append(f"- {field_name}: {field.description}")
        else:
            desc.append("None")
        desc.append("\nOptional Parameters:")

        required_fields: list[str] = [
            name for name, field in schema_fields.items() if field.default is None and field.is_required is not False
        ]

        if required_fields:
            for field_name in sorted(required_fields):
                field = schema_fields[field_name]
                desc.append(f"- {field_name}: {field.description}")
        else:
            desc.append("None")

        return "\n".join(desc)

    def _prepare_request_data(self, input_data: MailGunInputSchema) -> dict:
        """Prepare the request data from input schema."""
        logger.debug(f"Tool {self.name} - Preparing request data from input schema")

        try:
            data = {
                "from": input_data.from_email,
                "subject": input_data.subject,
            }

            if isinstance(input_data.to_emails, str):
                data["to"] = input_data.to_emails
            else:
                data["to"] = ", ".join(input_data.to_emails)

            if input_data.text:
                data["text"] = input_data.text
            if input_data.html:
                data["html"] = input_data.html

            if input_data.cc:
                if isinstance(input_data.cc, str):
                    data["cc"] = input_data.cc
                else:
                    data["cc"] = ", ".join(input_data.cc)

            if input_data.bcc:
                if isinstance(input_data.bcc, str):
                    data["bcc"] = input_data.bcc
                else:
                    data["bcc"] = ", ".join(input_data.bcc)

            logger.debug(f"Tool {self.name} - Request data prepared successfully")
            return data

        except Exception as e:
            logger.error(f"Tool {self.name} - Error preparing request data: {str(e)}")
            raise ToolExecutionException(f"Failed to prepare email data: {str(e)}", recoverable=True)

    def execute(self, input_data: MailGunInputSchema, config: RunnableConfig = None, **kwargs):
        """Execute the email sending operation.

        Args:
            input_data (MailGunInputSchema): The input data containing email details
            config (RunnableConfig, optional): Configuration for the execution
            **kwargs: Additional keyword arguments

        Returns:
            dict: A dictionary containing:
                - content (dict): The API response content
                - status_code (int): The HTTP status code

        Raises:
            ToolExecutionException: If the API request fails or required parameters are missing
            ValueError: If neither text nor html content is provided
        """
        logger.debug(f"Tool {self.name} - Starting email execution with input data: {input_data.model_dump()}")

        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        if not (input_data.text or input_data.html):
            logger.error(f"Tool {self.name} - No content provided (neither text nor HTML)")
            raise ToolExecutionException(
                "Either text or html content is required. Please provide at least one content type.", recoverable=True
            )

        try:
            base_url = self.connection.url or "https://api.mailgun.net/v3"
            url = f"{base_url}/{self.domain_name}/messages"
            logger.debug(f"Tool {self.name} - Using API URL: {url}")

            data = self._prepare_request_data(input_data)

            logger.debug(f"Tool {self.name} - Sending email request")
            response = self.client.request(
                method="POST",
                url=url,
                auth=("api", self.connection.api_key),
                data=data,
                timeout=self.timeout,
            )

            if response.status_code not in self.success_codes:
                error_message = f"Mailgun API request failed with status code: {response.status_code}"
                logger.error(f"Tool {self.name} - {error_message}")

                recoverable = response.status_code in [400, 401, 402, 422]
                raise ToolExecutionException(f"{error_message} and response: {response.text}", recoverable=recoverable)

            logger.debug(f"Tool {self.name} - Email sent successfully")
            return {"content": response.json(), "status_code": response.status_code}

        except ToolExecutionException:
            raise
        except Exception as e:
            logger.error(f"Tool {self.name} - Unexpected error during execution: {str(e)}")
            raise ToolExecutionException(f"Unexpected error while sending email: {str(e)}", recoverable=False)
