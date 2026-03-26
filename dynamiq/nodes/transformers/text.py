from typing import Any, ClassVar, Literal

from jinja2 import Environment, meta
from pydantic import BaseModel, ConfigDict, Field

from dynamiq.nodes import Node, NodeGroup
from dynamiq.nodes.node import ensure_config
from dynamiq.runnables import RunnableConfig


class TextTemplateInputSchema(BaseModel):
    template: str | None = Field(None, description="Parameter to provide template")
    model_config = ConfigDict(extra="allow", strict=True, arbitrary_types_allowed=True)


class TextTemplate(Node):
    group: Literal[NodeGroup.TRANSFORMERS] = NodeGroup.TRANSFORMERS
    name: str = "Text Template"
    description: str = "Node that replaces placeholders in a text template with corresponding input values."
    template: str | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_schema: ClassVar[type[TextTemplateInputSchema]] = TextTemplateInputSchema

    def __init__(self, **data):
        super().__init__(**data)

        from jinja2 import Template

        self._Template = Template

    @staticmethod
    def get_required_parameters(text: str) -> set[str]:
        """Extracts set of parameters required for messages.

        Returns:
            set[str]: Set of parameter names.
        """
        parameters = set()

        env = Environment(autoescape=True)
        ast = env.parse(text)
        parameters |= meta.find_undeclared_variables(ast)
        return parameters

    def validate_input_fields(self, input_data: TextTemplateInputSchema):
        template = input_data.template or self.template
        if template:
            required_parameters = TextTemplate.get_required_parameters(template)
            provided_parameters = set(input_data.model_dump().keys())

            if not required_parameters.issubset(provided_parameters):
                raise ValueError(
                    f"Invalid parameters were provided. Expected: {required_parameters}. " f"Got: {provided_parameters}"
                )
            return

        raise ValueError("Text template was not provided.")

    def execute(self, input_data: TextTemplateInputSchema, config: RunnableConfig = None, **kwargs) -> dict[str, Any]:
        """
        Replace placeholders in the given text with actual values.

        Args:
            input_data (TextTemplateInputSchema): input data for the tool, which includes actual values.
            config (RunnableConfig, optional): Configuration for the runnable, including callbacks.
            **kwargs: Additional arguments passed to the execution context.

        Returns:
            dict[str, Any]: A dictionary containing transformed text.
        """
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        self.validate_input_fields(input_data)
        template = input_data.template or self.template
        try:
            if not template:
                raise ValueError("Text template was not provided.")
            result = self._Template(template).render(**dict(input_data))
            return {"content": result}
        except Exception as e:
            raise ValueError(f"Encountered an error while performing transforming. \nError details: {e}")
