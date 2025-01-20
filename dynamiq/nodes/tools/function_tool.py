import inspect
from typing import Any, Callable, ClassVar, Generic, Literal, TypeVar

from pydantic import BaseModel, Field, create_model

from dynamiq.nodes import ErrorHandling, NodeGroup
from dynamiq.nodes.node import Node, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

T = TypeVar("T")


class FunctionTool(Node, Generic[T]):
    """
    A tool node for executing a specified function with the given input data.
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "Function Tool"
    description: str = Field(
        default="A tool for executing a function with given input."
    )
    error_handling: ErrorHandling = Field(
        default_factory=lambda: ErrorHandling(timeout_seconds=600)
    )

    def run_func(self, **_: Any) -> Any:
        """
        Execute the function logic with provided arguments.

        This method must be implemented by subclasses.

        :param kwargs: Arguments to pass to the function.
        :return: Result of the function execution.
        """
        raise NotImplementedError("run_func must be implemented by subclasses")

    def execute(self, input_data: dict[str, Any], config: RunnableConfig = None, **kwargs) -> dict[str, Any]:
        """
        Execute the tool with the provided input data and configuration.

        :param input_data: Dictionary of input data to be passed to the tool.
        :param config: Optional configuration for the runnable instance.
        :return: Dictionary with the execution result.
        """
        logger.info(f"Tool {self.name} - {self.id}: started with INPUT DATA:\n{input_data.model_dump()}")
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        result = self.run_func(input_data, config=config, **kwargs)

        logger.info(f"Tool {self.name} - {self.id}: finished with RESULT:\n{str(result)[:200]}...")
        return {"content": result}

    def get_schema(self):
        """
        Generate the schema for the input and output of the tool.

        :return: Dictionary representing the input and output schema.
        """
        cls = self.__class__
        run_tool_method = self.run_func
        if hasattr(cls, "_original_func"):
            run_tool_method = cls._original_func

        signature = inspect.signature(run_tool_method)
        parameters = signature.parameters

        fields = {}
        for name, param in parameters.items():
            if name == "self":
                continue
            annotation = (
                param.annotation if param.annotation != inspect.Parameter.empty else Any
            )
            default = ... if param.default == inspect.Parameter.empty else param.default
            fields[name] = (annotation, default)

        input_model = create_model(f"{cls.__name__}Input", **fields)

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": input_model.schema(),
            "output_schema": {
                "type": "object",
                "properties": {"content": {"type": "any"}},
            },
        }


def function_tool(func: Callable[..., T]) -> type[FunctionTool[T]]:
    """
    Decorator to convert a function into a FunctionTool subclass.

    :param func: Function to be converted into a tool.
    :return: A FunctionTool subclass that wraps the provided function.
    """

    def create_input_schema(func) -> type[BaseModel]:
        signature = inspect.signature(func)

        params_dict = {}

        for param in signature.parameters.values():
            if param.name == "kwargs" or param.name == "config":
                continue
            if param.default is inspect.Parameter.empty:
                params_dict[param.name] = (param.annotation, ...)
            else:
                params_dict[param.name] = (param.annotation, param.default)

        return create_model(
            "FunctionToolInputSchema",
            **params_dict,
            model_config=dict(extra="allow"),
        )

    class FunctionToolFromDecorator(FunctionTool[T]):
        name: str = Field(default=func.__name__)
        description: str = Field(
            default=(func.__doc__ or "") + "\nFunction signature:" + str(inspect.signature(func))
            or f"A tool for executing the {func.__name__} function with signature: {str(inspect.signature(func))}"
        )
        _original_func = staticmethod(func)
        input_schema: ClassVar[type[BaseModel]] = create_input_schema(func)

        def run_func(self, input_data: BaseModel, **kwargs) -> T:
            return func(**input_data.model_dump(), **kwargs)

    FunctionToolFromDecorator.__name__ = func.__name__
    FunctionToolFromDecorator.__qualname__ = (
        f"FunctionToolFromDecorator({func.__name__})"
    )
    FunctionToolFromDecorator.__module__ = func.__module__

    return FunctionToolFromDecorator
