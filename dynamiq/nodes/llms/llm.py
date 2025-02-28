import logging
import time
from typing import Any, Literal

from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.nodes import Node, NodeGroup
from dynamiq.nodes.node import ensure_config
from dynamiq.runnables import RunnableConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TracingWrapper(Node):
    group: Literal[NodeGroup.UTILS] = NodeGroup.UTILS

    def __init__(self, obj: Any, path: str = "openai", trace: bool = False, **kwargs):
        super().__init__(**kwargs)
        self._obj = obj
        self.name = path
        self._path = path
        self._trace = trace

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._obj, name, None)
        if attr is None:
            raise AttributeError(f"'{self._path}' has no attribute '{name}'")

        if callable(attr):
            return lambda *args, **kwargs: self.run(
                {"method": attr, "method_path": f"{self._path}.{name}", "kwargs": kwargs},
                config=RunnableConfig(callbacks=[TracingCallbackHandler()]) if self._trace is True else None,
            ).output

        return TracingWrapper(attr, f"{self._path}.{name}", self._trace)

    def __call__(self, *args, **kwargs):
        if callable(self._obj):
            return self.run(
                {"method": self._obj, "path": self._path, "kwargs": kwargs},
                config=RunnableConfig(callbacks=[TracingCallbackHandler()]) if self._trace is True else None,
            ).output
        raise TypeError(f"'{self._path}' object is not callable")

    def execute(self, input_data: dict[str, Any], config: RunnableConfig = None, **kwargs):

        if self._trace:
            config = RunnableConfig(callbacks=[TracingCallbackHandler()]) if not config else config
        config = ensure_config(config)
        self.run_on_node_execute_run(callbacks=config.callbacks, **kwargs)

        method = input_data.get("method")
        method_path = input_data.get("method_path")
        method_kwargs = input_data.get("kwargs")
        start_time = time.time()
        try:
            result = method(**method_kwargs)
            if kwargs.get("stream", False):
                for chunk in result:
                    self.run_on_node_execute_stream(config.callbacks, chunk.model_dump(), **kwargs)
            else:
                self.run_on_node_execute_run(callbacks=config.callbacks, **kwargs)

            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Failed {method_path} in {duration:.2f}s with error: {str(e)}")
            raise


class LLMDynamiqClient:

    def __init__(self, api_key: str | None = None, trace=False):
        self.api_key = api_key
        self.trace = trace

    def __getattr__(self, name: str) -> Any:
        if name == "openai":
            import openai as openai_module

            return TracingWrapper(openai_module, "openai", self.trace)

        elif name == "anthropic":
            import anthropic as anthropic_module

            return TracingWrapper(anthropic_module, "anthropic", self.trace)

        else:
            raise AttributeError(f"Module {__name__} has no attribute {name}")
