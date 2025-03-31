import logging
import os
from typing import Any
from uuid import UUID

from langfuse.utils import _get_timestamp
from langfuse.utils.base_callback_handler import LangfuseBaseCallbackHandler
from pydantic import BaseModel

from dynamiq import Workflow
from dynamiq.callbacks.base import BaseCallbackHandler, get_parent_run_id, get_run_id
from dynamiq.flows import Flow
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents import Agent
from dynamiq.runnables import RunnableConfig
from dynamiq.utils import format_value
from examples.llm_setup import setup_llm

logger = logging.getLogger(__name__)


class LangfuseCallbackHandler(LangfuseBaseCallbackHandler, BaseCallbackHandler):
    def __init__(
        self,
        public_key: str | None = None,
        secret_key: str | None = None,
        host: str | None = None,
        session_id: str | None = None,
        trace_name: str | None = None,
        release: str | None = None,
        version: str | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> None:
        LangfuseBaseCallbackHandler.__init__(
            self,
            public_key=public_key,
            secret_key=secret_key,
            host=host,
            session_id=session_id,
            trace_name=trace_name,
            release=release,
            version=version,
            metadata=metadata,
            tags=tags,
            sdk_integration="dynamiq",
        )

        self.runs = {}

    def on_workflow_start(self, serialized: dict[str, Any], input_data: dict[str, Any], **kwargs: Any):
        run_id = get_run_id(kwargs)
        self._generate_trace_and_parent(
            serialized=serialized,
            inputs=format_value(input_data)[0],
            run_id=run_id,
            version=self.version,
        )

        content = {
            "name": "Workflow",
            "metadata": {"workflow": {"id": serialized.get("id"), "version": serialized.get("version")}},
            "input": format_value(input_data)[0],
            "version": self.version,
        }

        if self.root_span is None:
            self.runs[run_id] = self.trace.span(**content)
        else:
            self.runs[run_id] = self.root_span.span(**content)

    def on_workflow_end(self, serialized: dict[str, Any], output_data: dict[str, Any], **kwargs: Any):
        run_id = get_run_id(kwargs)

        self.runs[run_id] = self.runs[run_id].end(
            output=format_value(output_data)[0],
            version=self.version,
            end_time=_get_timestamp(),
        )
        self._update_trace_and_remove_state(run_id, None, output_data)

    def on_workflow_error(self, serialized: dict[str, Any], error: BaseException, **kwargs: Any):
        run_id = get_run_id(kwargs)

        self.runs[run_id] = self.runs[run_id].end(
            level="ERROR",
            status_message=str(error),
            version=self.version,
        )

        self._update_trace_and_remove_state(run_id, None, error)

    def on_flow_start(self, serialized: dict[str, Any], input_data: dict[str, Any], **kwargs: Any):
        run_id = get_run_id(kwargs)
        parent_run_id = get_parent_run_id(kwargs)

        content = {
            "name": "Flow",
            "metadata": {"flow": {"id": serialized.get("id")}},
            "input": format_value(input_data)[0],
            "version": self.version,
        }
        self.runs[run_id] = self.runs[parent_run_id].span(**content)

    def on_flow_end(self, serialized: dict[str, Any], output_data: dict[str, Any], **kwargs: Any):
        run_id = get_run_id(kwargs)
        parent_run_id = get_parent_run_id(kwargs)

        self.runs[run_id] = self.runs[run_id].end(
            output=format_value(output_data)[0],
            version=self.version,
            end_time=_get_timestamp(),
        )
        self._update_trace_and_remove_state(run_id, parent_run_id, output_data)

    def on_flow_error(self, serialized: dict[str, Any], error: BaseException, **kwargs: Any):
        run_id = get_run_id(kwargs)
        parent_run_id = get_parent_run_id(kwargs)

        self.runs[run_id] = self.runs[run_id].end(
            level="ERROR",
            status_message=str(error),
            version=self.version,
        )

        self._update_trace_and_remove_state(run_id, parent_run_id, error)

    def on_node_start(self, serialized: dict[str, Any], input_data: dict[str, Any], **kwargs: Any):
        run_id = get_run_id(kwargs)
        parent_run_id = get_parent_run_id(kwargs)
        metadata = {"node": serialized, "run_depends": kwargs.get("run_depends", [])}
        self._generate_trace_and_parent(
            serialized,
            inputs=format_value(input_data)[0],
            run_id=run_id,
            parent_run_id=parent_run_id,
            metadata=metadata,
            version=self.version,
        )

        content = {
            "name": serialized.get("name"),
            "input": format_value(input_data)[0],
            "metadata": metadata,
            "version": self.version,
        }
        if serialized.get("group") == NodeGroup.LLMS:
            prompt = kwargs.get("prompt") or serialized.get("prompt")
            if isinstance(prompt, BaseModel):
                prompt = prompt.model_dump()

            content["model"] = serialized.get("model")
            content["input"]["prompt"] = prompt

            if parent_run_id in self.runs:
                self.runs[run_id] = self.runs[parent_run_id].generation(**content)
            elif self.root_span is not None and parent_run_id is None:
                self.runs[run_id] = self.root_span.generation(**content)
            else:
                self.runs[run_id] = self.trace.generation(**content)
        else:
            self.runs[run_id] = self.runs[parent_run_id].span(**content)

    def on_node_end(self, serialized: dict[str, Any], output_data: dict[str, Any], **kwargs: Any):
        run_id = get_run_id(kwargs)
        parent_run_id = get_parent_run_id(kwargs)

        content = {
            "output": format_value(output_data)[0],
            "version": self.version,
            "end_time": _get_timestamp(),
        }
        if serialized.get("group") == NodeGroup.LLMS:
            content["usage"] = None  # used special Langfuse UsageModel, so skipped for now
            content["model"] = serialized.get("model")

        self.runs[run_id] = self.runs[run_id].end(**content)
        self._update_trace_and_remove_state(run_id, parent_run_id, output_data)

    def on_node_error(self, serialized: dict[str, Any], error: BaseException, **kwargs: Any):
        run_id = get_run_id(kwargs)
        parent_run_id = get_parent_run_id(kwargs)
        self.runs[run_id] = self.runs[run_id].end(
            status_message=str(error),
            level="ERROR",
            version=self.version,
        )
        self._update_trace_and_remove_state(run_id, parent_run_id, error)

    def _generate_trace_and_parent(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any] | list[str] | str | None,
        run_id: UUID,
        version: str | None = None,
        parent_run_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        try:
            class_name = serialized.get("name") or serialized.get("id")

            # on a new invocation, and not user provided root, we want to initialise a new trace
            # parent_run_id is None when we are at the root of a langchain execution
            if self.trace is not None and parent_run_id is None and self.langfuse is not None:
                self.trace = None

            if (
                self.trace is not None
                and parent_run_id is None
                and self.langfuse is None  # StatefulClient was provided by user
                and self.update_stateful_client
            ):
                params = {
                    "name": self.trace_name if self.trace_name is not None else class_name,
                    "metadata": metadata,
                    "version": version or self.version,
                    "session_id": self.session_id,
                    "user_id": self.user_id,
                    "tags": self.tags,
                    "input": inputs,
                }

                if self.root_span:
                    self.root_span.update(**params)
                else:
                    self.trace.update(**params)

            # if we are at a root, but langfuse exists, it means we do not have a
            # root provided by a user. Initialise it by creating a trace and root span.
            if self.trace is None and self.langfuse is not None:
                trace = self.langfuse.trace(
                    id=str(run_id),
                    name=self.trace_name if self.trace_name is not None else class_name,
                    metadata=metadata,
                    version=self.version,
                    session_id=self.session_id,
                    user_id=self.user_id,
                    tags=self.tags,
                    input=inputs,
                )

                self.trace = trace

                if parent_run_id is not None and parent_run_id in self.runs:
                    self.runs[run_id] = self.trace.span(
                        trace_id=self.trace.id,
                        name=class_name,
                        metadata=metadata,
                        input=inputs,
                        version=self.version,
                    )

                return

        except Exception as e:
            logger.error(f"Error generating trace and parent: {e}")

    def _update_trace_and_remove_state(
        self,
        run_id: UUID,
        parent_run_id: UUID | None,
        output: any,
        *,
        keep_state: bool = False,
        **kwargs: Any,
    ):
        if (
            parent_run_id is None  # If we are at the root of the langchain execution -> reached the end of the root
            and self.trace is not None  # We do have a trace available
            and self.trace.id == str(run_id)  # The trace was generated by langchain and not by the user
        ):
            self.trace = self.trace.update(output=output, **kwargs)

        elif (
            parent_run_id is None
            and self.trace is not None  # We have a user-provided parent
            and self.update_stateful_client
        ):
            if self.root_span is not None:
                self.root_span = self.root_span.update(output=output, **kwargs)
            else:
                self.trace = self.trace.update(output=output, **kwargs)

        if not keep_state:
            del self.runs[run_id]


if __name__ == "__main__":
    llm = setup_llm()
    agent = Agent(
        name="Agent",
        llm=llm,
        role="Agent, goal is to provide information based on the user input",
    )

    workflow = Workflow(
        flow=Flow(nodes=[agent]),
    )
    langfuse_tracing = LangfuseCallbackHandler(
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        host=os.getenv("LANGFUSE_HOST"),
    )
    workflow.run(
        {"input": "explain the concept of quantum mechanics"},
        config=RunnableConfig(callbacks=[langfuse_tracing]),
    )
