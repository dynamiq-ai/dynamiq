from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from dynamiq import Workflow
from dynamiq.flows import Flow
from dynamiq.nodes.tools.python import Python
from dynamiq.runnables import RunnableConfig
from dynamiq.types.telemetry import ContentCaptureMode, TracingConfig

CODE = """
def run(input_data):
    return {"content": input_data["text"].upper()}
"""

# A provider passed through the run config is used for this run only; the global provider is not touched.
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

workflow = Workflow(flow=Flow(nodes=[Python(name="shout", code=CODE)]), name="run-settings-demo")

config = RunnableConfig(
    tracing=TracingConfig(
        tracer_provider=provider,
        session_id="session-42",
        user_id="user-7",
        tags=["demo", "examples"],
        metadata={"environment": "local", "attempt": 1},
        # Inputs and outputs are left off the spans; ids, names, status and timing are still recorded.
        content_capture=ContentCaptureMode.NO_CONTENT,
    )
)

result = workflow.run(input_data={"text": "hello"}, config=config)
print(result.output)

provider.shutdown()
