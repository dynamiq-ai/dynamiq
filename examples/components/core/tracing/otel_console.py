from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from dynamiq import Workflow
from dynamiq.flows import Flow
from dynamiq.nodes.node import NodeDependency
from dynamiq.nodes.tools.python import Python

CODE = """
def run(input_data):
    return {"content": input_data["text"].upper()}
"""

# Dynamiq only calls the OpenTelemetry API. Spans are recorded once the application installs an SDK provider.
provider = TracerProvider()
# SimpleSpanProcessor exports each span as it ends, which keeps the console output in order for a short script.
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)

# Nodes that make no LLM call, so the example runs without credentials. An agent or LLM node is traced the same way.
shout = Python(name="shout", code=CODE)
repeat = Python(
    name="repeat",
    code='def run(input_data):\n    return {"content": input_data["text"] * 2}',
    depends=[NodeDependency(shout)],
    input_mapping={"text": shout.outputs.content},
)

workflow = Workflow(flow=Flow(nodes=[shout, repeat]), name="console-trace-demo")

result = workflow.run(input_data={"text": "hello"})
print(result.output)

provider.shutdown()
