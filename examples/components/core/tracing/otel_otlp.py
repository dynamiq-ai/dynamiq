from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from dynamiq import Workflow
from dynamiq.flows import Flow
from dynamiq.nodes.tools.python import Python

# Configuration comes from the standard OpenTelemetry variables, which the exporter and Resource read themselves:
#   OTEL_EXPORTER_OTLP_ENDPOINT   base URL, for example http://localhost:4318 for a local OpenTelemetry Collector
#                                 or the OTLP/HTTP URL of a backend that accepts OTLP. The exporter appends /v1/traces.
#   OTEL_EXPORTER_OTLP_HEADERS    comma-separated key=value pairs, typically the backend's authorization header.
#   OTEL_SERVICE_NAME             name the backend shows for this application.
# Example:
#   OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 OTEL_SERVICE_NAME=dynamiq-demo python otel_otlp.py

CODE = """
def run(input_data):
    return {"content": input_data["text"].upper()}
"""

provider = TracerProvider(resource=Resource.create())
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
trace.set_tracer_provider(provider)

workflow = Workflow(flow=Flow(nodes=[Python(name="shout", code=CODE)]), name="otlp-trace-demo")

try:
    result = workflow.run(input_data={"text": "hello"})
    print(result.output)
finally:
    # BatchSpanProcessor exports in a background thread; without a flush the last spans are lost at exit.
    provider.force_flush()
    provider.shutdown()
