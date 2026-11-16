"""OTLP/JSON and OTLP/protobuf encodings of finished spans, through the SDK's own OTLP encoder."""

import base64
from typing import Any

from google.protobuf.json_format import MessageToDict
from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

# Bytes fields that OTLP/JSON writes as hex; protobuf's JSON mapping writes bytes as base64.
_ID_FIELDS = frozenset({"traceId", "spanId", "parentSpanId"})


def encode_request(spans: list) -> ExportTraceServiceRequest:
    return encode_spans(spans)


def request_to_json(request: ExportTraceServiceRequest) -> dict[str, Any]:
    """The OTLP/JSON form of ``request``: lowerCamelCase keys, enums as integers, 64-bit integers as strings, trace
    and span ids as lowercase hex."""
    return _hex_ids(MessageToDict(request, use_integers_for_enums=True))


def request_from_bytes(data: bytes) -> ExportTraceServiceRequest:
    request = ExportTraceServiceRequest()
    request.ParseFromString(data)
    return request


def _hex_ids(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: base64.b64decode(item).hex() if key in _ID_FIELDS and isinstance(item, str) else _hex_ids(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_hex_ids(item) for item in value]
    return value
