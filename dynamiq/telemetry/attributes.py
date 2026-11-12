"""Names and values of span attributes and events, pinned to OpenTelemetry semantic conventions v1.41.1.

Convention names are hardcoded here rather than imported from ``opentelemetry.semconv``: that package is not a
dependency of the API, and its GenAI constants are frozen and deprecated. Every ``dynamiq.*`` name is Dynamiq's own.
See docs/reference/opentelemetry-tracing.md.
"""

# GenAI convention (v1.41.1)
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_PROVIDER_NAME = "gen_ai.provider.name"
# Deprecated alias of gen_ai.provider.name, still the only provider key several backends read.
GEN_AI_SYSTEM = "gen_ai.system"
GEN_AI_CONVERSATION_ID = "gen_ai.conversation.id"

GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
GEN_AI_REQUEST_TOP_P = "gen_ai.request.top_p"
GEN_AI_REQUEST_TOP_K = "gen_ai.request.top_k"
GEN_AI_REQUEST_FREQUENCY_PENALTY = "gen_ai.request.frequency_penalty"
GEN_AI_REQUEST_PRESENCE_PENALTY = "gen_ai.request.presence_penalty"
GEN_AI_REQUEST_STOP_SEQUENCES = "gen_ai.request.stop_sequences"
GEN_AI_REQUEST_SEED = "gen_ai.request.seed"
GEN_AI_REQUEST_STREAM = "gen_ai.request.stream"
GEN_AI_OUTPUT_TYPE = "gen_ai.output.type"

GEN_AI_RESPONSE_ID = "gen_ai.response.id"
GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
GEN_AI_RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"

GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS = "gen_ai.usage.cache_read.input_tokens"
GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS = "gen_ai.usage.cache_creation.input_tokens"
# Upstream rename of gen_ai.usage.cache_creation.input_tokens after v1.41.1; emitted as an alias.
GEN_AI_USAGE_CACHE_WRITE_INPUT_TOKENS = "gen_ai.usage.cache_write.input_tokens"

GEN_AI_INPUT_MESSAGES = "gen_ai.input.messages"
GEN_AI_OUTPUT_MESSAGES = "gen_ai.output.messages"
GEN_AI_TOOL_DEFINITIONS = "gen_ai.tool.definitions"

GEN_AI_AGENT_NAME = "gen_ai.agent.name"
GEN_AI_AGENT_ID = "gen_ai.agent.id"
GEN_AI_AGENT_DESCRIPTION = "gen_ai.agent.description"

GEN_AI_TOOL_NAME = "gen_ai.tool.name"
GEN_AI_TOOL_TYPE = "gen_ai.tool.type"
GEN_AI_TOOL_CALL_ARGUMENTS = "gen_ai.tool.call.arguments"
GEN_AI_TOOL_CALL_RESULT = "gen_ai.tool.call.result"

GEN_AI_DATA_SOURCE_ID = "gen_ai.data_source.id"
GEN_AI_RETRIEVAL_QUERY_TEXT = "gen_ai.retrieval.query.text"
GEN_AI_RETRIEVAL_DOCUMENTS = "gen_ai.retrieval.documents"

GEN_AI_EMBEDDINGS_DIMENSION_COUNT = "gen_ai.embeddings.dimension.count"

GEN_AI_WORKFLOW_NAME = "gen_ai.workflow.name"

# General conventions
SESSION_ID = "session.id"
USER_ID = "user.id"
ERROR_TYPE = "error.type"
EXCEPTION_TYPE = "exception.type"
EXCEPTION_MESSAGE = "exception.message"
EXCEPTION_STACKTRACE = "exception.stacktrace"

# Dynamiq: every span
DYNAMIQ_RUN_ID = "dynamiq.run.id"
DYNAMIQ_RUN_TYPE = "dynamiq.run.type"
DYNAMIQ_RUN_IS_ROOT = "dynamiq.run.is_root"
DYNAMIQ_RUN_STATUS = "dynamiq.run.status"
DYNAMIQ_SOURCE_ID = "dynamiq.source.id"
DYNAMIQ_TAGS = "dynamiq.tags"
DYNAMIQ_METADATA_PREFIX = "dynamiq.metadata."

# Dynamiq: workflow and flow spans
DYNAMIQ_WORKFLOW_ID = "dynamiq.workflow.id"
DYNAMIQ_WORKFLOW_VERSION = "dynamiq.workflow.version"
DYNAMIQ_FLOW_ID = "dynamiq.flow.id"
DYNAMIQ_FLOW_RUN_ID = "dynamiq.flow.run_id"
DYNAMIQ_CONFIG_RUN_ID = "dynamiq.config.run_id"
DYNAMIQ_FAILED_NODES = "dynamiq.failed_nodes"
DYNAMIQ_CHECKPOINT_ID = "dynamiq.checkpoint.id"
DYNAMIQ_RESUME_CHECKPOINT_ID = "dynamiq.resume.checkpoint_id"
DYNAMIQ_RESUME_ORIGINAL_RUN_ID = "dynamiq.resume.original_run_id"

# Dynamiq: input and output of any run
DYNAMIQ_INPUT = "dynamiq.input"
DYNAMIQ_OUTPUT = "dynamiq.output"

# Dynamiq: node spans
DYNAMIQ_NODE_ID = "dynamiq.node.id"
DYNAMIQ_NODE_NAME = "dynamiq.node.name"
DYNAMIQ_NODE_TYPE = "dynamiq.node.type"
DYNAMIQ_NODE_GROUP = "dynamiq.node.group"
DYNAMIQ_NODE_MODEL = "dynamiq.node.model"
DYNAMIQ_NODE_DEPENDS = "dynamiq.node.depends"
DYNAMIQ_RUN_DEPENDS = "dynamiq.run.depends"
DYNAMIQ_NODE_IS_MOCKED = "dynamiq.node.is_mocked"
DYNAMIQ_NODE_IS_OUTPUT_FROM_CACHE = "dynamiq.node.is_output_from_cache"
DYNAMIQ_NODE_CONFIG = "dynamiq.node.config"
DYNAMIQ_SKIP = "dynamiq.skip"
DYNAMIQ_HUMAN_FEEDBACK = "dynamiq.human_feedback"
DYNAMIQ_TOOL_DATA = "dynamiq.tool.data"
DYNAMIQ_LLM_IS_FALLBACK = "dynamiq.llm.is_fallback"

# Dynamiq: usage and cost
DYNAMIQ_USAGE_INPUT_TOKENS = "dynamiq.usage.input_tokens"
DYNAMIQ_USAGE_OUTPUT_TOKENS = "dynamiq.usage.output_tokens"
DYNAMIQ_USAGE_TOTAL_TOKENS = "dynamiq.usage.total_tokens"
DYNAMIQ_USAGE_INPUT_COST_USD = "dynamiq.usage.input_cost_usd"
DYNAMIQ_USAGE_OUTPUT_COST_USD = "dynamiq.usage.output_cost_usd"
DYNAMIQ_USAGE_TOTAL_COST_USD = "dynamiq.usage.total_cost_usd"

# Dynamiq: chat spans under prompt compaction, which is opt-in
DYNAMIQ_PROMPT_MESSAGES_BASE_SPAN_ID = "dynamiq.prompt.messages_base.span_id"
DYNAMIQ_PROMPT_MESSAGES_BASE_COUNT = "dynamiq.prompt.messages_base.count"

# Dynamiq: event attributes
DYNAMIQ_EXECUTION_ID = "dynamiq.execution.id"
DYNAMIQ_EXECUTION_ATTEMPT = "dynamiq.execution.attempt"
DYNAMIQ_EXECUTION_STATUS = "dynamiq.execution.status"
DYNAMIQ_EXECUTION_START_TIME_UNIX_NANO = "dynamiq.execution.start_time_unix_nano"
DYNAMIQ_HOOK_NAME = "dynamiq.hook.name"
DYNAMIQ_HOOK_TYPE = "dynamiq.hook.type"
DYNAMIQ_HOOK_POINT = "dynamiq.hook.point"
DYNAMIQ_HOOK_DECISION = "dynamiq.hook.decision"
DYNAMIQ_HOOK_DETAILS = "dynamiq.hook.details"

# Dynamiq: link and scope attributes
DYNAMIQ_LINK_TYPE = "dynamiq.link.type"
DYNAMIQ_TRACE_SHAPE_VERSION = "dynamiq.trace.shape_version"

# Event names
EVENT_EXCEPTION = "exception"
EVENT_NODE_EXECUTION = "dynamiq.node.execution"
EVENT_AGENT_HOOK = "dynamiq.agent.hook"

# Values of gen_ai.operation.name
OPERATION_INVOKE_WORKFLOW = "invoke_workflow"
OPERATION_INVOKE_AGENT = "invoke_agent"
OPERATION_CHAT = "chat"
OPERATION_EMBEDDINGS = "embeddings"
OPERATION_RETRIEVAL = "retrieval"
OPERATION_EXECUTE_TOOL = "execute_tool"

# Values of gen_ai.tool.type and gen_ai.output.type
TOOL_TYPE_FUNCTION = "function"
OUTPUT_TYPE_JSON = "json"

# Values of gen_ai.provider.name defined by the convention
PROVIDER_OPENAI = "openai"
PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_AWS_BEDROCK = "aws.bedrock"
PROVIDER_AZURE_AI_OPENAI = "azure.ai.openai"
PROVIDER_GCP_GEMINI = "gcp.gemini"
PROVIDER_GCP_VERTEX_AI = "gcp.vertex_ai"
PROVIDER_COHERE = "cohere"
PROVIDER_GROQ = "groq"
PROVIDER_MISTRAL_AI = "mistral_ai"
PROVIDER_DEEPSEEK = "deepseek"
PROVIDER_PERPLEXITY = "perplexity"
PROVIDER_X_AI = "x_ai"
PROVIDER_IBM_WATSONX_AI = "ibm.watsonx.ai"

# Values of finish_reason on output messages and gen_ai.response.finish_reasons
FINISH_REASON_STOP = "stop"
FINISH_REASON_LENGTH = "length"
FINISH_REASON_CONTENT_FILTER = "content_filter"
FINISH_REASON_TOOL_CALL = "tool_call"
FINISH_REASON_ERROR = "error"

# Message roles and part types of the input and output message schemas
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_TOOL = "tool"
PART_TEXT = "text"
PART_TOOL_CALL = "tool_call"
PART_TOOL_CALL_RESPONSE = "tool_call_response"
PART_URI = "uri"
PART_BLOB = "blob"
PART_FILE = "file"
MODALITY_IMAGE = "image"
MODALITY_VIDEO = "video"
MODALITY_AUDIO = "audio"
# Not in the convention's Modality enum, which also accepts any string.
MODALITY_DOCUMENT = "document"

# Other values
RUN_TYPE_NODE = "node"
LINK_TYPE_RESUMED_FROM = "resumed_from"
CANCELED_ERROR_TYPE = "CanceledException"
# The convention's fallback value of error.type when the error is not known.
ERROR_TYPE_OTHER = "_OTHER"
EXECUTION_STATUS_SUCCEEDED = "succeeded"
EXECUTION_STATUS_FAILED = "failed"

# Values of Node.group (dynamiq.nodes.types.NodeGroup). Kept as strings: importing dynamiq.nodes.types runs
# dynamiq/nodes/__init__.py, which imports dynamiq.nodes.node, which imports this package.
NODE_GROUP_LLMS = "llms"
NODE_GROUP_EMBEDDERS = "embedders"
NODE_GROUP_RETRIEVERS = "retrievers"
NODE_GROUP_TOOLS = "tools"
NODE_GROUP_AGENTS = "agents"
