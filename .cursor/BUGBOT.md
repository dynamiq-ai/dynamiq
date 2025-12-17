# Bugbot Rules for Dynamiq

## Project Overview

Dynamiq is an orchestration framework for agentic AI and LLM applications. It provides nodes, workflows, agents, and tools for building AI-powered systems.

**Tech Stack**: Python 3.10+, Pydantic 2.x, async/await patterns, LiteLLM, various vector stores.

---

## Critical Security Rules

### Flag dangerous code execution

If any changed file contains `eval(`, `exec(`, or `compile(` outside of the `dynamiq/nodes/tools/python.py` sandbox:
- Add a blocking Bug titled "Dangerous dynamic code execution"
- Body: "Usage of eval/exec/compile detected outside the designated sandbox. These functions pose security risks. Use RestrictedPython or the existing E2B sandbox instead."

### Flag hardcoded secrets

If any changed file contains actual hardcoded credential values (not parameter/field definitions):
- Add a blocking Bug titled "Potential hardcoded secret"
- Body: "Hardcoded credential detected. Use environment variables or the connection management system instead."

**What to flag** (actual secrets assigned directly):
```python
# BAD - hardcoded secret value
api_key = "sk-abc123..."
client = OpenAI(api_key="sk-abc123...")
headers = {"Authorization": "Bearer sk-abc123..."}
```

**What NOT to flag** (these are legitimate patterns):
```python
# OK - Connection class field definitions with env var defaults
api_key: str = Field(default_factory=partial(get_env_var, "OPENAI_API_KEY"))

# OK - Type annotations in class definitions
api_key: str
password: str | None = None

# OK - Node input parameters
def execute(self, input_data: dict) -> dict:
    api_key = input_data.get("api_key")

# OK - Accessing from connection/config
api_key = self.connection.api_key
api_key = config.api_key

# OK - Environment variable access
api_key = os.environ.get("API_KEY")
api_key = get_env_var("API_KEY")

# OK - Parameter names in function signatures
def connect(self, api_key: str) -> Client:
```

### Use input_schema for node input validation

If a PR adds or modifies nodes without defining an `input_schema` for input validation:
- Add a Bug titled "Consider adding input_schema"
- Body: "Nodes should define `input_schema` as a Pydantic model to validate input data. This provides:
  - Automatic validation in `validate_input_schema()`
  - Clear documentation of expected inputs
  - Type safety and IDE support
  - Better error messages for invalid inputs

  Example:
  ```python
  from pydantic import BaseModel, Field

  class MyNodeInputSchema(BaseModel):
      query: str = Field(..., description='Search query')
      max_results: int = Field(default=10, ge=1, le=100)
      filters: dict[str, Any] | None = None

  class MyNode(Node):
      input_schema: type[BaseModel] = MyNodeInputSchema

      def execute(self, input_data: MyNodeInputSchema, config: RunnableConfig = None, **kwargs):
          # input_data is already validated as MyNodeInputSchema
          query = input_data.query
          ...
  ```

  For nodes accepting user input, validation is especially critical for security."

---

## Pydantic & Data Modeling

### Use Pydantic models instead of dataclasses

If a PR introduces new classes using `@dataclass` decorator instead of Pydantic `BaseModel`:
- Add a blocking Bug titled "Use Pydantic BaseModel instead of dataclass"
- Body: "This project uses Pydantic 2.x for all data models. Do not use `@dataclass` from `dataclasses` module. Use `pydantic.BaseModel` instead. Benefits:
  - Automatic validation and coercion
  - JSON serialization with `model_dump()` and `model_dump_json()`
  - Schema generation with `model_json_schema()`
  - Field descriptions with `Field()`
  - Computed fields with `@computed_field`
  - Private attributes with `PrivateAttr`

  Example:
  ```python
  from pydantic import BaseModel, Field

  class MyData(BaseModel):
      name: str = Field(..., description='Resource name')
      count: int = Field(default=0, ge=0)
  ```"

### Prefer Pydantic BaseModel for new classes

If a PR introduces new classes that hold structured data without extending `BaseModel`:
- Add a Bug titled "Consider extending BaseModel"
- Body: "For classes holding structured data, prefer extending `pydantic.BaseModel`. This provides:
  - Automatic `__init__`, `__repr__`, `__eq__`
  - Immutability options with `model_config`
  - Easy serialization with `model_dump()`
  - Validation hooks with `@field_validator`, `@model_validator`

  Plain classes are acceptable for:
  - Utility/helper classes with only methods
  - Abstract base classes defining interfaces
  - Classes that genuinely need mutable state beyond Pydantic's capabilities"

### Prefer Pydantic models over dicts

If a PR introduces new structured data using plain `dict` instead of Pydantic models:
- Add a Bug titled "Consider using Pydantic model"
- Body: "Prefer Pydantic models over raw dicts for structured data. This provides validation, serialization, and better IDE support."

### Class structure organization

If new classes don't follow the standard organization:
- Add a Bug titled "Consider reorganizing class structure"
- Body: "Organize class members: 1) ClassVars/constants, 2) model_config, 3) public fields, 4) private attrs, 5) validators, 6) computed fields, 7) magic methods, 8) public methods, 9) private methods"

---

## Code Style & Naming

### Follow existing codebase patterns

If a PR adds new nodes, connections, or components that don't follow existing patterns:
- Add a blocking Bug titled "Follow existing codebase patterns"
- Body: "New code must follow existing naming conventions and architectural patterns:

  **Node naming**: Use descriptive names matching existing nodes (e.g., `OpenAI`, `Pinecone`, `ScaleSerp`)

  **File structure**: Follow `dynamiq/nodes/{category}/{provider}.py` pattern

  **Class structure**: Match existing node patterns:
  ```python
  class MyNode(ConnectionNode):  # or Node, VectorStoreNode
      group: NodeGroup = NodeGroup.{CATEGORY}
      name: str = 'Human Readable Name'
      description: str = 'Clear description of what node does'

      connection: MyConnection | None = None
      client: Any | None = None

      # Fields with Field() for descriptions
      some_feature_enabled: bool = Field(default=False, description='Enable some feature')

      def init_components(self, connection_manager: ConnectionManager | None = None):
          ...

      def execute(self, input_data: InputSchema, config: RunnableConfig = None, **kwargs):
          ...
  ```

  **Connection naming**: Match `{Provider}Connection` pattern in `dynamiq/connections/`

  Review similar existing implementations before adding new components."

### Use enums, constants, and class variables instead of hardcoded strings

If a PR uses hardcoded strings for statuses, types, or repeated values:
- Add a Bug titled "Use enums/constants instead of hardcoded strings"
- Body: "Avoid hardcoded strings. Use enums, class variables, and constants:
  ```python
  # Bad - hardcoded strings
  if status == 'success':
      format_type = 'json'

  # Good - use enums
  class OutputFormat(str, Enum):
      JSON = 'json'
      YAML = 'yaml'
      XML = 'xml'

  if status == RunnableStatus.SUCCESS:
      format_type = OutputFormat.JSON

  # Good - ClassVar for numeric constants
  DEFAULT_TIMEOUT: ClassVar[int] = 3600
  MAX_RETRIES: ClassVar[int] = 3
  ```

  Common enums in the project:
  - `RunnableStatus` for execution status
  - `NodeGroup` for node categorization
  - `MessageRole` for prompt messages
  - `InferenceMode` for agent modes"

### Require type annotations

If any new Python function or method lacks type annotations for parameters or return type:
- Add a Bug titled "Missing type annotations"
- Body: "All functions and methods must have complete type annotations. Example: `def process_data(input: str) -> dict[str, Any]:`"

### Line length enforcement

If any changed Python file contains lines exceeding 120 characters (excluding URLs and long strings):
- Add a Bug titled "Line exceeds 120 characters"
- Body: "Maximum line length is 120 characters as configured in setup.cfg. Break long lines appropriately."

### Use descriptive names

If a PR uses single-letter variables, unclear names, or incorrect field naming patterns:
- Add a Bug titled "Use descriptive names"
- Body: "Names should reveal intent. Apply these patterns:

  **Variables** - describe what they hold:
  ```python
  # Bad
  x = get_users()
  d = {}
  n = len(items)

  # Good
  active_users = get_active_users()
  user_scores_by_id = {}
  item_count = len(items)
  ```

  **Functions/methods** - use verb prefixes:
  - `get_*`, `fetch_*` - retrieve data
  - `create_*`, `build_*` - construct new objects
  - `is_*`, `has_*`, `can_*` - boolean checks
  - `process_*`, `handle_*` - transform data
  - `validate_*`, `check_*` - verification

  **Field naming** - use suffix patterns for booleans and states:
  - Boolean flags: `*_enabled`, `*_allowed`, `*_required` (not `enable_*`, `allow_*`)
  - State checks: `is_*`, `has_*`, `can_*`
  - Examples: `streaming_enabled`, `delegation_allowed`, `is_postponed_component_init`
  ```python
  # Bad
  enable_streaming: bool = False
  allow_delegation: bool = False

  # Good
  streaming_enabled: bool = False
  delegation_allowed: bool = False
  is_initialized: bool = False
  ```

  **Exceptions for short names:**
  - Loop indices: `i`, `j`, `k` (only for simple numeric loops)
  - Coordinates: `x`, `y`, `z`
  - Lambda parameters in comprehensions: `x` when context is clear
  - Mathematical formulas matching literature"

### Method naming prefixes

If new methods don't follow the naming convention with clear prefixes:
- Add a Bug titled "Consider clearer method naming"
- Body: "Use well-known prefixes: `get_*`, `set_*`, `add_*`, `delete_*`, `is_*`, `has_*`, `process_*`, `validate_*`, `create_*`, `run_*`, `execute_*`"

### Private member naming

If internal methods/attributes don't use underscore prefix:
- Add a Bug titled "Use underscore for private members"
- Body: "Private methods and attributes should use single underscore prefix: `_internal_method`, `_private_attr`"

### Self-documenting code over comments

If a PR adds obvious comments that restate what code does instead of using clear naming:
- Add a Bug titled "Prefer self-documenting code"
- Body: "Code should be self-documenting through proper naming. Avoid comments that state the obvious:

  **Bad - obvious comments:**
  ```python
  # Get user by id
  def get(id):
      # Check if user exists
      if id in users:
          # Return the user
          return users[id]
      # Return None if not found
      return None
  ```

  **Good - self-documenting:**
  ```python
  def get_user_by_id(user_id: str) -> User | None:
      return self._users.get(user_id)
  ```

  **When comments ARE valuable:**
  - Explaining *why* (business logic, non-obvious decisions)
  - Documenting complex algorithms
  - Warning about edge cases or gotchas
  - TODOs with ticket references

  ```python
  # Use exponential backoff to avoid overwhelming the API during rate limits
  wait_time = base_delay * (backoff_rate ** attempt)
  ```"

### Docstrings for public APIs

If new public classes or methods in `dynamiq/**` lack docstrings:
- Add a Bug titled "Missing docstring"
- Body: "Public APIs should have Google-style docstrings with Args, Returns, and Raises sections where applicable."

### Conventional commits

If commit messages don't follow conventional commit format:
- Add a Bug titled "Use conventional commit format"
- Body: "Use conventional commits: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`"

---

## Serialization Rules

### Properly implement to_dict() for tracing and callbacks

If a PR adds or modifies classes (especially nodes, connections, prompts) without proper `to_dict()` implementation:
- Add a blocking Bug titled "Missing or improper to_dict() implementation"
- Body: "Classes used in workflows must implement `to_dict()` properly. This method is used by:
  - **Tracing callbacks** (`TracingCallbackHandler`) for observability
  - **Streaming callbacks** for real-time updates
  - **Node callbacks** (`on_node_start`, `on_node_end`, `on_node_error`, etc.)
  - **YAML serialization** for workflow persistence

  Key requirements:

  **1. Support `for_tracing` parameter:**
  ```python
  def to_dict(self, for_tracing: bool = False, **kwargs) -> dict:
      if for_tracing:
          # Return minimal data for tracing (exclude verbose fields)
          return {'id': self.id, 'name': self.name, 'type': self.type}
      return self.model_dump(**kwargs)
  ```

  **2. Handle secure information:**
  - Use `include_secure_params` flag to control sensitive data exposure
  - Define `to_dict_exclude_params` and `to_dict_exclude_secure_params` properties
  - Never expose API keys, tokens, or secrets in tracing output
  ```python
  @property
  def to_dict_exclude_secure_params(self):
      return self.to_dict_exclude_params | {'connection': True, 'api_key': True}
  ```

  **3. Format complex values:**
  - Use `format_value()` from `dynamiq.utils` for callables and complex objects
  - Nested objects should call their own `to_dict(for_tracing=for_tracing)`

  **4. Filter None values for tracing:**
  ```python
  if for_tracing:
      data = {k: v for k, v in data.items() if v is not None}
  ```

  See `Node.to_dict()` in `dynamiq/nodes/node.py` for reference implementation."

### Handle nested nodes in to_dict()

If a PR adds nodes with other nodes as parameters (e.g., agents with LLMs, tools) without proper nested serialization:
- Add a blocking Bug titled "Improper nested node serialization in to_dict()"
- Body: "When a node contains other nodes as parameters (e.g., `llm`, `tools`, `agent`), `to_dict()` must properly serialize them. Requirements:

  **1. Call nested node's to_dict() with same parameters:**
  ```python
  def to_dict(self, include_secure_params: bool = False, for_tracing: bool = False, **kwargs) -> dict:
      data = super().to_dict(include_secure_params=include_secure_params, for_tracing=for_tracing, **kwargs)

      # Serialize nested node
      if self.llm:
          data['llm'] = self.llm.to_dict(
              include_secure_params=include_secure_params,
              for_tracing=for_tracing,
              **kwargs
          )

      # Serialize list of nested nodes
      if self.tools:
          data['tools'] = [
              tool.to_dict(include_secure_params=include_secure_params, for_tracing=for_tracing, **kwargs)
              for tool in self.tools
          ]

      return data
  ```

  **2. For tracing, use minimal representation:**
  ```python
  if for_tracing and self.llm:
      data['llm'] = {'id': self.llm.id, 'name': self.llm.name, 'type': self.llm.type}
  ```

  **3. Exclude nested nodes from model_dump():**
  Add nested node fields to `to_dict_exclude_params` and handle them manually:
  ```python
  @property
  def to_dict_exclude_params(self):
      return super().to_dict_exclude_params | {'llm': True, 'tools': True}
  ```

  **4. Handle circular references:**
  For nodes that might reference each other, use `for_tracing=True` to break cycles:
  ```python
  if self.parent_node and not for_tracing:
      data['parent_node'] = self.parent_node.to_dict(for_tracing=True)  # Minimal to avoid cycles
  ```

  See `ReActAgent` or other agent implementations for reference patterns."

### Exclude non-serializable fields from to_dict()

If a PR adds fields like `client`, `vector_store`, or other runtime objects without excluding them from serialization:
- Add a Bug titled "Non-serializable field in to_dict()"
- Body: "Runtime objects (clients, stores, executors) must be excluded from `to_dict()`. Add them to `to_dict_exclude_params`:
  ```python
  @property
  def to_dict_exclude_params(self):
      return {
          'client': True,
          'vector_store': True,
          'executor': True,
          # ... other runtime objects
      }
  ```"

### Validate YAML loading and dumping for new nodes

If the PR adds new node classes in `dynamiq/nodes/**` or adds new parameters to existing nodes:
- Add a blocking Bug titled "Verify YAML serialization compatibility"
- Body: "New nodes and node parameter changes must be tested for YAML serialization roundtrip. Ensure the node can be:
  1. Serialized via `WorkflowYAMLDumper` (`dynamiq/serializers/dumpers/yaml.py`)
  2. Deserialized via `WorkflowYAMLLoader` (`dynamiq/serializers/loaders/yaml.py`)
  3. Tested with roundtrip: `Workflow.from_yaml_file()` → `wf.to_yaml_file()` → `Workflow.from_yaml_file()` with `init_components=True`

  Example test pattern (see `examples/components/core/dag/dag_map.py`):
  ```python
  wf = Workflow.from_yaml_file(file_path=yaml_path, connection_manager=cm, init_components=True)
  wf.to_yaml_file('output.yaml')
  wf_reloaded = Workflow.from_yaml_file(file_path='output.yaml', connection_manager=cm, init_components=True)
  ```"

### Node parameters must be serializable

If new node fields use non-serializable types (callables, lambdas, complex objects without `to_dict`):
- Add a blocking Bug titled "Non-serializable node parameter"
- Body: "Node parameters must be YAML-serializable. For complex types:
  - Use Pydantic models with proper `model_dump()` support
  - Implement `to_dict()` method for custom classes
  - Use `PrivateAttr` for runtime-only state that shouldn't be serialized
  - Register the type with `NodeManager` if it's a node reference"

### Connection references in YAML

If a PR adds nodes with connections but doesn't handle the connection ID reference pattern:
- Add a Bug titled "Connection serialization pattern"
- Body: "Connections in YAML should be referenced by ID, not embedded. The `WorkflowYAMLDumper` extracts connections to a separate `connections:` section and nodes reference them by ID. Ensure your node's connection field serializes to just the connection ID string."

### Nested node serialization in YAML

If a PR adds nodes that contain other nodes as parameters (e.g., agents with LLMs, tools):
- Add a Bug titled "Verify nested node serialization"
- Body: "Nodes containing other nodes must serialize/deserialize correctly. The `get_updated_node_init_data_with_initialized_nodes` method handles recursive initialization. Verify:
  - Nested nodes have proper `type` field with full dotted path (e.g., `dynamiq.nodes.llms.OpenAI`)
  - Nested nodes are registered in `NodeManager`
  - The roundtrip test passes with all nested components initialized"

### Prompt serialization

If a PR modifies prompt handling in nodes:
- Add a Bug titled "Verify prompt serialization"
- Body: "Prompts can be serialized either:
  1. As a reference ID to a prompt in the `prompts:` section
  2. As an inline prompt object with full data
  Ensure your changes support both patterns in `get_node_prompt()` and maintain backward compatibility."

---

## Connection Management

### New connection nodes must extend ConnectionNode

If a PR adds a new node that uses external connections/clients but doesn't extend `ConnectionNode`:
- Add a blocking Bug titled "Use ConnectionNode base class"
- Body: "Nodes requiring external connections must extend `ConnectionNode` from `dynamiq/nodes/node.py`. This provides:
  - Automatic client initialization via `ConnectionManager`
  - Connection reuse across nodes
  - Automatic reconnection via `ensure_client()`
  - Proper cleanup and lifecycle management

  Example pattern:
  ```python
  class MyServiceNode(ConnectionNode):
      connection: MyServiceConnection | None = None
      client: Any | None = None

      def init_components(self, connection_manager: ConnectionManager | None = None):
          connection_manager = connection_manager or ConnectionManager()
          self._connection_manager = connection_manager
          super().init_components(connection_manager)
          # Client is auto-initialized from connection
  ```"

### Connection classes must implement connect method

If a PR adds a new connection class in `dynamiq/connections/` without a `connect()` method:
- Add a blocking Bug titled "Connection missing connect() method"
- Body: "Connection classes must implement `connect() -> Any` method that returns the client instance. The `ConnectionManager` calls this method via `CONNECTION_METHOD_BY_INIT_TYPE[ConnectionClientInitType.DEFAULT]`. See `dynamiq/connections/managers.py`."

### ConnectionNode subclasses must call super().init_components()

If a PR overrides `init_components()` in a `ConnectionNode` subclass without calling `super().init_components()`:
- Add a blocking Bug titled "Missing super().init_components() call"
- Body: "When overriding `init_components()` in `ConnectionNode` subclasses, always call `super().init_components(connection_manager)` to ensure proper client initialization and `_connection_manager` assignment. This enables connection reuse and automatic reconnection."

### Implement is_client_closed() for custom clients

If a PR adds a node with a client that has non-standard closed state detection:
- Add a Bug titled "Consider overriding is_client_closed()"
- Body: "If your client doesn't use standard `closed`, `is_closed()`, or `_closed` attributes, override `is_client_closed()` to enable automatic reconnection. The base `ConnectionNode.ensure_client()` uses this to detect stale connections."

### VectorStoreNode pattern for vector stores

If a PR adds a node that connects to a vector store but doesn't extend `VectorStoreNode`:
- Add a Bug titled "Use VectorStoreNode for vector store nodes"
- Body: "Nodes connecting to vector stores should extend `VectorStoreNode` from `dynamiq/nodes/node.py`. This provides:
  - Automatic `vector_store` and `client` initialization
  - Proper `vector_store_params` handling with `BaseVectorStoreParams`
  - Reconnection support for both client and vector store
  - Required `vector_store_cls` property for type specification"

### Don't create clients directly in execute()

If a PR creates connection clients directly inside `execute()` method instead of using `init_components()`:
- Add a blocking Bug titled "Client creation in execute() bypasses ConnectionManager"
- Body: "Don't create clients in `execute()`. Use `init_components()` to initialize clients via `ConnectionManager`. This ensures:
  - Connection reuse across multiple executions
  - Proper connection pooling
  - Automatic reconnection on failures
  - Thread-safe client management"

### Store _connection_manager reference

If a PR adds a `ConnectionNode` subclass that doesn't store the `connection_manager` reference:
- Add a Bug titled "Store ConnectionManager reference"
- Body: "Store the `connection_manager` in `self._connection_manager` during `init_components()`. This enables `ensure_client()` to reinitialize closed connections:
  ```python
  def init_components(self, connection_manager: ConnectionManager | None = None):
      connection_manager = connection_manager or ConnectionManager()
      self._connection_manager = connection_manager
      super().init_components(connection_manager)
  ```"

### Callback integration

If new nodes don't integrate with the callback system for tracing:
- Add a Bug titled "Consider callback integration"
- Body: "Nodes should integrate with `dynamiq/callbacks/` for tracing and streaming support."

---

## Async & Performance

### Async patterns

If a PR adds a class with async `__init__` logic or blocking I/O in async methods:
- Add a blocking Bug titled "Improper async pattern"
- Body: "Use `async def startup()` and `async def shutdown()` methods for async initialization/cleanup. Avoid blocking I/O in async methods; use `asyncio.to_thread()` if needed."

### Avoid blocking in async context

If async functions contain synchronous blocking calls (file I/O, `time.sleep`, `requests.*`):
- Add a blocking Bug titled "Blocking call in async context"
- Body: "Use async alternatives: `aiofiles` for file I/O, `asyncio.sleep()` instead of `time.sleep()`, `httpx` or `aiohttp` instead of `requests`."

### Large file processing

If a PR processes files without streaming or chunking for potentially large files:
- Add a Bug titled "Consider streaming for large files"
- Body: "Process large files using streaming/chunking to avoid memory issues. Use generators or async iterators."

### Use Pythonic idioms for performance

If a PR uses inefficient patterns where Python idioms would be faster and more readable:
- Add a Bug titled "Consider Pythonic optimization"
- Body: "Use Python's built-in optimizations for better performance and readability:

  **List/dict/set comprehensions** (faster than loops with append):
  ```python
  # Bad
  result = []
  for item in items:
      if item.is_valid:
          result.append(item.value)

  # Good
  result = [item.value for item in items if item.is_valid]
  ```

  **Generator expressions** (memory efficient for large data):
  ```python
  # Bad - creates full list in memory
  sum([x * x for x in range(1000000)])

  # Good - lazy evaluation
  sum(x * x for x in range(1000000))
  ```

  **Dictionary operations**:
  ```python
  # Bad
  result = {}
  for k, v in items:
      result[k] = v

  # Good
  result = {k: v for k, v in items}
  # Or: result = dict(items)
  ```

  **Use built-ins**: `any()`, `all()`, `map()`, `filter()`, `zip()`, `enumerate()`
  ```python
  # Bad
  found = False
  for item in items:
      if item.matches(query):
          found = True
          break

  # Good
  found = any(item.matches(query) for item in items)
  ```"

### Avoid unnecessary intermediate collections

If a PR creates unnecessary lists or dicts that could be avoided:
- Add a Bug titled "Avoid unnecessary intermediate collections"
- Body: "Reduce memory usage by avoiding unnecessary intermediate collections:

  ```python
  # Bad - creates intermediate list
  filtered = [x for x in items if x.valid]
  result = [process(x) for x in filtered]

  # Good - single pass
  result = [process(x) for x in items if x.valid]

  # Better for large data - generator chain
  result = list(process(x) for x in items if x.valid)
  ```

  Use `itertools` for complex iterations:
  - `itertools.chain()` instead of concatenating lists
  - `itertools.islice()` instead of slicing large lists
  - `itertools.groupby()` for grouping operations"

---

## Error Handling

### Use project exceptions

If a PR introduces generic `Exception` raises instead of project-specific exceptions:
- Add a Bug titled "Use project-specific exceptions"
- Body: "Use exceptions from `dynamiq/nodes/exceptions.py` (NodeException, NodeFailedException, etc.) for better error categorization and handling."

### Retry logic with tenacity

If a PR adds custom retry logic instead of using tenacity:
- Add a Bug titled "Consider using tenacity for retries"
- Body: "The project uses `tenacity` for retry logic. Use decorators like `@retry` with exponential backoff instead of custom retry loops."

---

## Testing

### Require tests for new nodes

If the PR adds new files in `dynamiq/nodes/**` and there are no corresponding changes in `tests/**`:
- Add a Bug titled "Missing tests for new node"
- Body: "New node implementations should include unit tests. Add tests in `tests/unit/` or `tests/integration/`."

### Require tests for bug fixes

If the PR title or description contains "fix" or "bug" and there are no test changes:
- Add a Bug titled "Consider adding regression test"
- Body: "Bug fixes should include a regression test to prevent the issue from recurring."

---

## Dependencies

### New dependencies require justification

If the PR modifies `pyproject.toml` to add new dependencies:
- Add a Bug titled "New dependency added"
- Body: "Please justify the new dependency in the PR description. Consider: Is it actively maintained? Does it have acceptable licensing? Can we achieve this with existing dependencies?"

### Pin dependency versions

If new dependencies are added without version constraints:
- Add a Bug titled "Unversioned dependency"
- Body: "Dependencies should have version constraints (e.g., `~1.2.0` or `^1.2.0`) to ensure reproducible builds."
