# Pre-PR Review

Review code before creating a PR to catch issues that BUGBOT would flag.

## How to Use

Analyze changed/new files against the checklist below. Report issues with specific line references and suggest fixes.

---

## BLOCKING Issues (Must Fix)

### Security

- No `eval()`, `exec()`, `compile()` outside `dynamiq/nodes/tools/python.py`
- No hardcoded secrets (API keys, passwords, tokens as string literals)

```python
# BAD
api_key = "sk-abc123..."

# GOOD
api_key: str = Field(default_factory=partial(get_env_var, "API_KEY"))
```

### Pydantic

- No `@dataclass` - Use `pydantic.BaseModel` instead
- Structured data uses BaseModel, not plain dicts

### Node Patterns

- External service nodes extend `ConnectionNode`
- `super().init_components()` called when overriding
- `_connection_manager` stored in `init_components()`
- No client creation in `execute()` - use `init_components()`

### Serialization

- `to_dict()` supports `for_tracing` and `include_secure_params`
- Nested nodes call `to_dict()` with same params
- Non-serializable fields (`client`, `vector_store`, `executor`) in `to_dict_exclude_params`

### Async

- No blocking calls in async methods (`time.sleep()`, `requests.*`, sync file I/O)

---

## Should Fix

### Code Style

- Type annotations on all functions/methods
- Lines 120 characters or less
- Descriptive variable names (no single-letter except `i`, `j`, `k` in loops)
- Boolean fields use `*_enabled`, `*_allowed` (not `enable_*`, `allow_*`)
- Method prefixes: `get_*`, `is_*`, `has_*`, `create_*`, `validate_*`
- Private members use `_prefix`
- Enums over strings (`RunnableStatus.SUCCESS` not `"success"`)

### Node Development

- `input_schema` defined (Pydantic model for validation)
- `NodeGroup` set for categorization
- `name` and `description` are human-readable
- `Field()` with descriptions for public fields

### Documentation

- Docstrings on public APIs (Google-style)
- No obvious comments - code should be self-documenting

### Testing

- Tests for new nodes in `tests/unit/` or `tests/integration/`
- Regression test for bug fixes

---

## Quick Checks by File Type

### New Node (dynamiq/nodes/**/*.py)

```python
class MyNode(ConnectionNode):  # Correct base class
    group: NodeGroup = NodeGroup.TOOLS
    name: str = "My Tool"
    description: str = "Does something useful"
    
    connection: MyConnection | None = None
    client: Any | None = None
    input_schema: ClassVar[type[MyInputSchema]] = MyInputSchema
    
    @property
    def to_dict_exclude_params(self):
        return super().to_dict_exclude_params | {"client": True}
    
    def init_components(self, connection_manager: ConnectionManager | None = None):
        connection_manager = connection_manager or ConnectionManager()
        self._connection_manager = connection_manager  # Store reference
        super().init_components(connection_manager)    # Call super
    
    def execute(self, input_data: MyInputSchema, config: RunnableConfig = None, **kwargs) -> dict:
        ...
```

### New Connection (dynamiq/connections/*.py)

```python
class MyConnection(BaseApiKeyConnection):
    api_key: str = Field(default_factory=partial(get_env_var, "MY_API_KEY"))
    
    def connect(self) -> MyClient:
        return MyClient(api_key=self.api_key)
    
    @property
    def conn_params(self) -> dict:
        return {"api_key": self.api_key}
```

### Agent/Tool Changes

- Tool `input_schema` with required fields and descriptions
- Tool `description` clear enough for LLM to understand
- `is_optimized_for_agents` set if output needs string conversion

---

## Commit Message Format

```
feat: add new capability
fix: resolve issue with X
refactor: improve Y implementation
docs: update Z documentation
test: add tests for W
chore: update dependencies
```
