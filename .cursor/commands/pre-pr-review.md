Review the changed files before creating a PR. Check against the project's BUGBOT rules and report any issues that would be flagged.

**Check for blocking issues first:**

1. **Security**: No `eval()`/`exec()`/`compile()` outside the Python tool sandbox. No hardcoded secrets - use `Field(default_factory=partial(get_env_var, "KEY"))`.

2. **Pydantic**: No `@dataclass` - use `BaseModel`. All structured data should use Pydantic models.

3. **Node patterns**: 
   - External service nodes must extend `ConnectionNode`
   - `init_components()` must call `super().init_components()` and store `_connection_manager`
   - Never create clients in `execute()` - use `init_components()`

4. **Serialization**: 
   - `to_dict()` must support `for_tracing` and `include_secure_params` parameters
   - Nested nodes must call `to_dict()` with the same parameters
   - Add `client`, `vector_store`, `executor` to `to_dict_exclude_params`

5. **Async**: No blocking calls (`time.sleep()`, `requests.*`, sync file I/O) in async methods.

**Then check for style issues:**

6. **Types**: All functions need type annotations for parameters and return values.

7. **Naming**: 
   - Boolean fields use `*_enabled`/`*_allowed` suffix, not `enable_*`/`allow_*` prefix
   - Methods use clear prefixes: `get_*`, `is_*`, `has_*`, `create_*`, `validate_*`
   - No single-letter variables except loop indices

8. **Node fields**: Nodes need `input_schema`, `NodeGroup`, `name`, `description`, and `Field()` with descriptions.

9. **Tests**: New nodes need tests. Bug fixes need regression tests.

For each issue, provide the file, line number, what's wrong, and how to fix it.

End with a summary: how many blocking issues vs style issues found.
