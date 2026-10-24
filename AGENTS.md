# Dynamiq

Dynamiq is a Python framework for agentic AI and LLM applications: nodes, workflows, agents and tools, serialized to
YAML and run by the Dynamiq platform. It supports Python 3.10 and later, uses Pydantic 2, and is managed with uv.

## Set up and check

- `make install` installs every extra and the pre-commit hooks.
- `uv run pytest tests --ignore=tests/integration_with_creds -m "not smoke" -n auto --dist worksteal` runs the suite
  as CI does.
- `make prepare` runs the pre-commit hooks on every file: darker with isort, flake8 at 120 columns, pyupgrade for
  Python 3.10 and bandit.
- The default install, without extras, has a CI job of its own: `uv sync --frozen`, then `uv run pytest
  tests/unit/utils/test_lazy_imports.py tests/unit/test_lazy_loading.py tests/unit/nodes/tools/test_file_tools.py`.
- `make build-mkdocs` builds the API docs, which CI builds too.

`tests/unit` and `tests/integration` run without credentials. Mock LLMs and embedders with the fixtures in
`tests/conftest.py` (`mock_llm_executor`, `mock_embedding_executor`); every test gets a mocked tracing client.
`tests/helpers/agent_hooks.py` scripts agent runs across inference modes. `tests/integration_with_creds` calls real
providers and needs their keys. Smoke tests are slow and paid, and run only through `make test-smoke` or the
`run-smoke-tests` pull request label.

## Rules for changes

### Imports and dependencies

- Import at the top of the module. A local import is only for breaking a circular import or for deferring a slow SDK
  on purpose, with a comment saying which.
- Use litellm through `dynamiq.utils.litellm` (`get_litellm()` and its wrappers), never `import litellm` at module
  level: it takes seconds to import. Register changes to litellm's global state with `on_litellm_import()`.
- A package `__init__.py` that defines `_LAZY_EXPORTS` imports nothing eagerly. Add a new export to `_LAZY_EXPORTS`,
  to its `if TYPE_CHECKING:` block and to `__all__`.
- An SDK that one integration uses goes in an extra in `pyproject.toml` and in `OPTIONAL_DEPENDENCIES` in
  `dynamiq/utils/lazy_imports.py`, so that a missing package names its extra. Import it only from that integration's
  modules.
- Give every dependency a range below its next major version, run `uv lock`, and say in the pull request why a new
  one is needed.

### Nodes and data

- Model data with Pydantic `BaseModel`, not dataclasses or loose dicts. Mutable defaults use
  `Field(default_factory=...)`, and runtime-only state uses `PrivateAttr`.
- Validate node input with an `input_schema` model. An agent sees only a tool's `description` and each field's
  `description`, so say what to pass and when.
- Follow the existing layout and names: nodes in `dynamiq/nodes/<group>/<provider>.py`, connection classes named after
  the provider in `dynamiq/connections/connections.py`, data types in `types.py`, managers in `managers.py`. Boolean
  fields end in `_enabled`, `_allowed` or `_required`.
- Use enums and named constants (`RunnableStatus`, `NodeGroup`, `InferenceMode`) instead of string literals and magic
  numbers.
- Log with `dynamiq.utils.logger`, never `print`. Comments say why, not what; docstrings and descriptions carry no
  narration or promotional wording.
- Fail loudly: raise an exception from `dynamiq/nodes/exceptions.py` rather than return `None` or `""` for a failure,
  and never default a credential to `""`.

### Serialization and compatibility

- The UI, catalyst and the runtime load workflows stored as YAML. A new node or field must survive loading, dumping
  and reloading with `init_components=True`.
- New fields are optional with a default. Renaming or removing a field needs the old name to keep working, and a
  breaking change is called out in the pull request.
- `to_dict()` leaves out runtime objects such as clients and stores through `to_dict_exclude_params`, never puts
  secrets in tracing output, and serializes nested nodes with their own `to_dict()` and the same arguments.

### Connections

- A node with an external client extends `ConnectionNode`, or `VectorStoreNode` for a vector store, and gets the
  client in `init_components()` through `ConnectionManager`, calling `super().init_components()`. Never create a client
  in `execute()`.
- `ConnectionManager` shares clients across runs and tenants: close only a client the code created, and keep
  connection ids stable.

### Checkpoints and resume

- A `to_checkpoint_state()` override calls `super()` and merges its fields. `reset_run_state()` keeps state when
  `is_resumed` is set. An iteration counts as done only after its work succeeds.
- Use `utc_now()` for timestamps, `Prompt.serialize_messages()` for messages, and atomic writes
  (`tempfile.mkstemp()`, then `os.replace()`) in the file system backend.

### Concurrency and tenants

- One node instance serves many runs and is cloned for Map items and parallel tool calls. Reset per-run state in
  `reset_run_state()`, also when a run fails.
- Module-level state such as caches, registries and stores is shared by every run and tenant in a server. Key it by
  what the server vouches for (`RunnableConfig.trusted_context`), never only by ids the client sends; guard it with a
  lock; and let a registration happen twice, since a module runs again after a failed import.
- Session state on a shared database connection, such as a Postgres `SET`, needs the same lock as the query that
  depends on it.
- Async code makes no blocking calls (`time.sleep`, `requests`, file I/O): use async clients or `asyncio.to_thread()`.

### Security

- `eval`, `exec` and `compile` run only in `dynamiq/nodes/tools/python.py`, under RestrictedPython.
- No credentials in code: connections read them from their fields or from environment variables.
- A type string from a request names code to import (`import_type`): resolve only types the caller may use.

## Finding bugs

Start from a real caller, such as the YAML loader, the runtime or an agent's tool loop, and follow the data into the
change. Most bugs found in this codebase were of these kinds:

- An integration that fails in the default install, or a lazy package that fails with `DYNAMIQ_EAGER_IMPORTS=1` or
  when several threads first import it at once.
- State that leaks between runs or tenants: process-wide stores keyed by ids the client chooses, per-run state that is
  not reset, a shared client that one run closes or reconfigures.
- Resume: a paused run's checkpoint that is lost, overwritten or never cleaned up, or state reset when a run resumes.
- Serialization: a YAML round trip that drops or renames a field, or tracing output that includes a secret.
- A third-party SDK that behaves differently at the version in `uv.lock`: check the installed version, not the one
  you remember.

Prove each bug with a failing test before fixing it, and keep the test as its regression test. Test behavior through
public APIs, and skip tests that restate the implementation or only check that a mock was called.

## Commits and pull requests

Commit messages and pull request titles follow Conventional Commits (`feat:`, `fix:`, `perf:`, `refactor:`, `docs:`,
`test:`, `chore:`), which a pre-commit hook and a CI check enforce. Mark a breaking change with `!` and a
`BREAKING CHANGE:` footer.
