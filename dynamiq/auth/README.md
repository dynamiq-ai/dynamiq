# Tool Authentication

This document summarizes the current authentication capabilities in Dynamiq, how agents surface missing credentials, and how to resume execution once credentials are supplied.

## Architecture Overview

```
User / Client  <–– dynamiq_request_credential event ––  Agent  –– ToolExecutionContext –– Tool
                                 ^                                    |
                                 |                                    v
                       AuthRequestLoggingCallback   CredentialManager (cache)
```

### Key Components

- **Auth models** (`dynamiq/auth/models.py`)
  - `AuthScheme`, `AuthCredential`, `AuthConfig`: Describe how an external API expects credentials (API key, OAuth2, etc.) and the data available.
  - `AuthRequest`: Structured payload emitted whenever a tool cannot proceed without credentials.

- **Tool execution context** (`dynamiq/nodes/tools/context.py`)
  - Every tool receives a `ToolExecutionContext` containing any pre-supplied auth payload and a `request_auth()` helper. Calling `request_auth()` raises a `ToolAuthRequiredException`, emits a stream/callback event, and pauses execution.

- **Agent runtime** (`dynamiq/nodes/agents/base.py`)
  - Merges auth from `tool_auth`, cached credentials, and tool-level overrides.
  - Emits `dynamiq_request_credential` stream events and `on_node_auth_request` callbacks when tools cannot continue.
  - Returns `status="auth_required"` and `auth_requests=[...]` in the node output, allowing higher layers to prompt the user.

- **Credential manager** (`dynamiq/auth/manager.py`)
  - Current prototype is in-memory. It stores merged payloads (headers, query params, etc.) per tool. When credentials return, they are cached for subsequent runs within the same process.

- **Callbacks** (`dynamiq/callbacks/auth.py`)
  - `AuthRequestLoggingCallback` is a ready-to-use observer that prints auth requests. You can implement your own callback to forward the payload to UIs or other services.

## Runtime Flow

1. Agent runs normally until a tool needs credentials.
2. Tool calls `tool_context.request_auth(auth_config)`.
   - Agent raises `ToolAuthRequiredException`.
   - Streaming clients receive an event with `event: "dynamiq_request_credential"`.
   - Callbacks receive the `AuthRequest` via `on_node_auth_request`.
   - Agent output contains `status: "auth_required"` and serialized `auth_requests`.
3. Client prompts the user and collects credentials (API key, OAuth redirect URL, etc.).
4. Client re-runs the workflow/agent, injecting credentials via `tool_auth` or a stored credential cache.
5. Tool picks up the credentials from the execution context and continues.

See `examples/components/auth/agent_with_exa_auth.py` for a runnable demonstration. The example performs two runs: first to trigger the auth handshake, second with credentials supplied.

## Configuring Credentials

### Passing Credentials via `tool_auth`

Agents accept an optional `tool_auth` field that mirrors `tool_params`. You can target:

```python
{
    "global": AuthConfig | dict,
    "by_name": {"Tool Name": AuthConfig | dict},
    "by_id": {"tool-id": AuthConfig | dict}
}
```

- When using `AuthConfig`, the agent automatically converts it into header/query payloads via `to_tool_payload()`.
- Raw dicts can also be merged, but `AuthConfig` is recommended for consistency.

### Caching Credentials

- The in-memory credential manager merges any `AuthConfig` payloads returned from the client with cached state.
- Credentials survive until process restart; extend or replace `InMemoryCredentialManager` to persist secrets elsewhere (e.g., Redis, Vault).

## Streaming & Callback Integration

- **Streaming**: When auth is required, the agent emits `{"type": "auth_request", "event": "dynamiq_request_credential", ...}`. Clients listening on the streaming channel can pause UI execution and display an auth prompt.
- **Callbacks**: Implement `on_node_auth_request()` to capture requests during workflow execution. `AuthRequestLoggingCallback` is provided for quick debugging.

## Current Limitations & Roadmap

- The workflow does not yet automatically resume the original tool invocation once credentials arrive; re-running the agent/workflow with new credentials is required.
- Credentials are cached only in memory. Production deployments should extend the manager to encrypt and persist secrets securely.
- OAuth/OIDC flows currently require manual handling on the client. Mapping `auth_requests` onto a full OAuth redirect/resume flow is planned (see `docs/authentication-roadmap.md`).

## Quick Start Checklist

1. Add `tool_auth` to your agent input with known API keys or tokens.
2. Attach `AuthRequestLoggingCallback` (or your custom handler) to monitor auth requests.
3. Listen for `status="auth_required"` in agent outputs and collect credentials from the user.
4. Re-run the agent/workflow with updated `tool_auth` or pre-populate the credential manager.
5. For OAuth flows, use `auth_requests` metadata to guide the user to the provider, then send the resulting callback URL back in your follow-up call.

For more depth, review:
- `examples/components/auth/agent_with_exa_auth.py` for a practical demo.
- `dynamiq/auth/models.py` to understand how tokens are represented.
