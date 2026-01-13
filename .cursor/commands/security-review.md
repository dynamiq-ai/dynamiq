# Security Review

Analyze the code for security vulnerabilities specific to the Dynamiq AI orchestration framework.

## What to Analyze

Review the code against these security areas:

### 1. Credential Management (CRITICAL)

- Hardcoded API keys, tokens, or passwords in source code
- Credentials logged via `logger.info/debug/error` statements
- Secrets included in `to_dict()` without `include_secure_params=False`
- Connection objects serialized without credential masking

**Secure pattern:**
```python
api_key: str = Field(default_factory=partial(get_env_var, "API_KEY"))
```

### 2. Dynamic Code Execution (CRITICAL)

- Usage of `eval()`, `exec()`, `compile()` outside `dynamiq/nodes/tools/python.py`
- Import statements bypassing `ALLOWED_MODULES` whitelist
- Attribute access bypassing `DynamiqRestrictingNodeTransformer`
- `__builtins__` modification attempts

**Sandbox boundaries:**
- All code execution must go through `compile_restricted()`
- Blocked modules: os, subprocess, sys, socket, pickle, marshal

### 3. Input Validation (HIGH)

- Nodes without `input_schema` Pydantic model
- String interpolation in Cypher/SQL queries without parameterization
- User input directly embedded in LLM prompts
- File paths from user input without sanitization
- JSONPath expressions from untrusted sources

**Injection vectors to check:**
- `dynamiq/nodes/tools/cypher_executor.py` - Cypher injection
- Agent system prompts - Prompt injection
- `WorkflowYAMLLoader` - YAML injection

### 4. Serialization Security (HIGH)

- `pickle.loads()` or `jsonpickle.decode()` on untrusted data
- YAML deserialization without safe_load equivalent
- Node type resolution from untrusted YAML type strings
- `NodeManager` registry allowing arbitrary class instantiation

### 5. Authentication (HIGH)

- Missing authentication on HTTP endpoints
- Bearer tokens in query parameters instead of headers
- Missing TLS/SSL verification (`verify_certs=False`)
- Connection strings with embedded credentials

**Connection security checks:**
- Elasticsearch: `verify_certs` and `ca_path`
- PostgreSQL: Password in connection string
- AWS: IAM role vs hardcoded access keys
- Neo4j: Password in URI vs auth tuple

### 6. LLM Security (HIGH)

- System prompts containing sensitive internal information
- User input concatenated into prompts without sanitization
- Tool descriptions exposing internal API details
- Model outputs executed without validation

### 7. File System (MEDIUM)

- Path traversal: `../` in file paths from user input
- Unrestricted file uploads without type validation
- Temporary files with sensitive data not cleaned up
- `is_files_allowed=True` without proper access controls

### 8. Vector Store Security (MEDIUM)

- PII/sensitive data stored without encryption
- Missing tenant isolation in multi-tenant stores
- Document metadata containing sensitive information
- Memory backends storing history indefinitely

### 9. Network Security (MEDIUM)

- HTTP connections instead of HTTPS
- Missing timeout configuration
- SSRF vulnerabilities in URL-accepting parameters

### 10. Dependencies (MEDIUM)

- Outdated dependencies with known CVEs
- Unpinned dependency versions
- Review `# nosec` annotations in code

**Critical dependencies to audit:**
- `RestrictedPython` - Sandbox security
- `litellm` - LLM API security
- `jsonpickle` - Serialization security

## Severity Classification

| Severity | Description |
|----------|-------------|
| CRITICAL | Immediate exploitation risk (code execution, credential exposure) |
| HIGH | Significant impact (injection, auth bypass) |
| MEDIUM | Potential impact (path traversal, missing validation) |

## Output Format

Provide findings as:

```markdown
## Security Analysis Report

### Critical Vulnerabilities
- [ ] CRITICAL: Description with file:line reference

### High-Risk Findings
- [ ] HIGH: Description with file:line reference

### Medium-Risk Issues
- [ ] MEDIUM: Description with file:line reference

### Recommendations
1. Actionable security improvement
```

## Key Files to Check

- `dynamiq/nodes/tools/python.py` - RestrictedPython sandbox
- `dynamiq/nodes/tools/cypher_executor.py` - Query injection
- `dynamiq/connections/connections.py` - Credential handling
- `dynamiq/serializers/loaders/yaml.py` - Deserialization
- `dynamiq/nodes/agents/agent.py` - Prompt injection
