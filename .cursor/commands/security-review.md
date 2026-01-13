Perform a security audit of the current code, focusing on vulnerabilities specific to the Dynamiq AI orchestration framework.

Check for these issues in order of severity:

**CRITICAL:**

1. **Hardcoded Secrets**: Search for API keys, passwords, or tokens as string literals. They should use `Field(default_factory=partial(get_env_var, "VAR_NAME"))` instead.

2. **Dangerous Code Execution**: Find any `eval()`, `exec()`, or `compile()` usage outside of `dynamiq/nodes/tools/python.py`. All dynamic code must go through the RestrictedPython sandbox.

3. **Credential Exposure**: Check `to_dict()` implementations - they must support `include_secure_params=False` and never log credentials.

**HIGH:**

4. **Injection Vulnerabilities**: Look for string interpolation in Cypher/SQL queries. Find user input directly embedded in LLM prompts without sanitization.

5. **Missing Input Validation**: Identify nodes without `input_schema` Pydantic models, especially those accepting user input.

6. **Unsafe Deserialization**: Check for `pickle.loads()` or `jsonpickle.decode()` on untrusted data. Review YAML loading for unsafe type resolution.

7. **Authentication Issues**: Find `verify_certs=False`, missing TLS configuration, or credentials in connection strings.

**MEDIUM:**

8. **Path Traversal**: Check file operations for `../` handling and path validation against allowed directories.

9. **Missing Tenant Isolation**: Review vector store and memory backends for multi-tenant data separation.

10. **Dependency Vulnerabilities**: Note any `# nosec` annotations and check critical dependencies like RestrictedPython, litellm, jsonpickle.

For each finding, report:
- Severity level (CRITICAL/HIGH/MEDIUM)
- File path and line number
- Description of the vulnerability
- Recommended fix
