# Parallel Search MCP

This separate example discovers `web_search` and `web_fetch` through Dynamiq's
`MCPServer` and runs them using `MCPStreamableHTTP`. It prints web search results
and content from a URL you choose. It needs neither a Parallel API key nor an LLM.

From the repository root, with [uv](https://docs.astral.sh/uv/) installed:

```sh
uv sync --python 3.12 --no-install-project --no-default-groups
uv run --python 3.12 --no-default-groups python -m examples.components.tools.mcp_server_as_tool.use_parallel_search
```

The default query searches for Dynamiq MCP integration and fetches the Dynamiq
repository page. To research another topic and read a specific source:

```sh
uv run --python 3.12 --no-default-groups python -m examples.components.tools.mcp_server_as_tool.use_parallel_search \
  --query "Python asyncio TaskGroup documentation" \
  --url https://docs.python.org/3/library/asyncio-task.html
```

`--url` is explicit; the example does not automatically choose a search result or
generate an answer. The discovered tools use Dynamiq's normal node execution and
return their MCP content. The connection sends a Dynamiq example User-Agent and
no Authorization header. Existing examples, model/provider selections, and
connection timeout defaults are unchanged.

The [Parallel Search MCP documentation](https://docs.parallel.ai/integrations/mcp/search-mcp)
describes anonymous access at `https://search.parallel.ai/mcp` as free for
exploration and light use, with lower rate limits. Both calls share one session
identifier. Tool or transport failures stop the example; if rate limited, wait
for the server's retry deadline before running it again.
