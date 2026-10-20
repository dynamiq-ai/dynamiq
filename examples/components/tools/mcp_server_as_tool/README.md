# MCP tool examples

## Local code lookup with Sem

With [Sem](https://github.com/Ataraxy-Labs/sem) installed on `PATH` (`sem --version`),
use the stdio adapter to look up a symbol in a trusted local checkout:

```sh
uv sync --python 3.12 --no-install-project --no-default-groups
uv run --python 3.12 --no-default-groups python -m examples.components.tools.mcp_server_as_tool.use_sem \
  --repo /absolute/path/to/checkout --query parse_config
```

This example was tested with Sem 0.27.0. It discovers tools through `MCPServer`,
selects `sem_find` and `sem_grep`, and invokes `sem_find` as a normal Dynamiq node.
The printed result contains code context, not an LLM-generated answer. No model
or API key is required. Existing examples and provider defaults are unchanged.

`SEM_REPO` selects the checkout; it is not a sandbox. Sem may write its local
index. The full server also exposes validation capabilities that can execute
commands, so retain the tool selection when adding this server to an agent.
If an agent uses the results, source content goes to its configured model.

## Parallel Search MCP

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
