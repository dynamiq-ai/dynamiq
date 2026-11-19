# Agents Tutorial

## Simple Agent

An agent that has access to the E2B Code Interpreter and is capable of solving complex coding tasks.

### Step-by-Step Guide

**Import Necessary Libraries**

```python
from dynamiq.nodes.llms.openai import OpenAI
from dynamiq.connections import OpenAI as OpenAIConnection, E2B as E2BConnection
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.tools.e2b_sandbox import E2BInterpreterTool
```

**Initialize the E2B Tool**

Set up the E2B tool with the necessary API key.

```python
e2b_tool = E2BInterpreterTool(
    connection=E2BConnection(api_key="$API_KEY")
)
```

**Setup Your LLM**

Configure the Large Language Model (LLM) with the necessary parameters such as the model, temperature, and maximum tokens.

```python
llm = OpenAI(
    id="openai",
    connection=OpenAIConnection(api_key="$API_KEY"),
    model="gpt-4o",
    temperature=0.3,
    max_tokens=1000,
)
```

**Create the Agent**

Create an agent that uses the LLM and the E2B tool to solve coding tasks.

```python
agent = Agent(
    name="react-agent",
    llm=llm,
    tools=[e2b_tool],
    role="Senior Data Scientist",
    max_loops=10,
)
```

**Run the Agent with an Input**

Execute the agent with a specific input task.

```python
result = agent.run(
    input_data={
        "input": "Add the first 10 numbers and tell if the result is prime.",
    }
)

print(result.output.get("content"))
```

---

## Manager-led Multi-Agent Workflow

This pattern treats specialist agents as tools that a manager agent can call, allowing you to parallelize work and keep responsibilities focused.

### Step-by-Step Guide

**Import Necessary Libraries**

```python
from dynamiq import Workflow
from dynamiq.connections import OpenAI as OpenAIConnection, ScaleSerp as ScaleSerpConnection
from dynamiq.flows import Flow
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.tools.scale_serp import ScaleSerpTool
from dynamiq.nodes.types import Behavior, InferenceMode
```

**Initialize Tools**

Set up any external tools your specialists need. Here we use a web search tool to gather fresh market information.

```python
search_tool = ScaleSerpTool(
    connection=ScaleSerpConnection(api_key="$SCALESERP_API_KEY")
)
```

**Initialize LLM**

Configure the shared Large Language Model that all agents will use.

```python
llm = OpenAI(
    connection=OpenAIConnection(api_key="$OPENAI_API_KEY"),
    model="gpt-4o",
    temperature=0.1,
)
```

**Define Specialist Agents**

Create the agents that perform research and writing. Each specializes in a single task and expects inputs under the `"input"` key when invoked as a tool.

```python
research_agent = Agent(
    name="Research Analyst",
    role="Find recent market news and provide referenced highlights.",
    llm=llm,
    tools=[search_tool],
    inference_mode=InferenceMode.XML,
    max_loops=6,
    behaviour_on_max_loops=Behavior.RETURN,
)

writer_agent = Agent(
    name="Brief Writer",
    role="Turn research highlights into a concise executive brief.",
    llm=llm,
    inference_mode=InferenceMode.XML,
    max_loops=4,
    behaviour_on_max_loops=Behavior.RETURN,
)
```

**Define the Manager Agent**

The manager agent coordinates the specialists, calls them as tools, and assembles the final answer.

```python
manager_agent = Agent(
    name="Manager",
    role=(
        "Delegate research and writing to sub-agents.\n"
        "Always call tools with {'input': '<task>'} payloads and assemble the final brief."
    ),
    llm=llm,
    tools=[research_agent, writer_agent],
    inference_mode=InferenceMode.XML,
    parallel_tool_calls_enabled=True,
    max_loops=8,
    behaviour_on_max_loops=Behavior.RETURN,
)
```

**Create and Run the Workflow**

Wrap the manager agent in a workflow and provide an input brief. The manager will delegate work to the sub-agents and return a synthesized result.

```python
workflow = Workflow(flow=Flow(nodes=[manager_agent]))

result = workflow.run(
    input_data={"input": "Summarize the latest developments in battery technology for investors."},
)

print(result.output[manager_agent.id]["output"]["content"])
```

## Remote lifecycle authorization

`RemoteLifecycleHook` checks a fixed HTTPS connection before each model and tool
call, including each context-summary chunk, merge and retry. Configure a POST
`Http` connection and add the hook to the agent's `hooks`. It is inherited by
sub-agents by default. The endpoint cannot be chosen by the model.

The host passes opaque authorization state using
`RunnableConfig(trusted_context={"control_context": {...}, "app_id": ...,
"deployment_id": ...})`. This is separate from agent input and is excluded from
checkpoint serialization; the host supplies current state again on resume.

The hook sends only `phase` (`before_model` or `before_tool`), the root `run_id`,
`app_id`, `deployment_id`, and `control_context`. No prompts, model responses or
tool arguments are sent. The endpoint must return HTTP 200 JSON with a Boolean
`allow`; denial, malformed output and transport failure stop the run. Response
reasons are ignored. Requests do not follow redirects or use environment proxies,
and both the request timeout and response size are bounded. A lifecycle endpoint
must be a read-only admission check, not a consequential operation.

Authorization runs once immediately before each SDK-controlled model transport
request or tool execution, including automatic node retries, parameter recovery,
and context-summary requests. Other agent hooks still run at their normal logical
boundaries; a skipped or policy-blocked tool does not make an authorization request.
Tool selection retains the agent's authored identity, including MCP server groups
and sub-agent metadata, after a factory resolves or a parallel call clones its
target. Every node group dispatched as an agent tool is checked. A composite
tool's ordinary tool descendants and internal model requests retain their own
checks, using their own identities. This dispatch state is private to the current
call and is not serialized. A remote context failure raises a terminal node error;
it does not become an observation for another model turn. Explicit node retry policies
remain separate; those attempts still receive their own authorization check.
Provider libraries may retry an already-dispatched HTTP operation internally;
configure those libraries' retry limits separately when an individual HTTP attempt
needs to be controlled.

SDK node output caching is bypassed when the run supplies the `control_context`
key, or the node has an authored/inherited remote lifecycle hook. This prevents
cached model, tool or whole-agent results from bypassing live authorization or
being reused across opaque private contexts. Supplying the required trusted
context at the top level also covers cached composite parents before their child
agent installs its hooks. Unguarded runs that omit the key retain their cache
behavior. An admission layer may normalize an opted-in request's absent context
to an empty object; those runs conservatively bypass caching too. Do not strip
authorization context to reenable caching.
Hosts wrapping guarded agents in arbitrary Python closures must still supply that
top-level context and control their own caches; the SDK cannot discover hidden
calls. Operator-authored mocks suppress external work and lifecycle requests;
their output is a fixture, not proof that authorization succeeded.

The `Dynamiq` router LLM maps an explicitly configured `num_retries` to the
OpenAI SDK client's per-call `max_retries`; unset values preserve that client's
default. These internal HTTP retries still occur inside one SDK invocation.

LLM usage metadata now includes `complete`: false means required provider counters
were absent or invalid, even when compatibility token fields contain zero.
Unknown prices remain null. Hosts must preserve this distinction when recording
spend after failures or cancellation.

`dynamiq.nodes.tools.RemoteContextTool` uses the same fixed connection transport for
host-authorized tools. Configure its `parameters` with an object JSON Schema. It
posts `{run_id, app_id, deployment_id, control_context, input}`; only `input` comes
from validated tool arguments. The connection URL and opaque context are never
tool parameters. Responses must be JSON objects and are limited to 1 MiB. The
runtime must reauthorize the resource request at that endpoint before returning
private data.

Exa search, Jina search/Reader and Firecrawl scrape emit a `research_receipt`
callback observation. It contains bounded query/URL/status metadata from the
actual transport, independently of the model-facing output format. Empty scraped
content does not count as a successful fetch. Receipts contain no document
bodies, headers or error messages. Exa's `costDollars.total`, when valid, is
retained as its estimated `cost_usd`, not an invoice. Jina Reader reports
`usage_units` in `tokens` from `X-Usage-Tokens` or `data.usage.tokens`;
Firecrawl reports `credits` from `data.metadata.creditsUsed`. Zero differs from
unknown, and malformed optional accounting stays unknown. No USD conversion is
inferred from units; an absent receipt price remains unknown usage.

Jina Reader and Firecrawl scrape optionally accept a static `unit_price_usd`
configuration, in USD per reported token or credit respectively. It defaults to
`None`. Only observed units multiplied by an explicitly declared, validated rate
produce an estimated receipt cost. The rate is a JSON number, survives YAML
loading (including explicit zero), and is not a model input or upstream parameter.
It is a host-declared accounting estimate, not provider billing reconciliation.

Jina Reader has optional `page_timeout_seconds`, `proxy` (`auto`, `none`, or a
two-letter country), `token_budget`, `wait_for_selector`, and
`cache_tolerance_seconds` settings. They become `X-Timeout`, `X-Proxy`,
`X-Token-Budget`, `X-Wait-For-Selector`, and `X-Cache-Tolerance` headers.
`no_cache=True` suppresses cache tolerance. Timeout, proxy, budget and cache
controls are hidden from the agent schema; their configured defaults remain
unchanged when omitted.
