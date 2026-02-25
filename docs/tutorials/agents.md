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
    inference_mode=InferenceMode.FUNCTION_CALLING,
    max_loops=6,
    behaviour_on_max_loops=Behavior.RETURN,
)

writer_agent = Agent(
    name="Brief Writer",
    role="Turn research highlights into a concise executive brief.",
    llm=llm,
    inference_mode=InferenceMode.FUNCTION_CALLING,
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
    inference_mode=InferenceMode.FUNCTION_CALLING,
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

---

This tutorial provides a comprehensive guide to setting up and running agents using Dynamiq. By following these steps, you can create individual agents to tackle complex tasks and combine specialists under a manager to deliver richer multi-agent workflows.
