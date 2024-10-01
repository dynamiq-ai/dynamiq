# Agents Tutorial

## Simple ReAct Agent

An agent that has access to the E2B Code Interpreter and is capable of solving complex coding tasks.

### Step-by-Step Guide

**Import Necessary Libraries**

```python
from dynamiq.nodes.llms.openai import OpenAI
from dynamiq.connections import OpenAI as OpenAIConnection, E2B as E2BConnection
from dynamiq.nodes.agents.react import ReActAgent
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

**Create the ReAct Agent**

Create an agent that uses the LLM and the E2B tool to solve coding tasks.

```python
agent = ReActAgent(
    name="react-agent",
    llm=llm,
    tools=[e2b_tool],
    role="Senior Data Scientist",
    goal="Provide well-explained final answers to analytical questions",
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

## Multi-agent Orchestration

### Step-by-Step Guide

**Import Necessary Libraries**

```python
from dynamiq.connections import (OpenAI as OpenAIConnection,
                                 ScaleSerp as ScaleSerpConnection,
                                 E2B as E2BConnection)
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.agents.orchestrators.adaptive import AdaptiveOrchestrator
from dynamiq.nodes.agents.orchestrators.adaptive_manager import AdaptiveAgentManager
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.agents.reflection import ReflectionAgent
from dynamiq.nodes.tools.e2b_sandbox import E2BInterpreterTool
from dynamiq.nodes.tools.scale_serp import ScaleSerpTool
```

**Initialize Tools**

Set up the tools required for coding and web search tasks.

```python
python_tool = E2BInterpreterTool(
    connection=E2BConnection(api_key="$E2B_API_KEY")
)
search_tool = ScaleSerpTool(
    connection=ScaleSerpConnection(api_key="$SCALESERP_API_KEY")
)
```

**Initialize LLM**

Configure the Large Language Model (LLM) with the necessary parameters.

```python
llm = OpenAI(
    connection=OpenAIConnection(api_key="$OPENAI_API_KEY"),
    model="gpt-4o",
    temperature=0.2,
)
```

**Define Agents**

Create agents with specific roles and goals.

```python
coding_agent = ReActAgent(
    name="coding-agent",
    llm=llm,
    tools=[python_tool],
    role="Expert agent with coding skills.",
    goal="Provide the solution to the input task using Python software engineering skills.",
    max_loops=15,
)

planner_agent = ReflectionAgent(
    name="planner-agent",
    llm=llm,
    role="Expert agent with planning skills.",
    goal="Analyze complex requests and provide detailed action plan.",
)

search_agent = ReActAgent(
    name="search-agent",
    llm=llm,
    tools=[search_tool],
    role="Expert agent with web search skills.",
    goal="Provide the best information available using web browsing and searching skills.",
    max_loops=10,
)
```

**Initialize the Adaptive Agent Manager**

Set up the manager to handle the orchestration of multiple agents.

```python
agent_manager = AdaptiveAgentManager(llm=llm)
```

**Create the Orchestrator**

Create an orchestrator to manage the execution of multiple agents.

```python
orchestrator = AdaptiveOrchestrator(
    name="adaptive-orchestrator",
    agents=[coding_agent, planner_agent, search_agent],
    manager=agent_manager,
)
```

**Define the Input Task**

Specify the task that the orchestrator will handle.

```python
input_task = (
    "Use coding skills to gather data about Nvidia and Intel stock prices for the last 10 years, "
    "calculate the average per year for each company, and create a table. Then craft a report "
    "and add a conclusion: what would have been better if I had invested $100 ten years ago?"
)
```

**Run the Orchestrator**

Execute the orchestrator with the defined input task.

```python
result = orchestrator.run(
    input_data={"input": input_task},
)

# Print the result
print(result.output.get("content"))
```

---

This tutorial provides a comprehensive guide to setting up and running agents using Dynamiq. By following these steps, you can create agents capable of solving complex tasks and orchestrate multiple agents to handle more sophisticated workflows.
