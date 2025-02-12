
<p align="center">
  <a href="https://www.getdynamiq.ai/"><img src="https://github.com/dynamiq-ai/dynamiq/blob/main/docs/img/Dynamiq_Logo_Universal_Github.png?raw=true" alt="Dynamiq"></a>
</p>


<p align="center">
    <em>Dynamiq is an orchestration framework for agentic AI and LLM applications</em>
</p>

<p align="center">
  <a href="https://getdynamiq.ai">
    <img src="https://img.shields.io/website?label=website&up_message=online&url=https%3A%2F%2Fgetdynamiq.ai" alt="Website">
  </a>
  <a href="https://github.com/dynamiq-ai/dynamiq/releases" target="_blank">
    <img src="https://img.shields.io/github/release/dynamiq-ai/dynamiq" alt="Release Notes">
  </a>
  <a href="#" target="_blank">
    <img src="https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg" alt="Python 3.10+">
  </a>
  <a href="https://github.com/dynamiq-ai/dynamiq/blob/main/LICENSE" target="_blank">
    <img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License">
  </a>
  <a href="https://dynamiq-ai.github.io/dynamiq" target="_blank">
    <img src="https://img.shields.io/website?label=documentation&up_message=online&url=https%3A%2F%2Fdynamiq-ai.github.io%2Fdynamiq" alt="Documentation">
  </a>
</p>


Welcome to Dynamiq! 🤖

Dynamiq is your all-in-one Gen AI framework, designed to streamline the development of AI-powered applications. Dynamiq specializes in orchestrating retrieval-augmented generation (RAG) and large language model (LLM) agents.

## Getting Started

Ready to dive in? Here's how you can get started with Dynamiq:

### Installation

First, let's get Dynamiq installed. You'll need Python, so make sure that's set up on your machine. Then run:

```sh
pip install dynamiq
```

Or build the Python package from the source code:
```sh
git clone https://github.com/dynamiq-ai/dynamiq.git
cd dynamiq
poetry install
```

## Documentation
For more examples and detailed guides, please refer to our [documentation](https://dynamiq-ai.github.io/dynamiq).

## Examples

### Simple LLM Flow

Here's a simple example to get you started with Dynamiq:

```python
from dynamiq.nodes.llms.openai import OpenAI
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.prompts import Prompt, Message

# Define the prompt template for translation
prompt_template = """
Translate the following text into English: {{ text }}
"""

# Create a Prompt object with the defined template
prompt = Prompt(messages=[Message(content=prompt_template, role="user")])

# Setup your LLM (Large Language Model) Node
llm = OpenAI(
    id="openai",  # Unique identifier for the node
    connection=OpenAIConnection(api_key="OPENAI_API_KEY"),  # Connection using API key
    model="gpt-4o",  # Model to be used
    temperature=0.3,  # Sampling temperature for the model
    max_tokens=1000,  # Maximum number of tokens in the output
    prompt=prompt  # Prompt to be used for the model
)

# Run the LLM node with the input data
result = llm.run(
    input_data={
        "text": "Hola Mundo!"  # Text to be translated
    }
)

# Print the result of the translation
print(result.output)
```

### Simple ReAct Agent
An agent that has the access to E2B Code Interpreter and is capable of solving complex coding tasks.

```python
from dynamiq.nodes.llms.openai import OpenAI
from dynamiq.connections import OpenAI as OpenAIConnection, E2B as E2BConnection
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.tools.e2b_sandbox import E2BInterpreterTool

# Initialize the E2B tool
e2b_tool = E2BInterpreterTool(
    connection=E2BConnection(api_key="E2B_API_KEY")
)

# Setup your LLM
llm = OpenAI(
    id="openai",
    connection=OpenAIConnection(api_key="OPENAI_API_KEY"),
    model="gpt-4o",
    temperature=0.3,
    max_tokens=1000,
)

# Create the ReAct agent
agent = ReActAgent(
    name="react-agent",
    llm=llm, # Language model instance
    tools=[e2b_tool],  # List of tools that the agent can use
    role="Senior Data Scientist",  # Role of the agent
    max_loops=10, # Limit on the number of processing loops
)

# Run the agent with an input
result = agent.run(
    input_data={
        "input": "Add the first 10 numbers and tell if the result is prime.",
    }
)

print(result.output.get("content"))
```

### Configuring Two Parallel Agents with WorkFlow

```python
from dynamiq import Workflow
from dynamiq.nodes.llms import OpenAI
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.agents.reflection import ReflectionAgent

# Setup your LLM
llm = OpenAI(
    connection=OpenAIConnection(api_key="OPENAI_API_KEY"),
    model="gpt-4o",
    temperature=0.1,
)

# Define the first agent: a question answering agent
first_agent = ReflectionAgent(
    name="Expert Agent",
    llm=llm,
    role="Professional writer with the goal of producing well-written and informative responses.",
    id="agent_1",
    max_loops=5
)

# Define the second agent: a poetic writer
second_agent = ReflectionAgent(
    name="Poetic Rewriter Agent",
    llm=llm,
    role="Professional writer with the goal of rewriting user input as a poem without changing its meaning.",
    id="agent_2",
    max_loops=5
)


# Create a workflow to run both agents with the same input
# The `Workflow` class simplifies setting up and executing a series of nodes in a pipeline.
# It automatically handles running the agents in parallel where possible.
wf = Workflow()
wf.flow.add_nodes(first_agent)
wf.flow.add_nodes(second_agent)

# Equivalent alternative way to define the workflow:
# from dynamiq.flows import Flow
# wf = Workflow(flow=Flow(nodes=[agent_first, agent_second]))

# Run the workflow with an input
result = wf.run(
        input_data={"input": "How are sin(x) and cos(x) connected in electrodynamics?"},
    )

# Print the input and output for both agents
print('--- Agent 1: Input ---\n', result.output[first_agent.id].get("input").get('input'))
print('--- Agent 1: Output ---\n', result.output[first_agent.id].get("output").get('content'))
print('--- Agent 2: Input ---\n', result.output[second_agent.id].get("input").get('input'))
print('--- Agent 2: Output ---\n', result.output[second_agent.id].get("output").get('content'))
```

### Configuring Two Sequential Agents with WorkFlow

```python
from dynamiq import Workflow
from dynamiq.nodes.llms import OpenAI
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.agents.reflection import ReflectionAgent

from dynamiq.nodes.node import InputTransformer, NodeDependency

# Setup your LLM
llm = OpenAI(
    connection=OpenAIConnection(api_key="OPENAI_API_KEY"),
    model="gpt-4o",
    temperature=0.1,
)

first_agent = ReflectionAgent(
    name="Expert Agent",
    llm=llm,
    role="Professional writer with the goal of producing well-written and informative responses.",  # Role of the agent
    id="agent_1",
    max_loops=5
)

second_agent = ReflectionAgent(
    name="Poetic Rewriter Agent",
    llm=llm,
    role="Professional writer with the goal of rewriting user input as a poem without changing its meaning.",  # Role of the agent
    id="agent_2",
    depends=[NodeDependency(first_agent)],  # Set dependency on the first agent
    input_transformer=InputTransformer(
        selector={"input": f"${[first_agent.id]}.output.content"}  # Extract the output of the first agent as input
    ),
    max_loops=5
)

# Create a workflow to run the agents sequentially based on dependencies.
# Without a workflow, you would need to run `first_agent`, collect its output,
# and then manually pass that output as input to `second_agent`. The workflow automates this process.
wf = Workflow()
wf.flow.add_nodes(first_agent)
wf.flow.add_nodes(second_agent)

# Equivalent alternative way to define the workflow:
# from dynamiq.flows import Flow
# wf = Workflow(flow=Flow(nodes=[agent_first, agent_second]))

# Run the workflow with an input
result = wf.run(
        input_data={"input": "How are sin(x) and cos(x) connected in electrodynamics?"},
    )

# Print the input and output for both agents
print('--- Agent 1: Input ---\n', result.output[first_agent.id].get("input").get('input'))
print('--- Agent 1: Output ---\n', result.output[first_agent.id].get("output").get('content'))
print('--- Agent 2: Input ---\n', result.output[second_agent.id].get("input").get('input'))
print('--- Agent 2: Output ---\n', result.output[second_agent.id].get("output").get('content'))
```

### Multi-agent orchestration
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

# Initialize tools
python_tool = E2BInterpreterTool(
    connection=E2BConnection(api_key="E2B_API_KEY")
)
search_tool = ScaleSerpTool(
    connection=ScaleSerpConnection(api_key="SCALESERP_API_KEY")
)

# Initialize LLM
llm = OpenAI(
    connection=OpenAIConnection(api_key="OPENAI_API_KEY"),
    model="gpt-4o",
    temperature=0.1,
)

# Define agents
coding_agent = ReActAgent(
    name="coding-agent",
    llm=llm,
    tools=[python_tool],
    role=("Expert agent with coding skills."
          "Goal is to provide the solution to the input task"
          "using Python software engineering skills."),
    max_loops=15,
)

planner_agent = ReflectionAgent(
    name="planner-agent",
    llm=llm,
    role=("Expert agent with planning skills."
          "Goal is to analyze complex requests"
          "and provide a detailed action plan."),
)

search_agent = ReActAgent(
    name="search-agent",
    llm=llm,
    tools=[search_tool],
    role=("Expert agent with web search skills."
          "Goal is to provide the solution to the input task"
          "using web search and summarization skills."),
    max_loops=10,
)

# Initialize the adaptive agent manager
agent_manager = AdaptiveAgentManager(llm=llm)

# Create the orchestrator
orchestrator = AdaptiveOrchestrator(
    name="adaptive-orchestrator",
    agents=[coding_agent, planner_agent, search_agent],
    manager=agent_manager,
)

# Define the input task
input_task = (
    "Use coding skills to gather data about Nvidia and Intel stock prices for the last 10 years, "
    "calculate the average per year for each company, and create a table. Then craft a report "
    "and add a conclusion: what would have been better if I had invested $100 ten years ago?"
)

# Run the orchestrator
result = orchestrator.run(
    input_data={"input": input_task},
)

# Print the result
print(result.output.get("content"))

```

### RAG - document indexing flow
This workflow takes input PDF files, pre-processes them, converts them to vector embeddings, and stores them in the Pinecone vector database.
The example provided is for an existing index in Pinecone. You can find examples for index creation on the `docs/tutorials/rag` page.

```python
from io import BytesIO

from dynamiq import Workflow
from dynamiq.connections import OpenAI as OpenAIConnection, Pinecone as PineconeConnection
from dynamiq.nodes.converters import PyPDFConverter
from dynamiq.nodes.splitters.document import DocumentSplitter
from dynamiq.nodes.embedders import OpenAIDocumentEmbedder
from dynamiq.nodes.writers import PineconeDocumentWriter

rag_wf = Workflow()

# PyPDF document converter
converter = PyPDFConverter(document_creation_mode="one-doc-per-page")
rag_wf.flow.add_nodes(converter)  # add node to the DAG

# Document splitter
document_splitter = (
    DocumentSplitter(
        split_by="sentence",
        split_length=10,
        split_overlap=1,
    )
    .inputs(documents=converter.outputs.documents)  # map converter node output to the expected input of the current node
    .depends_on(converter)
)
rag_wf.flow.add_nodes(document_splitter)

# OpenAI vector embeddings
embedder = (
    OpenAIDocumentEmbedder(
        connection=OpenAIConnection(api_key="OPENAI_API_KEY"),
        model="text-embedding-3-small",
    )
    .inputs(documents=document_splitter.outputs.documents)
    .depends_on(document_splitter)
)
rag_wf.flow.add_nodes(embedder)

# Pinecone vector storage
vector_store = (
    PineconeDocumentWriter(
        connection=PineconeConnection(api_key="PINECONE_API_KEY"),
        index_name="default",
        dimension=1536,
    )
    .inputs(documents=embedder.outputs.documents)
    .depends_on(embedder)
)
rag_wf.flow.add_nodes(vector_store)

# Prepare input PDF files
file_paths = ["example.pdf"]
input_data = {
    "files": [
        BytesIO(open(path, "rb").read()) for path in file_paths
    ],
    "metadata": [
        {"filename": path} for path in file_paths
    ],
}

# Run RAG indexing flow
rag_wf.run(input_data=input_data)
```

### RAG - document retrieval flow
Simple retrieval RAG flow that searches for relevant documents and answers the original user question using retrieved documents.

```python
from dynamiq import Workflow
from dynamiq.connections import OpenAI as OpenAIConnection, Pinecone as PineconeConnection
from dynamiq.nodes.embedders import OpenAITextEmbedder
from dynamiq.nodes.retrievers import PineconeDocumentRetriever
from dynamiq.nodes.llms import OpenAI
from dynamiq.prompts import Message, Prompt

# Initialize the RAG retrieval workflow
retrieval_wf = Workflow()

# Shared OpenAI connection
openai_connection = OpenAIConnection(api_key="OPENAI_API_KEY")

# OpenAI text embedder for query embedding
embedder = OpenAITextEmbedder(
    connection=openai_connection,
    model="text-embedding-3-small",
)
retrieval_wf.flow.add_nodes(embedder)

# Pinecone document retriever
document_retriever = (
    PineconeDocumentRetriever(
        connection=PineconeConnection(api_key="PINECONE_API_KEY"),
        index_name="default",
        dimension=1536,
        top_k=5,
    )
    .inputs(embedding=embedder.outputs.embedding)
    .depends_on(embedder)
)
retrieval_wf.flow.add_nodes(document_retriever)

# Define the prompt template
prompt_template = """
Please answer the question based on the provided context.

Question: {{ query }}

Context:
{% for document in documents %}
- {{ document.content }}
{% endfor %}

"""

# OpenAI LLM for answer generation
prompt = Prompt(messages=[Message(content=prompt_template, role="user")])

answer_generator = (
    OpenAI(
        connection=openai_connection,
        model="gpt-4o",
        prompt=prompt,
    )
    .inputs(
        documents=document_retriever.outputs.documents,
        query=embedder.outputs.query,
    )  # take documents from the vector store node and query from the embedder
    .depends_on([document_retriever, embedder])
)
retrieval_wf.flow.add_nodes(answer_generator)

# Run the RAG retrieval flow
question = "What are the line intems provided in the invoice?"
result = retrieval_wf.run(input_data={"query": question})

answer = result.output.get(answer_generator.id).get("output", {}).get("content")
print(answer)
```

### Simple Chatbot with Memory
A simple chatbot that uses the `Memory` module to store and retrieve conversation history.

```python
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.memory import Memory
from dynamiq.memory.backends.in_memory import InMemory
from dynamiq.nodes.agents.simple import SimpleAgent
from dynamiq.nodes.llms import OpenAI

AGENT_ROLE = "helpful assistant, goal is to provide useful information and answer questions"
llm = OpenAI(
    connection=OpenAIConnection(api_key="OPENAI_API_KEY"),
    model="gpt-4o",
    temperature=0.1,
)

memory = Memory(backend=InMemory())

agent = SimpleAgent(
    name="Agent",
    llm=llm,
    role=AGENT_ROLE,
    id="agent",
    memory=memory,
)


def main():
    print("Welcome to the AI Chat! (Type 'exit' to end)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        response = agent.run({"input": user_input})
        response_content = response.output.get("content")
        print(f"AI: {response_content}")


if __name__ == "__main__":
    main()
```

### Graph Orchestrator
Graph Orchestrator allows to create any architecture tailored to specific use cases.
Example of simple workflow that manages iterative process of feedback and refinement of email.

```python
from typing import Any

from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.agents.orchestrators.graph import END, START, GraphOrchestrator
from dynamiq.nodes.agents.orchestrators.graph_manager import GraphAgentManager
from dynamiq.nodes.agents.simple import SimpleAgent
from dynamiq.nodes.llms import OpenAI

llm = OpenAI(
    connection=OpenAIConnection(api_key="OPENAI_API_KEY"),
    model="gpt-4o",
    temperature=0.1,
)

email_writer = SimpleAgent(
    name="email-writer-agent",
    llm=llm,
    role="Write personalized emails taking into account feedback.",
)


def gather_feedback(context: dict[str, Any], **kwargs):
    """Gather feedback about email draft."""
    feedback = input(
        f"Email draft:\n"
        f"{context.get('history', [{}])[-1].get('content', 'No draft')}\n"
        f"Type in SEND to send email, CANCEL to exit, or provide feedback to refine email: \n"
    )

    reiterate = True

    result = f"Gathered feedback: {feedback}"

    feedback = feedback.strip().lower()
    if feedback == "send":
        print("####### Email was sent! #######")
        result = "Email was sent!"
        reiterate = False
    elif feedback == "cancel":
        print("####### Email was canceled! #######")
        result = "Email was canceled!"
        reiterate = False

    return {"result": result, "reiterate": reiterate}


def router(context: dict[str, Any], **kwargs):
    """Determines next state based on provided feedback."""
    if context.get("reiterate", False):
        return "generate_sketch"

    return END


orchestrator = GraphOrchestrator(
    name="Graph orchestrator",
    manager=GraphAgentManager(llm=llm),
)

# Attach tasks to the states. These tasks will be executed when the respective state is triggered.
orchestrator.add_state_by_tasks("generate_sketch", [email_writer])
orchestrator.add_state_by_tasks("gather_feedback", [gather_feedback])

# Define the flow between states by adding edges.
# This configuration creates the sequence of states from START -> "generate_sketch" -> "gather_feedback".
orchestrator.add_edge(START, "generate_sketch")
orchestrator.add_edge("generate_sketch", "gather_feedback")

# Add a conditional edge to the "gather_feedback" state, allowing the flow to branch based on a condition.
# The router function will determine whether the flow should go to "generate_sketch" (reiterate) or END (finish the process).
orchestrator.add_conditional_edge("gather_feedback", ["generate_sketch", END], router)


if __name__ == "__main__":
    print("Welcome to email writer.")
    email_details = input("Provide email details: ")
    orchestrator.run(input_data={"input": f"Write and post email, provide feedback about status of email: {email_details}"})
```

## Contributing

We love contributions! Whether it's bug reports, feature requests, or pull requests, head over to our [CONTRIBUTING.md](CONTRIBUTING.md) to see how you can help.

## License

Dynamiq is open-source and available under the [Apache 2 License](LICENSE).

Happy coding! 🚀
