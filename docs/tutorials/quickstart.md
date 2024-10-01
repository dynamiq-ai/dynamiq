# Quickstart Tutorial

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

## Examples

### Simple LLM Flow

Here's a simple example to get you started with Dynamiq:

**Import Necessary Libraries**

```python
from dynamiq.nodes.llms.openai import OpenAI
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq import Workflow
from dynamiq.prompts import Prompt, Message
```

**Define the Prompt Template for Translation**

Create a template for the prompt that will be used to translate text into English.

```python
prompt_template = """
Translate the following text into English: {{ text }}
"""
```

**Create a Prompt Object with the Defined Template**

```python
prompt = Prompt(messages=[Message(content=prompt_template, role="user")])
```

**Setup Your LLM (Large Language Model) Node**

Configure the LLM node with the necessary parameters such as the model, temperature, and maximum tokens.

```python
llm = OpenAI(
    id="openai",  # Unique identifier for the node
    connection=OpenAIConnection(api_key="$OPENAI_API_KEY"),  # Connection using API key
    model="gpt-4o",  # Model to be used
    temperature=0.3,  # Sampling temperature for the model
    max_tokens=1000,  # Maximum number of tokens in the output
    prompt=prompt  # Prompt to be used for the model
)
```

**Create a Workflow Object**

Initialize a workflow to manage the nodes and their execution.

```python
workflow = Workflow()
```

**Add the LLM Node to the Workflow**

Add the configured LLM node to the workflow.

```python
workflow.flow.add_nodes(llm)
```

**Run the Workflow with the Input Data**

Execute the workflow with the input data that needs to be translated.

```python
result = workflow.run(
    input_data={
        "text": "Hola Mundo!"  # Text to be translated
    }
)
```

**Print the Result of the Translation**

Output the result of the translation to the console.

```python
print(result.output)
```

---

This tutorial provides a quick and easy way to get started with Dynamiq. By following these steps, you can set up a simple workflow to translate text using a large language model.
