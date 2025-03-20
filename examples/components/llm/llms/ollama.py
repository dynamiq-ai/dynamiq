from dynamiq.nodes.llms import Ollama
from dynamiq.prompts import Prompt


def run_ollama_node(prompt: Prompt):
    ollama_node = Ollama(
        model="ollama/qwq",
    )
    response = ollama_node.run(input_data={}, prompt=prompt)
    return response


prompt = Prompt(
    messages=[
        {
            "role": "user",
            "content": "Explain the concept of entropy.",
        },
    ]
)


if __name__ == "__main__":
    response = run_ollama_node(prompt)
    print(response)
