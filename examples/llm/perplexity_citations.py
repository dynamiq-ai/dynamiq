from dynamiq.nodes.llms import Perplexity
from dynamiq.prompts import Prompt


def run_perplexity_node(prompt: Prompt):
    openai_node = Perplexity(
        model="llama-3.1-sonar-small-128k-online",
        return_citations=True,
    )
    response = openai_node.run(input_data={}, prompt=prompt)
    return response


prompt = Prompt(
    messages=[
        {
            "role": "user",
            "content": ("Who won euro 2024?"),
        },
    ]
)


if __name__ == "__main__":
    response = run_perplexity_node(prompt)
    print(response)
