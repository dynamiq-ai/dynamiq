from dynamiq.connections import Mistral as MistralConnection
from dynamiq.nodes.llms import Mistral
from dynamiq.prompts import Message, Prompt


def run_mistral_node(prompt: Prompt):
    connection = MistralConnection()
    mistral_node = Mistral(
        model="mistral/mistral-large-latest",
        connection=connection,
    )
    response = mistral_node.run(input_data={}, prompt=prompt)
    return response


prompt_with_assistant_last = Prompt(
    messages=[
        Message(role="system", content="Be friendly"),
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Can I help you?", prefix=True),
    ]
)


if __name__ == "__main__":
    for msg in prompt_with_assistant_last.messages:
        print(msg)
    response = run_mistral_node(prompt_with_assistant_last)
    print(response)
