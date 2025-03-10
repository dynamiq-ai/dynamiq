import os

from dynamiq.nodes.llms import DynamiqClient

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


def run_llm_client_with_openai_with_tracing():
    llm_client = DynamiqClient(trace=True)

    client = llm_client.openai
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a very accurate calculator."},
            {"role": "user", "content": "1 + 1 = "},
        ],
    )
    print("Response:", response)

    response = client.embeddings.create(
        model="text-embedding-ada-002", input=["Hello, how are you?", "AI is the future"]
    )
    print("Embeddings response:", response)

    response = client.moderations.create(input="This is some potentially harmful content.")
    print("Moderation response:", response)


def run_llm_client_with_anthropic():
    llm_client = DynamiqClient()

    anthropic = llm_client.anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    response = anthropic.messages.create(
        model="claude-2", messages=[{"role": "user", "content": "What is the meaning of life?"}], max_tokens=100
    )
    print("Anthropic response:", response)

    response = anthropic.messages.create(
        model="claude-2",
        max_tokens=200,
        stream=True,
        messages=[{"role": "user", "content": "Tell me about the future of AI."}],
    )

    for chunk in response:
        print(chunk)


if __name__ == "__main__":
    run_llm_client_with_openai_with_tracing()
    run_llm_client_with_anthropic()
