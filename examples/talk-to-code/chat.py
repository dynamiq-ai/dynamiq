import argparse
import os
from typing import Any

from dynamiq import Workflow
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.connections import Gemini as GeminiConnection
from dynamiq.flows import Flow
from dynamiq.nodes.llms import Gemini
from dynamiq.prompts import Message, Prompt
from dynamiq.runnables import RunnableConfig


def get_user_input(prompt: str) -> str:
    """Prompts the user for input interactively."""
    while True:
        user_input = input(prompt).strip()
        if user_input:
            return user_input
        print("Please enter a value.")


def configure_api_key(api_key: str = None) -> str:
    """Configures the Gemini API key."""
    if api_key is None:
        api_key = os.environ.get("GEMINI_API_KEY")
    if api_key is None:
        api_key = get_user_input("Enter your Gemini API key: ")
    return api_key


def read_code_from_file(file_path: str) -> str:
    """Reads the code content from the specified file path."""
    try:
        with open(file_path) as f:
            return f.read()
    except FileNotFoundError:
        raise ValueError(f"File not found: {file_path}")
    except Exception as e:
        raise ValueError(f"Error reading file: {e}")


def create_workflow(api_key: str, model_name: str) -> Workflow:
    """Creates a workflow for code analysis using Gemini."""
    prompt_text = """
    You are an advanced AI assistant and a senior software engineer.
    You have been given a code snippet and a question from a software engineer.
    Your task is to provide a detailed and accurate answer to the question by analyzing the provided code.

    Here is the code snippet:
    {{code_snippet}}

    The software engineer's question is:
    {{user_question}}

    Please analyze the code snippet and provide a clear, detailed answer to the question.
    Your response should include explanations of the relevant parts of the code and
    any suggestions for improvement or corrections if applicable.
    """

    prompt = Prompt(messages=[Message(role="user", content=prompt_text)])

    gemini_node = Gemini(
        id="gemini",
        name="Gemini Code Analyzer",
        model=model_name,
        connection=GeminiConnection(api_key=api_key),
        prompt=prompt,
    )

    flow = Flow(nodes=[gemini_node])
    return Workflow(flow=flow)


def main():
    parser = argparse.ArgumentParser(description="Analyze code using Gemini API")
    parser.add_argument("--api_key", type=str, help="Gemini API key")
    parser.add_argument("--file_path", type=str, help="Path to the code file")
    parser.add_argument("--question", type=str, help="Question about the code")
    parser.add_argument("--model_name", type=str, default="gemini/gemini-1.5-pro-latest", help="Gemini model name")
    args = parser.parse_args()

    try:
        api_key = configure_api_key(args.api_key)

        file_path = args.file_path
        if not file_path:
            file_path = get_user_input("Enter the path to your code file: ")

        code_snippet = read_code_from_file(file_path)

        question = args.question
        if not question:
            question = get_user_input("Enter your question about the code: ")

        workflow = create_workflow(api_key, args.model_name)

        input_data: dict[str, Any] = {
            "code_snippet": code_snippet,
            "user_question": question
        }

        config = RunnableConfig(callbacks=[TracingCallbackHandler()])
        result = workflow.run(input_data=input_data, config=config)

        print("\nAnalysis Result:")
        print(result.output['gemini'].get('output').get('content'))

    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
