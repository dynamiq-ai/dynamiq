import argparse
import os

import google.generativeai as genai


def configure_api_key(api_key: str = None) -> str:
    """
    Configures the Gemini API key.
    """
    if api_key is None:
        api_key = os.environ.get("API_KEY")
    if api_key is None:
        raise ValueError(
            "Missing API key. Please set the API_KEY environment variable or provide it interactively."
        )
    return api_key


def get_user_input(prompt: str) -> str:
    """
    Prompts the user for input interactively.
    """
    while True:
        user_input = input(prompt)
        if user_input:
            return user_input
        print("Please enter a value.")


def validate_file_path(file_path: str) -> None:
    """
    Checks if the provided file path exists and is readable.
    """
    if not os.path.exists(file_path):
        raise ValueError(f"File not found: {file_path}")
    if not os.access(file_path, os.R_OK):
        raise ValueError(
            f"File '{file_path}' is not readable. Please check permissions."
        )


def read_code_from_file(file_path: str) -> str:
    """
    Reads the code content from the specified file path.
    """
    try:
        with open(file_path) as f:
            code_snippet = f.read()
    except FileNotFoundError:
        raise ValueError(f"File not found: {file_path}")
    except Exception as e:
        raise ValueError(f"Error reading file: {e}")
    return code_snippet


def generate_response(
    model_name: str, api_key: str, code_snippet: str, user_question: str
) -> str:
    """
    Generates a response using the Google Gemini API.
    """
    prompt_text = f"""
    You are an advanced AI assistant and a senior software engineer.
    You have been given a code snippet and a question from a software engineer.
    Your task is to provide a detailed and accurate answer to the question by analyzing the provided code.

    Here is the code snippet:
    {code_snippet}

    The software engineer's question is:
    {user_question}

    Please analyze the code snippet and provide a clear, detailed answer to the question.
    Your response should include explanations of the relevant parts of the code and
    any suggestions for improvement or corrections if applicable.
    """

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name=model_name)
    response = model.generate_content(prompt_text)

    return response.text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Use the Gemini API to analyze code and answer questions about it."
    )
    parser.add_argument("--api_key", type=str, help="Google Gemini API key")
    parser.add_argument("--file_path", type=str, help="Path to the codebase file")
    parser.add_argument("--question", type=str, help="Question about the code")
    parser.add_argument(
        "--model_name",
        type=str,
        default="gemini-1.5-pro-latest",
        help="Name of the Gemini model to use",
    )

    args = parser.parse_args()

    api_key = args.api_key
    if not api_key:
        try:
            api_key = configure_api_key()
        except ValueError:
            api_key = get_user_input("Enter your Google Gemini API key: ")

    file_path = args.file_path
    if not file_path:
        file_path = get_user_input("Enter the path to your codebase file: ")

    try:
        validate_file_path(file_path)
    except ValueError as e:
        print(e)
        exit(1)

    try:
        code_snippet = read_code_from_file(file_path)
    except ValueError as e:
        print(e)
        exit(1)

    user_question = args.question
    if not user_question:
        user_question = get_user_input("What question do you have about the code? ")

    model_name = args.model_name

    try:
        response = generate_response(model_name, api_key, code_snippet, user_question)
        print(response)
    except Exception as e:
        print(f"Error generating response: {e}")
        exit(1)


if __name__ == "__main__":
    main()
