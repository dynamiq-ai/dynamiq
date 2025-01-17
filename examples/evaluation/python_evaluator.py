import logging
import sys

from dotenv import find_dotenv, load_dotenv

from dynamiq.evaluations import PythonEvaluator


def main():
    """
    Main function to demonstrate the usage of PythonEvaluator.
    """
    # Load environment variables if needed (e.g., for other evaluators)
    load_dotenv(find_dotenv())

    # Configure logging to display INFO level and above
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    # Uncomment the following line to enable verbose DEBUG logging
    # logging.getLogger().setLevel(logging.DEBUG)

    # Sample user-defined Python code
    user_code = """
def run(input_data):
    # Retrieve arbitrary values from input_data
    answer = input_data.get("answer")
    expected = input_data.get("expected")
    return 1.0 if answer == expected else 0.0
"""

    # Sample data: list of dictionaries with 'answer' and 'expected' keys
    input_data_list = [
        {"answer": "Paris", "expected": "Paris"},
        {"answer": "Berlin", "expected": "Berlin"},
        {"answer": "Madrid", "expected": "Barcelona"},
    ]

    # Initialize PythonEvaluator with user-defined code
    python_evaluator = PythonEvaluator(code=user_code)

    # Run evaluator on multiple data points
    python_scores = python_evaluator.run(input_data_list=input_data_list)

    # Display the results
    for idx, score in enumerate(python_scores, start=1):
        print(f"Pair {idx}:")
        print(f"Answer: {input_data_list[idx - 1]['answer']}")
        print(f"Expected: {input_data_list[idx - 1]['expected']}")
        print(f"Python Metric Score: {score}")
        print("-" * 60)

    # Optionally, print all scores together
    print("All Python Metric Scores:")
    print(python_scores)


if __name__ == "__main__":
    main()
