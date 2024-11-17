import logging
import sys

from dotenv import find_dotenv, load_dotenv

from dynamiq.evaluations.metrics.factual_correctness import FactualCorrectnessEvaluator
from dynamiq.nodes.llms import OpenAI


def main():
    # Load environment variables for OpenAI API
    load_dotenv(find_dotenv())

    # Configure logging level (set to DEBUG to see verbose output)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    # Uncomment the following line to enable verbose logging
    # logging.getLogger().setLevel(logging.DEBUG)

    # Initialize the LLM (replace 'gpt-4' with your available model)
    llm = OpenAI(model="gpt-4")

    # Sample data (can be replaced with your data)
    responses = [
        (
            "Albert Einstein was a German theoretical physicist. "
            "He developed the theory of relativity and contributed "
            "to quantum mechanics."
        ),
        ("The Eiffel Tower is located in Berlin, Germany. " "It was constructed in 1889."),
    ]
    references = [
        ("Albert Einstein was a German-born theoretical physicist. " "He developed the theory of relativity."),
        ("The Eiffel Tower is located in Paris, France. " "It was constructed in 1887 and opened in 1889."),
    ]

    # Initialize evaluator and evaluate
    evaluator = FactualCorrectnessEvaluator(llm=llm)
    correctness_scores = evaluator.run(
        responses=responses, references=references, verbose=True  # Set to False to disable verbose logging
    )

    # Print the results
    for idx, score in enumerate(correctness_scores):
        print(f"Response: {responses[idx]}")
        print(f"Factual Correctness Score: {score}")
        print("-" * 50)

    print("Factual Correctness Scores:")
    print(correctness_scores)


if __name__ == "__main__":
    main()
