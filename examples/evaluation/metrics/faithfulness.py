import logging
import sys

from dotenv import find_dotenv, load_dotenv

from dynamiq.evaluations.metrics import FaithfulnessEvaluator
from dynamiq.nodes.llms import OpenAI


def main():
    # Load environment variables for OpenAI API
    load_dotenv(find_dotenv())

    # Configure logging level (set to DEBUG to see verbose output)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    # Uncomment the following line to enable verbose logging
    # logging.getLogger().setLevel(logging.DEBUG)

    # Initialize the LLM (replace 'gpt-4o-mini' with your available model)
    llm = OpenAI(model="gpt-4o-mini")

    # Sample data (can be replaced with your data)
    questions = ["Who was Albert Einstein and what is he best known for?", "Tell me about the Great Wall of China."]
    answers = [
        (
            "He was a German-born theoretical physicist, widely acknowledged to be one "
            "of the greatest and most influential physicists of all time. "
            "He was best known for developing the theory of relativity, he also made "
            "important contributions to the development of the theory of quantum mechanics."
        ),
        (
            "The Great Wall of China is a large wall in China. "
            "It was built to keep out invaders. "
            "It is visible from space."
        ),
    ]
    contexts = [
        ("Albert Einstein was a German-born theoretical physicist. " "He developed the theory of relativity."),
        (
            "The Great Wall of China is a series of fortifications that were built "
            "across the historical northern borders of ancient Chinese states and "
            "Imperial China as protection against various nomadic groups."
        ),
    ]

    # Initialize evaluator and evaluate
    evaluator = FaithfulnessEvaluator(llm=llm)
    faithfulness_scores = evaluator.run(
        question=questions,
        answer=answers,
        context=contexts,
        verbose=True,  # Set to False to disable verbose logging
    )

    # Print the results
    for idx, score in enumerate(faithfulness_scores):
        print(f"Question: {questions[idx]}")
        print(f"Faithfulness Score: {score}")
        print("-" * 50)

    print("Faithfulness Scores:")
    print(faithfulness_scores)


if __name__ == "__main__":
    main()
