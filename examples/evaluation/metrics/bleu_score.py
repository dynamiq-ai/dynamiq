import logging
import sys

from dotenv import find_dotenv, load_dotenv

from dynamiq.evaluations.metrics import BleuScoreEvaluator


def main():
    """
    Main function to demonstrate the usage of BleuScoreEvaluator.
    """
    # Load environment variables if needed (e.g., for other evaluators)
    load_dotenv(find_dotenv())

    # Configure logging to display INFO level and above
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    # Uncomment the following line to enable verbose DEBUG logging
    # logging.getLogger().setLevel(logging.DEBUG)

    # Sample data: ground truth answers and corresponding system-generated answers
    ground_truth_answers = [
        "The cat sits on the mat.",
        "A quick brown fox jumps over the lazy dog.",
        "Python is a versatile programming language used in various domains.",
    ]
    answers = [
        "The cat sits on the mat.",  # Perfect match
        "A fast brown fox leaps over the lazy dog.",  # Slight variation
        "Python is a powerful programming language used across many fields.",  # Slight variation
    ]

    # Initialize BleuScoreEvaluator
    bleu_evaluator = BleuScoreEvaluator()

    # Run evaluator
    bleu_scores = bleu_evaluator.run(references=ground_truth_answers, responses=answers)

    # Display the results
    for idx, score in enumerate(bleu_scores):
        print(f"Pair {idx + 1}:")
        print(f"Ground Truth Answer: {ground_truth_answers[idx]}")
        print(f"System Answer: {answers[idx]}")
        print(f"BLEU Score: {score}")
        print("-" * 60)

    # Optionally, print all scores together
    print("All BLEU Scores:")
    print(bleu_scores)


if __name__ == "__main__":
    main()
