import logging
import sys

from dotenv import find_dotenv, load_dotenv

from dynamiq.evaluations.metrics import RougeScoreEvaluator


def main():
    """
    Main function to demonstrate the usage of RougeScoreEvaluator.
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

    # Initialize RougeScoreEvaluator with default rouge_type ('rougeL') and measure_type ('fmeasure')
    rouge_evaluator = RougeScoreEvaluator()

    # Optionally, initialize with different rouge_type and measure_type
    # rouge_evaluator = RougeScoreEvaluator(rouge_type="rouge1", measure_type="recall")

    # Run evaluator
    rouge_scores = rouge_evaluator.run(ground_truth_answers=ground_truth_answers, answers=answers)

    # Display the results
    for idx, score in enumerate(rouge_scores):
        print(f"Pair {idx + 1}:")
        print(f"Ground Truth Answer: {ground_truth_answers[idx]}")
        print(f"System Answer: {answers[idx]}")
        print(f"ROUGE Score ({rouge_evaluator.rouge_type} - {rouge_evaluator.measure_type}): {score}")
        print("-" * 60)

    # Optionally, print all scores together
    print("All ROUGE Scores:")
    print(rouge_scores)


if __name__ == "__main__":
    main()
