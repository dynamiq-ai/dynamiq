import logging
import sys

from dotenv import find_dotenv, load_dotenv

# Importing the evaluators (adjust the import path as necessary)
from dynamiq.evaluations.metrics import (
    DistanceMeasure,
    ExactMatchEvaluator,
    StringPresenceEvaluator,
    StringSimilarityEvaluator,
)


def main():
    """
    Main function to demonstrate the usage of string evaluators:
    - ExactMatchEvaluator
    - StringPresenceEvaluator
    - StringSimilarityEvaluator
    """
    # Load environment variables if needed (not essential for string evaluators)
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
        "The cat sits on the mat.",
        "A fast brown fox leaps over the lazy dog.",
        "Python is a versatile language used for many applications.",
    ]

    # Initialize ExactMatchEvaluator
    exact_match_evaluator = ExactMatchEvaluator()

    # Initialize StringPresenceEvaluator
    string_presence_evaluator = StringPresenceEvaluator()

    # Initialize StringSimilarityEvaluator with different distance measures
    string_similarity_evaluator_lev = StringSimilarityEvaluator(distance_measure=DistanceMeasure.LEVENSHTEIN)
    string_similarity_evaluator_jw = StringSimilarityEvaluator(distance_measure=DistanceMeasure.JARO_WINKLER)

    # Run ExactMatchEvaluator
    exact_match_scores = exact_match_evaluator.run(ground_truth_answers=ground_truth_answers, answers=answers)

    # Run StringPresenceEvaluator
    string_presence_scores = string_presence_evaluator.run(ground_truth_answers=ground_truth_answers, answers=answers)

    # Run StringSimilarityEvaluator with Levenshtein distance
    string_similarity_scores_lev = string_similarity_evaluator_lev.run(
        ground_truth_answers=ground_truth_answers, answers=answers
    )

    # Run StringSimilarityEvaluator with Jaro-Winkler distance
    string_similarity_scores_jw = string_similarity_evaluator_jw.run(
        ground_truth_answers=ground_truth_answers, answers=answers
    )

    # Display the results
    for idx in range(len(ground_truth_answers)):
        print(f"Pair {idx + 1}:")
        print(f"Ground Truth Answer: {ground_truth_answers[idx]}")
        print(f"System Answer: {answers[idx]}")
        print(f"Exact Match Score: {exact_match_scores[idx]}")
        print(f"String Presence Score: {string_presence_scores[idx]}")
        print(f"String Similarity Score (Levenshtein): {string_similarity_scores_lev[idx]}")
        print(f"String Similarity Score (Jaro-Winkler): {string_similarity_scores_jw[idx]}")
        print("-" * 60)

    # Optionally, print all scores together
    print("All Exact Match Scores:")
    print(exact_match_scores)
    print("\nAll String Presence Scores:")
    print(string_presence_scores)
    print("\nAll String Similarity Scores (Levenshtein):")
    print(string_similarity_scores_lev)
    print("\nAll String Similarity Scores (Jaro-Winkler):")
    print(string_similarity_scores_jw)


if __name__ == "__main__":
    main()
