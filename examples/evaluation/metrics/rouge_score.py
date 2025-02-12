from dotenv import find_dotenv, load_dotenv
from dynamiq.evaluations.metrics import RougeScoreEvaluator


def main():
    load_dotenv(find_dotenv())

    ground_truth_answers = [
        "The cat sits on the mat.",
        "A quick brown fox jumps over the lazy dog.",
        "Python is a versatile programming language used in various domains.",
    ]
    answers = [
        "The cat sits on the mat.",
        "A fast brown fox leaps over the lazy dog.",
        "Python is a powerful programming language used across many fields.",
    ]

    rouge_evaluator = RougeScoreEvaluator()  # Defaults: rougeL/fmeasure
    # For a different measure, uncomment the following:
    # rouge_evaluator = RougeScoreEvaluator(rouge_type="rouge1", measure_type="recall")

    rouge_scores = rouge_evaluator.run(ground_truth_answers=ground_truth_answers, answers=answers)

    for idx, score in enumerate(rouge_scores):
        print(f"Pair {idx + 1}:")
        print(f"Ground Truth Answer: {ground_truth_answers[idx]}")
        print(f"System Answer: {answers[idx]}")
        print(f"ROUGE Score ({rouge_evaluator.rouge_type} - {rouge_evaluator.measure_type}): {score}")
        print("-" * 60)

    print("All ROUGE Scores:")
    print(rouge_scores)


if __name__ == "__main__":
    main()
