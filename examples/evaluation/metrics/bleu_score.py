from dynamiq.evaluations.metrics import BleuScoreEvaluator


def main():
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

    bleu_evaluator = BleuScoreEvaluator()
    bleu_scores = bleu_evaluator.run(ground_truth_answers=ground_truth_answers, answers=answers)

    for idx, score in enumerate(bleu_scores):
        print(f"Pair {idx + 1}:")
        print(f"Ground Truth Answer: {ground_truth_answers[idx]}")
        print(f"System Answer: {answers[idx]}")
        print(f"BLEU Score: {score}")
        print("-" * 60)

    print("All BLEU Scores:")
    print(bleu_scores)


if __name__ == "__main__":
    main()
