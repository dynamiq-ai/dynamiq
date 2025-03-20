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

    # Batch evaluation
    bleu_scores = bleu_evaluator.run(ground_truth_answers=ground_truth_answers, answers=answers)
    print("Batch Evaluation:")
    for idx, score in enumerate(bleu_scores):
        print(f"Pair {idx + 1}:")
        print(f"Ground Truth Answer: {ground_truth_answers[idx]}")
        print(f"System Answer: {answers[idx]}")
        print(f"BLEU Score: {score}")
        print("-" * 60)
    print("All BLEU Scores (Batch):")
    print(bleu_scores)

    # Single evaluation for a specific pair
    print("\nSingle Evaluation:")
    gt_single = "The cat sits on the mat."
    ans_single = "The cat sits on the mat."

    single_score = bleu_evaluator.run_single(ground_truth_answer=gt_single, answer=ans_single)

    print("Ground Truth Answer:", gt_single)
    print("System Answer:", ans_single)
    print("BLEU Score (Single):", single_score)


if __name__ == "__main__":
    main()
