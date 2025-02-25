from dynamiq.evaluations.metrics import BleuScoreEvaluator


def test_bleu_score_evaluator():
    """
    Test the BleuScoreEvaluator to ensure it correctly computes BLEU scores between
    ground truth answers and answers.
    """
    # Sample data
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

    # Initialize evaluator
    bleu_evaluator = BleuScoreEvaluator()

    # Run evaluator
    bleu_scores = bleu_evaluator.run(ground_truth_answers=ground_truth_answers, answers=answers)

    expected_scores = [1.0, 0.37, 0.26]  # Replace with actual printed scores
    for computed, expected in zip(bleu_scores, expected_scores):
        assert abs(computed - expected) < 0.01, f"Expected {expected}, got {computed}"
