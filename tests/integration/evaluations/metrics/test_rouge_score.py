from dynamiq.evaluations.metrics import RougeScoreEvaluator


def test_rouge_score_evaluator_partial_matches():
    """
    Test the RougeScoreEvaluator with partial matches to ensure it correctly
    computes ROUGE scores that reflect partial similarity.
    """
    # Sample data
    ground_truth_answers = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world.",
    ]
    answers = [
        "A quick brown fox leaps over the lazy dog.",  # Partial match
        "AI is changing the world in unprecedented ways.",  # Partial match
    ]

    # Initialize RougeScoreEvaluator with 'rouge1' and 'recall'
    rouge_evaluator_rouge1_recall = RougeScoreEvaluator(rouge_type="rouge1", measure_type="recall")

    # Initialize RougeScoreEvaluator with 'rouge2' and 'precision'
    rouge_evaluator_rouge2_precision = RougeScoreEvaluator(rouge_type="rouge2", measure_type="precision")

    # Run evaluators
    rouge_scores_rouge1_recall = rouge_evaluator_rouge1_recall.run(
        ground_truth_answers=ground_truth_answers, answers=answers
    )
    rouge_scores_rouge2_precision = rouge_evaluator_rouge2_precision.run(
        ground_truth_answers=ground_truth_answers, answers=answers
    )

    expected_scores_rouge1_recall = [0.78, 0.5]  # Replace with actual printed scores
    expected_scores_rouge2_precision = [0.62, 0.14]  # Replace with actual printed scores

    for computed, expected in zip(rouge_scores_rouge1_recall, expected_scores_rouge1_recall):
        assert abs(computed - expected) < 0.01, f"Expected {expected}, got {computed}"

    for computed, expected in zip(rouge_scores_rouge2_precision, expected_scores_rouge2_precision):
        assert abs(computed - expected) < 0.01, f"Expected {expected}, got {computed}"


def test_rouge_score_evaluator_empty_strings():
    """
    Test the RougeScoreEvaluator with empty strings to ensure it handles edge cases gracefully.
    """
    # Sample data
    ground_truth_answers = [
        "",
        "Deep learning models require vast amounts of data.",
    ]
    answers = [
        "",  # Both empty
        "",  # Answer is empty
    ]

    # Initialize RougeScoreEvaluator with default settings
    rouge_evaluator = RougeScoreEvaluator()

    # Run evaluator
    rouge_scores = rouge_evaluator.run(ground_truth_answers=ground_truth_answers, answers=answers)

    expected_scores = [0.0, 0.0]  # Replace with actual printed scores

    for computed, expected in zip(rouge_scores, expected_scores):
        assert abs(computed - expected) < 0.01, f"Expected {expected}, got {computed}"
