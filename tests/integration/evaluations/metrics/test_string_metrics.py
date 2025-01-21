from dynamiq.evaluations.metrics import (
    DistanceMeasure,
    ExactMatchEvaluator,
    StringPresenceEvaluator,
    StringSimilarityEvaluator,
)


def test_exact_match_evaluator():
    """
    Test the ExactMatchEvaluator to ensure it correctly identifies exact matches between
    ground truth answers and answers.
    """
    # Sample data
    ground_truth_answers = [
        "The cat sits on the mat.",
        "A quick brown fox jumps over the lazy dog.",
        "Python is a versatile programming language.",
    ]
    answers = [
        "The cat sits on the mat.",  # Exact match
        "A quick brown fox jumps over the lazy dog.",  # Exact match
        "Python is a versatile programming language.",  # Exact match
    ]

    # Initialize evaluator
    evaluator = ExactMatchEvaluator()

    # Run evaluator
    exact_match_scores = evaluator.run(ground_truth_answers=ground_truth_answers, answers=answers)

    # Expected scores
    expected_scores = [1.0, 1.0, 1.0]

    # Assert
    assert exact_match_scores == expected_scores, f"Expected {expected_scores}, got {exact_match_scores}"


def test_exact_match_evaluator_partial_matches():
    """
    Test the ExactMatchEvaluator with partial matches to ensure it correctly
    identifies non-exact matches.
    """
    # Sample data
    ground_truth_answers = [
        "The cat sits on the mat.",
        "A quick brown fox jumps over the lazy dog.",
        "Python is a versatile programming language.",
    ]
    answers = [
        "The cat is lounging on the mat.",  # Not an exact match
        "A quick brown fox jumps over the lazy dog.",  # Exact match
        "Python is a powerful programming language.",  # Not an exact match
    ]

    # Initialize evaluator
    evaluator = ExactMatchEvaluator()

    # Run evaluator
    exact_match_scores = evaluator.run(ground_truth_answers=ground_truth_answers, answers=answers)

    # Expected scores
    expected_scores = [0.0, 1.0, 0.0]

    # Assert
    assert exact_match_scores == expected_scores, f"Expected {expected_scores}, got {exact_match_scores}"


def test_string_presence_evaluator():
    """
    Test the StringPresenceEvaluator to ensure it correctly identifies whether the
    ground truth answers are substrings within the answers.
    """
    # Sample data
    ground_truth_answers = [
        "cat sits on the mat",
        "brown fox",
        "versatile programming",
    ]
    answers = [
        "The cat sits on the mat.",  # 'cat sits on the mat' is present
        "A fast brown fox leaps over the dog.",  # 'brown fox' is present
        "Python is a versatile language.",  # 'versatile programming' not present
    ]

    # Initialize evaluator
    evaluator = StringPresenceEvaluator()

    # Run evaluator
    presence_scores = evaluator.run(ground_truth_answers=ground_truth_answers, answers=answers)

    # Expected scores
    expected_scores = [1.0, 1.0, 0.0]

    # Assert
    assert presence_scores == expected_scores, f"Expected {expected_scores}, got {presence_scores}"


def test_string_presence_evaluator_no_presence():
    """
    Test the StringPresenceEvaluator to ensure it correctly identifies when
    ground truth answers are not substrings within the answers.
    """
    # Sample data
    ground_truth_answers = [
        "hello world",
        "nonexistent substring",
    ]
    answers = [
        "The quick brown fox jumps over the lazy dog.",
        "Python is a versatile programming language.",
    ]

    # Initialize evaluator
    evaluator = StringPresenceEvaluator()

    # Run evaluator
    presence_scores = evaluator.run(ground_truth_answers=ground_truth_answers, answers=answers)

    # Expected scores
    expected_scores = [0.0, 0.0]

    # Assert
    assert presence_scores == expected_scores, f"Expected {expected_scores}, got {presence_scores}"


def test_string_similarity_evaluator_levenshtein():
    """
    Test the StringSimilarityEvaluator using the Levenshtein distance measure to ensure
    it correctly computes similarity scores between ground truth answers and answers.
    """
    # Sample data
    ground_truth_answers = [
        "The cat sits on the mat.",
        "A quick brown fox jumps over the lazy dog.",
        "Python is a versatile programming language.",
    ]
    answers = [
        "The cat sits on the mat.",  # Similarity 1.0
        "A fast brown fox leaps over the lazy dog.",  # Similarity <1
        "Python is a powerful programming language.",  # Similarity <1
    ]

    # Initialize evaluator with Levenshtein distance
    evaluator = StringSimilarityEvaluator(distance_measure=DistanceMeasure.LEVENSHTEIN)

    # Run evaluator
    similarity_scores = evaluator.run(ground_truth_answers=ground_truth_answers, answers=answers)

    expected_scores = [1.0, 0.81, 0.81]
    for computed, expected in zip(similarity_scores, expected_scores):
        assert abs(computed - expected) < 0.01, f"Expected {expected}, got {computed}"


def test_string_similarity_evaluator_jaro_winkler():
    """
    Test the StringSimilarityEvaluator using the Jaro-Winkler distance measure to ensure
    it correctly computes similarity scores between ground truth answers and answers.
    """
    # Sample data
    ground_truth_answers = [
        "The cat sits on the mat.",
        "A quick brown fox jumps over the lazy dog.",
        "Python is a versatile programming language.",
    ]
    answers = [
        "The cat sits on the mat.",  # Similarity 1.0
        "A fast brown fox leaps over the lazy dog.",  # Similarity <1
        "Python is a powerful programming language.",  # Similarity <1
    ]

    # Initialize evaluator with Jaro-Winkler distance
    evaluator = StringSimilarityEvaluator(distance_measure=DistanceMeasure.JARO_WINKLER)

    # Run evaluator
    similarity_scores = evaluator.run(ground_truth_answers=ground_truth_answers, answers=answers)

    expected_scores = [1.0, 0.81, 0.89]
    for computed, expected in zip(similarity_scores, expected_scores):
        assert abs(computed - expected) < 0.01, f"Expected {expected}, got {computed}"
