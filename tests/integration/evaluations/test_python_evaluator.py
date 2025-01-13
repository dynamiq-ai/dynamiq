from dynamiq.evaluations import PythonEvaluator


def test_python_evaluator_perfect_matches():
    """
    Test the PythonEvaluator to ensure it correctly identifies perfect matches.
    """
    # Sample user-defined Python code
    user_code = """
def run(input_data):
    # Retrieve arbitrary values from input_data
    answer = input_data.get("answer")
    expected = input_data.get("expected")
    return 1.0 if answer == expected else 0.0
"""

    # Sample data: all answers match expected
    input_data_list = [
        {"answer": "Paris", "expected": "Paris"},
        {"answer": "Berlin", "expected": "Berlin"},
        {"answer": "Madrid", "expected": "Madrid"},
    ]

    # Initialize PythonEvaluator with user-defined code
    python_evaluator = PythonEvaluator(code=user_code)

    # Run evaluator on multiple data points
    python_scores = python_evaluator.run(input_data_list=input_data_list)

    # Print the Python scores
    print("Perfect Match Python Scores:", python_scores)

    # Expected scores based on the user-defined code
    expected_scores = [1.0, 1.0, 1.0]

    # Assert that the evaluator's scores match the expected scores
    for computed, expected in zip(python_scores, expected_scores):
        assert abs(computed - expected) < 0.01, f"Expected {expected}, got {computed}"


def test_python_evaluator_partial_matches():
    """
    Test the PythonEvaluator to ensure it correctly identifies partial matches.
    """
    # Sample user-defined Python code
    user_code = """
def run(input_data):
    # Retrieve arbitrary values from input_data
    answer = input_data.get("answer")
    expected = input_data.get("expected")
    return 1.0 if answer == expected else 0.0
"""

    # Sample data: some answers match expected, some do not
    input_data_list = [
        {"answer": "Paris", "expected": "Paris"},  # Match
        {"answer": "Berlin", "expected": "Berlin"},  # Match
        {"answer": "Madrid", "expected": "Barcelona"},  # No Match
    ]

    # Initialize PythonEvaluator with user-defined code
    python_evaluator = PythonEvaluator(code=user_code)

    # Run evaluator on multiple data points
    python_scores = python_evaluator.run(input_data_list=input_data_list)

    # Print the Python scores
    print("Partial Match Python Scores:", python_scores)

    # Expected scores based on the user-defined code
    expected_scores = [1.0, 1.0, 0.0]

    # Assert that the evaluator's scores match the expected scores
    for computed, expected in zip(python_scores, expected_scores):
        assert abs(computed - expected) < 0.01, f"Expected {expected}, got {computed}"


def test_python_evaluator_empty_strings():
    """
    Test the PythonEvaluator with empty strings to ensure it handles edge cases gracefully.
    """
    # Sample user-defined Python code
    user_code = """
def run(input_data):
    # Retrieve arbitrary values from input_data
    answer = input_data.get("answer")
    expected = input_data.get("expected")
    return 1.0 if answer == expected else 0.0
"""

    # Sample data: empty answers and expected
    input_data_list = [
        {"answer": "", "expected": ""},  # Match (both empty)
        {"answer": "New York", "expected": ""},  # No Match
        {"answer": "", "expected": "Los Angeles"},  # No Match
    ]

    # Initialize PythonEvaluator with user-defined code
    python_evaluator = PythonEvaluator(code=user_code)

    # Run evaluator on multiple data points
    python_scores = python_evaluator.run(input_data_list=input_data_list)

    # Print the Python scores
    print("Empty Strings Python Scores:", python_scores)

    # Expected scores based on the user-defined code
    expected_scores = [1.0, 0.0, 0.0]

    # Assert that the evaluator's scores match the expected scores
    for computed, expected in zip(python_scores, expected_scores):
        assert abs(computed - expected) < 0.01, f"Expected {expected}, got {computed}"
