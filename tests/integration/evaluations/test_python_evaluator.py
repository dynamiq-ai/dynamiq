import pytest

from dynamiq.evaluations import PythonEvaluator


def test_python_evaluator_perfect_matches():
    user_code = """
def evaluate(answer, expected):
    return 1.0 if answer == expected else 0.0
"""
    input_data_list = [
        {"answer": "Paris", "expected": "Paris"},
        {"answer": "Berlin", "expected": "Berlin"},
        {"answer": "Madrid", "expected": "Madrid"},
    ]
    evaluator = PythonEvaluator(code=user_code)
    scores = evaluator.run(input_data_list=input_data_list)
    expected_scores = [1.0, 1.0, 1.0]
    for computed, expected in zip(scores, expected_scores):
        assert abs(computed - expected) < 0.01, f"Expected {expected}, got {computed}"


def test_python_evaluator_partial_matches():
    user_code = """
def evaluate(answer, expected):
    return 1.0 if answer == expected else 0.0
"""
    input_data_list = [
        {"answer": "Paris", "expected": "Paris"},
        {"answer": "Berlin", "expected": "Berlin"},
        {"answer": "Madrid", "expected": "Barcelona"},
    ]
    evaluator = PythonEvaluator(code=user_code)
    scores = evaluator.run(input_data_list=input_data_list)
    expected_scores = [1.0, 1.0, 0.0]
    for computed, expected in zip(scores, expected_scores):
        assert abs(computed - expected) < 0.01, f"Expected {expected}, got {computed}"


def test_python_evaluator_empty_strings():
    user_code = """
def evaluate(answer, expected):
    return 1.0 if answer == expected else 0.0
"""
    input_data_list = [
        {"answer": "", "expected": ""},
        {"answer": "New York", "expected": ""},
        {"answer": "", "expected": "Los Angeles"},
    ]
    evaluator = PythonEvaluator(code=user_code)
    scores = evaluator.run(input_data_list=input_data_list)
    expected_scores = [1.0, 0.0, 0.0]
    for computed, expected in zip(scores, expected_scores):
        assert abs(computed - expected) < 0.01, f"Expected {expected}, got {computed}"


def test_python_evaluator_missing_keys():
    user_code = """
def evaluate(answer, expected):
    return 1.0 if answer == expected else 0.0
"""
    # Missing the "expected" key.
    input_data = {"answer": "Paris"}
    evaluator = PythonEvaluator(code=user_code)
    with pytest.raises(ValueError, match="Missing required keys"):
        evaluator.run_single(input_data)


def test_python_evaluator_extra_keys():
    user_code = """
def evaluate(answer, expected):
    return 1.0 if answer == expected else 0.0
"""
    # Extra key "extra"
    input_data = {"answer": "Paris", "expected": "Paris", "extra": "unexpected"}
    evaluator = PythonEvaluator(code=user_code)
    with pytest.raises(ValueError, match="Unexpected keys provided"):
        evaluator.run_single(input_data)


def test_python_evaluator_default_parameters():
    user_code = """
def evaluate(answer, expected=42):
    return 1.0 if answer == expected else 0.0
"""
    # "expected" is optional.
    input_data = {"answer": 42}
    evaluator = PythonEvaluator(code=user_code)
    score = evaluator.run_single(input_data)
    assert abs(score - 1.0) < 0.01, f"Expected score 1.0, got {score}"


def test_python_evaluator_non_numeric():
    user_code = """
def evaluate(answer, expected):
    return "invalid"
"""
    input_data = {"answer": "Paris", "expected": "Paris"}
    evaluator = PythonEvaluator(code=user_code)
    with pytest.raises(ValueError, match="non-numeric"):
        evaluator.run_single(input_data)
