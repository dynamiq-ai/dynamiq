import time
from concurrent.futures import ThreadPoolExecutor

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


@pytest.mark.parametrize(
    "user_code",
    [
        "x = 1\n",
        "def evaluate_answer(answer):\n    return 1.0\n",
    ],
)
def test_python_evaluator_rejects_code_without_evaluate(user_code):
    with pytest.raises(ValueError):
        PythonEvaluator(code=user_code)


def test_python_evaluator_rejects_code_that_fails_to_compile():
    with pytest.raises(ValueError, match="compilation error"):
        PythonEvaluator(code="def evaluate(answer)\n    return 1.0\n")


def test_python_evaluator_stops_code_past_its_time_limit():
    user_code = """
def evaluate(answer):
    while True:
        pass
"""
    evaluator = PythonEvaluator(code=user_code, timeout_seconds=1)
    started = time.monotonic()
    with pytest.raises(ValueError, match="time limit exceeded"):
        evaluator.run_single({"answer": "Paris"})
    assert time.monotonic() - started < 5


def test_python_evaluator_stops_code_past_its_memory_limit():
    user_code = """
def evaluate(answer):
    return len(answer * (10 ** 9))
"""
    evaluator = PythonEvaluator(code=user_code, max_memory_bytes=16 * 1024 * 1024)
    with pytest.raises(ValueError, match="memory limit exceeded"):
        evaluator.run_single({"answer": "Paris"})


def test_python_evaluator_cannot_read_host_environment(monkeypatch):
    monkeypatch.setenv("DYNAMIQ_EVALUATOR_TEST_SECRET", "s3cret")
    user_code = """
import os
def evaluate(answer):
    return float(len(os.getenv("DYNAMIQ_EVALUATOR_TEST_SECRET")))
"""
    evaluator = PythonEvaluator(code=user_code)
    with pytest.raises(ValueError) as excinfo:
        evaluator.run_single({"answer": "Paris"})
    assert "s3cret" not in str(excinfo.value)


def test_python_evaluator_survives_code_that_overflows_a_native_stack():
    # Serializing a deeply nested value segfaulted the host process on pydantic-monty 0.0.16, which ran
    # the code in-process.
    user_code = """
import json
def evaluate(answer):
    nested = []
    for _ in range(200000):
        nested = [nested]
    return float(len(json.dumps(nested)))
"""
    evaluator = PythonEvaluator(code=user_code)
    with pytest.raises(ValueError):
        evaluator.run_single({"answer": "Paris"})
    assert PythonEvaluator(code="def evaluate(answer):\n    return 1.0\n").run_single({"answer": "Paris"}) == 1.0


def test_python_evaluator_runs_each_input_in_a_fresh_interpreter():
    user_code = """
calls = []
def evaluate(answer):
    calls.append(answer)
    return float(len(calls))
"""
    evaluator = PythonEvaluator(code=user_code)
    assert evaluator.run([{"answer": "Paris"}, {"answer": "Berlin"}, {"answer": "Madrid"}]) == [1.0, 1.0, 1.0]


def test_python_evaluator_concurrent_runs_from_threads():
    evaluator = PythonEvaluator(code="def evaluate(a, b):\n    return float(a * b)\n")
    pairs = [(a, b) for a in range(8) for b in range(8)]
    with ThreadPoolExecutor(max_workers=8) as executor:
        scores = list(executor.map(lambda pair: evaluator.run_single({"a": pair[0], "b": pair[1]}), pairs))
    assert scores == [float(a * b) for a, b in pairs]
