from unittest.mock import MagicMock

from dynamiq.evaluations.metrics import FactualCorrectnessEvaluator


def test_factual_correctness_evaluator(openai_node):
    # Sample data
    answers = [
        (
            "Albert Einstein was a German theoretical physicist. "
            "He developed the theory of relativity and contributed "
            "to quantum mechanics."
        ),
        ("The Eiffel Tower is located in Berlin, Germany. " "It was constructed in 1889."),
    ]
    contexts = [
        ("Albert Einstein was a German-born theoretical physicist. " "He developed the theory of relativity."),
        ("The Eiffel Tower is located in Paris, France. " "It was constructed in 1887 and opened in 1889."),
    ]

    # Initialize evaluator with the provided openai_node
    evaluator = FactualCorrectnessEvaluator(llm=openai_node)

    # Mock the run method of the claim decomposer
    evaluator._claim_decomposer.run = MagicMock(
        side_effect=[
            # First call to decompose claims from responses
            {
                "results": [
                    {
                        "claims": [
                            "Albert Einstein was a German theoretical physicist.",
                            "He developed the theory of relativity.",
                            "He contributed to quantum mechanics.",
                        ]
                    }
                ]
            },
            # Second call to decompose claims from references
            {
                "results": [
                    {
                        "claims": [
                            "Albert Einstein was a German-born theoretical physicist.",
                            "He developed the theory of relativity.",
                        ]
                    }
                ]
            },
            # Third call to decompose claims from responses
            {
                "results": [
                    {"claims": ["The Eiffel Tower is located in Berlin, Germany.", "It was constructed in 1889."]}
                ]
            },
            # Fourth call to decompose claims from references
            {
                "results": [
                    {
                        "claims": [
                            "The Eiffel Tower is located in Paris, France.",
                            "It was constructed in 1887.",
                            "It opened in 1889.",
                        ]
                    }
                ]
            },
        ]
    )

    # Mock the run method of the NLI evaluator
    evaluator._nli_evaluator.run = MagicMock(
        side_effect=[
            # First call: Verify response claims against reference (precision)
            {
                "results": [
                    {
                        "results": [
                            {
                                "claim": "Albert Einstein was a German theoretical physicist.",
                                "verdict": "1",
                                "reason": "The reference mentions he was a German-born theoretical physicist.",
                            },
                            {
                                "claim": "He developed the theory of relativity.",
                                "verdict": "1",
                                "reason": "This is explicitly mentioned in the reference.",
                            },
                            {
                                "claim": "He contributed to quantum mechanics.",
                                "verdict": "0",
                                "reason": "The reference does not mention quantum mechanics.",
                            },
                        ]
                    }
                ]
            },
            # Second call: Verify reference claims against response (recall)
            {
                "results": [
                    {
                        "results": [
                            {
                                "claim": "Albert Einstein was a German-born theoretical physicist.",
                                "verdict": "1",
                                "reason": "The response claims he was a German theoretical physicist.",
                            },
                            {
                                "claim": "He developed the theory of relativity.",
                                "verdict": "1",
                                "reason": "This is mentioned in the response.",
                            },
                        ]
                    }
                ]
            },
            # Third call: Verify response claims against reference (precision)
            {
                "results": [
                    {
                        "results": [
                            {
                                "claim": "The Eiffel Tower is located in Berlin, Germany.",
                                "verdict": "0",
                                "reason": "The reference states it is in Paris, France.",
                            },
                            {
                                "claim": "It was constructed in 1889.",
                                "verdict": "1",
                                "reason": "The reference mentions it opened in 1889.",
                            },
                        ]
                    }
                ]
            },
            # Fourth call: Verify reference claims against response (recall)
            {
                "results": [
                    {
                        "results": [
                            {
                                "claim": "The Eiffel Tower is located in Paris, France.",
                                "verdict": "0",
                                "reason": "The response states it is in Berlin, Germany.",
                            },
                            {
                                "claim": "It was constructed in 1887.",
                                "verdict": "0",
                                "reason": "The response does not mention construction start year.",
                            },
                            {
                                "claim": "It opened in 1889.",
                                "verdict": "1",
                                "reason": "The response mentions it was constructed in 1889.",
                            },
                        ]
                    }
                ]
            },
        ]
    )

    # Run the evaluator
    correctness_scores = evaluator.run(answers=answers, contexts=contexts, verbose=False)

    # Expected scores based on the mocked data
    expected_scores = [0.8, 0.4]  # For the first item  # For the second item

    # Assert that the correctness scores are as expected
    for computed, expected in zip(correctness_scores, expected_scores):
        assert abs(computed - expected) < 0.01, f"Expected {expected}, got {computed}"


def test_factual_correctness_evaluator_with_list_of_lists(openai_node):
    """
    Test that the FactualCorrectnessEvaluator can handle contexts passed as list[list[str]].
    It should join each sub-list into a single string under the hood, then proceed as normal.
    """

    # Same answers as in the original test
    answers = [
        (
            "Albert Einstein was a German theoretical physicist. "
            "He developed the theory of relativity and contributed "
            "to quantum mechanics."
        ),
        ("The Eiffel Tower is located in Berlin, Germany. " "It was constructed in 1889."),
    ]

    # contexts passed as list[list[str]] instead of list[str]
    contexts = [
        [
            "Albert Einstein was a German-born theoretical physicist.",
            "He developed the theory of relativity.",
        ],
        [
            "The Eiffel Tower is located in Paris, France.",
            "It was constructed in 1887 and opened in 1889.",
        ],
    ]

    # Initialize evaluator with the provided openai_node
    evaluator = FactualCorrectnessEvaluator(llm=openai_node)

    # Mock the run method of the claim decomposer (decompose_claims)
    evaluator._claim_decomposer.run = MagicMock(
        side_effect=[
            # 1) Decompose claims from first response
            {
                "results": [
                    {
                        "claims": [
                            "Albert Einstein was a German theoretical physicist.",
                            "He developed the theory of relativity.",
                            "He contributed to quantum mechanics.",
                        ]
                    }
                ]
            },
            # 2) Decompose claims from first context (joined from sub-list)
            {
                "results": [
                    {
                        "claims": [
                            "Albert Einstein was a German-born theoretical physicist.",
                            "He developed the theory of relativity.",
                        ]
                    }
                ]
            },
            # 3) Decompose claims from second response
            {
                "results": [
                    {
                        "claims": [
                            "The Eiffel Tower is located in Berlin, Germany.",
                            "It was constructed in 1889.",
                        ]
                    }
                ]
            },
            # 4) Decompose claims from second context (joined from sub-list)
            {
                "results": [
                    {
                        "claims": [
                            "The Eiffel Tower is located in Paris, France.",
                            "It was constructed in 1887.",
                            "It opened in 1889.",
                        ]
                    }
                ]
            },
        ]
    )

    # Mock the run method of the NLI evaluator (verify_claims)
    evaluator._nli_evaluator.run = MagicMock(
        side_effect=[
            # 1) Verify first response claims against first context (precision)
            {
                "results": [
                    {
                        "results": [
                            {
                                "claim": "Albert Einstein was a German theoretical physicist.",
                                "verdict": "1",
                                "reason": "The reference mentions he was German-born.",
                            },
                            {
                                "claim": "He developed the theory of relativity.",
                                "verdict": "1",
                                "reason": "Explicitly mentioned in the reference text.",
                            },
                            {
                                "claim": "He contributed to quantum mechanics.",
                                "verdict": "0",
                                "reason": "The reference doesn't mention quantum mechanics.",
                            },
                        ]
                    }
                ]
            },
            # 2) Verify first context claims against first response (recall)
            {
                "results": [
                    {
                        "results": [
                            {
                                "claim": "Albert Einstein was a German-born theoretical physicist.",
                                "verdict": "1",
                                "reason": "The response says 'German theoretical physicist'.",
                            },
                            {
                                "claim": "He developed the theory of relativity.",
                                "verdict": "1",
                                "reason": "Mentioned in the response.",
                            },
                        ]
                    }
                ]
            },
            # 3) Verify second response claims against second context (precision)
            {
                "results": [
                    {
                        "results": [
                            {
                                "claim": "The Eiffel Tower is located in Berlin, Germany.",
                                "verdict": "0",
                                "reason": "Reference says it's in Paris, France.",
                            },
                            {
                                "claim": "It was constructed in 1889.",
                                "verdict": "1",
                                "reason": "Reference mentions it opened in 1889 (close enough).",
                            },
                        ]
                    }
                ]
            },
            # 4) Verify second context claims against second response (recall)
            {
                "results": [
                    {
                        "results": [
                            {
                                "claim": "The Eiffel Tower is located in Paris, France.",
                                "verdict": "0",
                                "reason": "Response says Berlin, Germany.",
                            },
                            {
                                "claim": "It was constructed in 1887.",
                                "verdict": "0",
                                "reason": "Response doesn't mention 1887 start date.",
                            },
                            {
                                "claim": "It opened in 1889.",
                                "verdict": "1",
                                "reason": "Response says it was constructed in 1889.",
                            },
                        ]
                    }
                ]
            },
        ]
    )

    # Run the evaluator
    correctness_scores = evaluator.run(answers=answers, contexts=contexts, verbose=False)

    # Expected scores based on the mocked data
    expected_scores = [0.8, 0.4]  # same as in the original test

    # Assert that the correctness scores match our expectations
    for computed, expected in zip(correctness_scores, expected_scores):
        assert abs(computed - expected) < 0.01, f"Expected {expected}, got {computed}"
