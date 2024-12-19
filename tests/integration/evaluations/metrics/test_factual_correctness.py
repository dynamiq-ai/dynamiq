from unittest.mock import MagicMock

from dynamiq.evaluations.metrics import FactualCorrectnessEvaluator


def test_factual_correctness_evaluator(openai_node):
    # Sample data
    answer = [
        (
            "Albert Einstein was a German theoretical physicist. "
            "He developed the theory of relativity and contributed "
            "to quantum mechanics."
        ),
        ("The Eiffel Tower is located in Berlin, Germany. " "It was constructed in 1889."),
    ]
    context = [
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
    correctness_scores = evaluator.run(answer=answer, context=context, verbose=False)

    # Expected scores based on the mocked data
    expected_scores = [0.8, 0.4]  # For the first item  # For the second item

    # Assert that the correctness scores are as expected
    for computed, expected in zip(correctness_scores, expected_scores):
        assert abs(computed - expected) < 0.01, f"Expected {expected}, got {computed}"
