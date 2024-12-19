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
            # First call to decompose claims from answers (First answer)
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
            # Second call to decompose claims from contexts (First context)
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
            # Third call to decompose claims from answers (Second answer)
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
            # Fourth call to decompose claims from contexts (Second context)
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
            # First call: Verify answer claims against context (precision for first pair)
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
            # Second call: Verify context claims against answer (recall for first pair)
            {
                "results": [
                    {
                        "results": [
                            {
                                "claim": "Albert Einstein was a German-born theoretical physicist.",
                                "verdict": "1",
                                "reason": "The answer states he was a German theoretical physicist.",
                            },
                            {
                                "claim": "He developed the theory of relativity.",
                                "verdict": "1",
                                "reason": "This is mentioned in the answer.",
                            },
                        ]
                    }
                ]
            },
            # Third call: Verify answer claims against context (precision for second pair)
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
            # Fourth call: Verify context claims against answer (recall for second pair)
            {
                "results": [
                    {
                        "results": [
                            {
                                "claim": "The Eiffel Tower is located in Paris, France.",
                                "verdict": "0",
                                "reason": "The answer states it is in Berlin, Germany.",
                            },
                            {
                                "claim": "It was constructed in 1887.",
                                "verdict": "0",
                                "reason": "The answer does not mention construction start year.",
                            },
                            {
                                "claim": "It opened in 1889.",
                                "verdict": "1",
                                "reason": "The answer mentions it was constructed in 1889.",
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
    expected_scores = [
        0.75,  # For the first pair: 3 out of 4 statements attributed
        0.3333,  # For the second pair: 1 out of 3 statements attributed
    ]

    # Assert that the correctness scores are as expected
    for computed, expected in zip(correctness_scores, expected_scores):
        assert abs(computed - expected) < 0.01, f"Expected {expected}, got {computed}"
