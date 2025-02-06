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
        ("The Eiffel Tower is located in Berlin, Germany. It was constructed in 1889."),
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
            # First call (answer claim decomposition for 1st answer)
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
            # Second call (context claim decomposition for 1st context)
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
            # Third call (answer claim decomposition for 2nd answer)
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
            # Fourth call (context claim decomposition for 2nd context)
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
            # First call: Verify answer claims (1st answer) against 1st context (precision)
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
            # Second call: Verify context claims (1st context) against 1st answer (recall)
            {
                "results": [
                    {
                        "results": [
                            {
                                "claim": "Albert Einstein was a German-born theoretical physicist.",
                                "verdict": "1",
                                "reason": "The response asserts he was a German theoretical physicist.",
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
            # Third call: Verify answer claims (2nd answer) against 2nd context (precision)
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
            # Fourth call: Verify context claims (2nd context) against 2nd answer (recall)
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
                                "reason": "The response does not mention the construction start year.",
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
    run_output = evaluator.run(answers=answers, contexts=contexts, verbose=False)
    # Extract computed scores from output results
    correctness_scores = [res.score for res in run_output.results]

    # Expected scores based on the mocked data:
    # For first item: 2 out of 3 answer claims are supported => TP=2, FP=1.
    # For recall: 2 out of 2 context claims are supported => FN=0.
    # F1 = (2*precision*recall)/(precision+recall) = (2*0.667*1)/(0.667+1)=0.8 (approx).
    # For second item: 1 out of 2 answer claims supported => TP=1, FP=1.
    # For recall: 1 out of 3 context claims supported => FN=2.
    # F1 = (2*0.5*0.333)/(0.5+0.333)=0.4 (approx).
    expected_scores = [0.8, 0.4]

    # Assert that the correctness scores are as expected.
    for computed, expected in zip(correctness_scores, expected_scores):
        assert abs(computed - expected) < 0.01, f"Expected {expected}, got {computed}"


def test_factual_correctness_evaluator_with_list_of_lists(openai_node):
    """
    Test that the FactualCorrectnessEvaluator can handle contexts passed as list[list[str]].
    The evaluator should join each sub-list into a single string and proceed as normal.
    """
    answers = [
        (
            "Albert Einstein was a German theoretical physicist. "
            "He developed the theory of relativity and contributed "
            "to quantum mechanics."
        ),
        ("The Eiffel Tower is located in Berlin, Germany. It was constructed in 1889."),
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

    # Initialize evaluator
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
                                "reason": "Reference mentions he was German-born.",
                            },
                            {
                                "claim": "He developed the theory of relativity.",
                                "verdict": "1",
                                "reason": "Explicitly mentioned in the reference text.",
                            },
                            {
                                "claim": "He contributed to quantum mechanics.",
                                "verdict": "0",
                                "reason": "Reference does not mention quantum mechanics.",
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
                                "reason": "Response states 'German theoretical physicist'.",
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
                                "reason": "Reference states it's in Paris, France.",
                            },
                            {
                                "claim": "It was constructed in 1889.",
                                "verdict": "1",
                                "reason": "Reference mentions it opened in 1889.",
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
                                "reason": "Response states Berlin, Germany.",
                            },
                            {
                                "claim": "It was constructed in 1887.",
                                "verdict": "0",
                                "reason": "Response does not mention 1887.",
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

    # Run the evaluator with contexts passed as list[list[str]]
    run_output = evaluator.run(answers=answers, contexts=contexts, verbose=False)
    correctness_scores = [res.score for res in run_output.results]

    # Expected scores (same as before): [0.8, 0.4]
    expected_scores = [0.8, 0.4]

    for computed, expected in zip(correctness_scores, expected_scores):
        assert abs(computed - expected) < 0.01, f"Expected {expected}, got {computed}"
