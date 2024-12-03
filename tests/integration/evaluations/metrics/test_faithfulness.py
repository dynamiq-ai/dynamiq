from unittest.mock import MagicMock

from dynamiq.evaluations.metrics import FaithfulnessEvaluator


def test_faithfulness_evaluator(openai_node):
    # Sample data
    questions = ["Who was Albert Einstein and what is he best known for?", "Tell me about the Great Wall of China."]
    answers = [
        (
            "He was a German-born theoretical physicist, widely acknowledged to be one "
            "of the greatest and most influential physicists of all time. "
            "He was best known for developing the theory of relativity, he also made "
            "important contributions to the development of the theory of quantum mechanics."
        ),
        (
            "The Great Wall of China is a large wall in China. "
            "It was built to keep out invaders. "
            "It is visible from space."
        ),
    ]
    contexts_list = [
        [
            (
                "Albert Einstein (14 March 1879 - 18 April 1955) was a German-born "
                "theoretical physicist, widely held to be one of the greatest and "
                "most influential scientists of all time. Best known for developing "
                "the theory of relativity, he also made important contributions to "
                "quantum mechanics."
            )
        ],
        [
            (
                "The Great Wall of China is a series of fortifications that were built "
                "across the historical northern borders of ancient Chinese states and "
                "Imperial China as protection against various nomadic groups."
            )
        ],
    ]

    # Initialize evaluator with the provided openai_node
    evaluator = FaithfulnessEvaluator(llm=openai_node)

    # Mock the statement simplifier
    evaluator._statement_simplifier.run = MagicMock(
        side_effect=[
            # First call to simplify statements for the first answer
            {
                "results": [
                    {
                        "statements": [
                            "Albert Einstein was a German-born theoretical physicist.",
                            "He is widely acknowledged to be one of the greatest"
                            " and most influential physicists of all time.",
                            "He was best known for developing the theory of relativity.",
                            "He also made important contributions to the development"
                            " of the theory of quantum mechanics.",
                        ]
                    }
                ]
            },
            # Second call to simplify statements for the second answer
            {
                "results": [
                    {
                        "statements": [
                            "The Great Wall of China is a large wall in China.",
                            "It was built to keep out invaders.",
                            "It is visible from space.",
                        ]
                    }
                ]
            },
        ]
    )

    # Mock the NLI evaluator
    evaluator._nli_evaluator.run = MagicMock(
        side_effect=[
            # First call to check faithfulness for the first question
            {
                "results": [
                    {
                        "results": [
                            {
                                "statement": "Albert Einstein was a German-born theoretical physicist.",
                                "verdict": "1",
                                "reason": "This is mentioned in the context.",
                            },
                            {
                                "statement": "He is widely acknowledged to be one of the greatest"
                                " and most influential physicists of all time.",
                                "verdict": "1",
                                "reason": "This is mentioned in the context.",
                            },
                            {
                                "statement": "He was best known for developing the theory of relativity.",
                                "verdict": "1",
                                "reason": "This is mentioned in the context.",
                            },
                            {
                                "statement": "He also made important contributions to the development"
                                " of the theory of quantum mechanics.",
                                "verdict": "1",
                                "reason": "This is mentioned in the context.",
                            },
                        ]
                    }
                ]
            },
            # Second call to check faithfulness for the second question
            {
                "results": [
                    {
                        "results": [
                            {
                                "statement": "The Great Wall of China is a large wall in China.",
                                "verdict": "1",
                                "reason": "This is consistent with the context.",
                            },
                            {
                                "statement": "It was built to keep out invaders.",
                                "verdict": "1",
                                "reason": "This is mentioned in the context.",
                            },
                            {
                                "statement": "It is visible from space.",
                                "verdict": "0",
                                "reason": "The context does not mention this, and it is a common myth.",
                            },
                        ]
                    }
                ]
            },
        ]
    )

    # Run the evaluator
    faithfulness_scores = evaluator.run(
        questions=questions, answers=answers, contexts_list=contexts_list, verbose=False
    )

    # Expected scores based on the mocked data
    expected_scores = [
        1.0,  # All statements are faithful for the first question
        0.6667,  # 2 out of 3 statements are faithful for the second question
    ]

    # Assert that the faithfulness scores are as expected
    for computed, expected in zip(faithfulness_scores, expected_scores):
        assert abs(computed - expected) < 0.01, f"Expected {expected}, got {computed}"
