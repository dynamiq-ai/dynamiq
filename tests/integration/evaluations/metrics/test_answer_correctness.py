from unittest.mock import MagicMock

from dynamiq.evaluations.metrics import AnswerCorrectnessEvaluator


def test_answer_correctness_evaluator(openai_node):
    # Sample data
    questions = ["What powers the sun and what is its primary function?", "What is the boiling point of water?"]
    answers = [
        (
            "The sun is powered by nuclear fission, similar to nuclear reactors on Earth."
            " Its primary function is to provide light to the solar system."
        ),
        "The boiling point of water is 100 degrees Celsius at sea level.",
    ]
    ground_truths = [
        (
            "The sun is powered by nuclear fusion, where hydrogen atoms fuse to form helium."
            " This fusion process releases a tremendous amount of energy. The sun provides"
            " heat and light, which are essential for life on Earth."
        ),
        (
            "The boiling point of water is 100 degrees Celsius (212 degrees Fahrenheit) at"
            " sea level. The boiling point can change with altitude."
        ),
    ]

    # Initialize evaluator with the provided openai_node
    evaluator = AnswerCorrectnessEvaluator(llm=openai_node)

    # Mock the LLMEvaluator's run methods
    evaluator._statement_extractor.run = MagicMock(
        side_effect=[
            # First call - extract statements from answers
            {
                "results": [
                    {
                        "statements": [
                            "The sun is powered by nuclear fission.",
                            "Its primary function is to provide light to the solar system.",
                        ]
                    },
                    {"statements": ["The boiling point of water is 100 degrees Celsius at sea level."]},
                ]
            },
            # Second call - extract statements from ground truths
            {
                "results": [
                    {
                        "statements": [
                            "The sun is powered by nuclear fusion.",
                            "This fusion process releases a tremendous amount of energy.",
                            "The sun provides heat and light, which are essential for life on Earth.",
                        ]
                    },
                    {
                        "statements": [
                            "The boiling point of water is 100 degrees Celsius at sea level.",
                            "The boiling point can change with altitude.",
                        ]
                    },
                ]
            },
        ]
    )

    evaluator._statement_classifier.run = MagicMock(
        return_value={
            "results": [
                {
                    "classifications": {
                        "TP": ["Its primary function is to provide light to the solar system."],
                        "FP": ["The sun is powered by nuclear fission."],
                        "FN": [
                            "The sun is powered by nuclear fusion.",
                            "This fusion process releases a tremendous amount of energy.",
                            "The sun provides heat and light, which are essential for life on Earth.",
                        ],
                    }
                },
                {
                    "classifications": {
                        "TP": ["The boiling point of water is 100 degrees Celsius at sea level."],
                        "FP": [],
                        "FN": ["The boiling point can change with altitude."],
                    }
                },
            ]
        }
    )

    evaluator._similarity_evaluator.run = MagicMock(
        return_value={"results": [{"similarity_score": "0.7"}, {"similarity_score": "1.0"}]}
    )

    # Run the evaluator
    correctness_scores = evaluator.run(questions=questions, answers=answers, ground_truths=ground_truths, verbose=False)

    # Expected scores based on the mocked data and computations
    expected_scores = [0.425, 0.75]  # Pre-calculated expected scores

    # Assert that the correctness scores are as expected
    for computed, expected in zip(correctness_scores, expected_scores):
        assert abs(computed - expected) < 0.01, f"Expected {expected}, got {computed}"
