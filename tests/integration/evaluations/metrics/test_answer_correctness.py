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
    ground_truth_answers = [
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
    # The first two calls to _statement_extractor.run are for extracting from answers and ground truths
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

    # Mock the _statement_classifier.run method to return predefined classifications
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

    # Mock the _similarity_evaluator.run method to return predefined similarity scores
    evaluator._similarity_evaluator.run = MagicMock(
        return_value={"results": [{"similarity_score": "0.7"}, {"similarity_score": "1.0"}]}
    )

    # Run the evaluator
    correctness_scores = evaluator.run(
        questions=questions, answers=answers, ground_truth_answers=ground_truth_answers, verbose=False
    )

    # Expected scores based on the mocked data and computations
    # Calculation:
    # For the first item:
    #   F1 Score:
    #     TP = 1
    #     FP = 1
    #     FN = 3
    #     Precision = 1 / (1 + 1) = 0.5
    #     Recall = 1 / (1 + 3) = 0.25
    #     F1 = 2 * (0.5 * 0.25) / (0.5 + 0.25) = 0.333...
    #   Similarity Score = 0.7
    #   Final Score = 0.75 * 0.333... + 0.25 * 0.7 ≈ 0.25 + 0.175 = 0.425

    # For the second item:
    #   F1 Score:
    #     TP = 1
    #     FP = 0
    #     FN = 1
    #     Precision = 1 / (1 + 0) = 1.0
    #     Recall = 1 / (1 + 1) = 0.5
    #     F1 = 2 * (1.0 * 0.5) / (1.0 + 0.5) = 0.666...
    #   Similarity Score = 1.0
    #   Final Score = 0.75 * 0.666... + 0.25 * 1.0 ≈ 0.5 + 0.25 = 0.75

    expected_scores = [0.425, 0.75]  # Pre-calculated expected scores

    # Assert that the correctness scores are as expected
    for computed, expected in zip(correctness_scores, expected_scores):
        assert abs(computed - expected) < 0.01, f"Expected {expected}, got {computed}"
