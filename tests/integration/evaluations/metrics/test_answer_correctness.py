from unittest.mock import MagicMock

from dynamiq.evaluations.metrics.answer_correctness import AnswerCorrectnessEvaluator


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
            "The boiling point of water is 100 degrees Celsius (212°F) at sea level. "
            "The boiling point can change with altitude."
        ),
    ]

    evaluator = AnswerCorrectnessEvaluator(llm=openai_node)

    # Mock extraction: each question calls extract_statements twice:
    # For question 1: first call for the answer, second call for the ground truth.
    # For question 2: similarly.
    evaluator._statement_extractor.run = MagicMock(
        side_effect=[
            # Q1: extraction for answer
            {
                "results": [
                    {
                        "statements": [
                            "The sun is powered by nuclear fission.",
                            "Its primary function is to provide light to the solar system.",
                        ]
                    }
                ]
            },
            # Q1: extraction for ground truth answer
            {
                "results": [
                    {
                        "statements": [
                            "The sun is powered by nuclear fusion.",
                            "Fusion releases energy.",
                            "The sun provides heat and light.",
                        ]
                    }
                ]
            },
            # Q2: extraction for answer
            {"results": [{"statements": ["The boiling point of water is 100 degrees Celsius at sea level."]}]},
            # Q2: extraction for ground truth answer
            {
                "results": [
                    {
                        "statements": [
                            "The boiling point of water is 100 degrees Celsius at sea level.",
                            "Boiling point can change with altitude.",
                        ]
                    }
                ]
            },
        ]
    )

    # Mock classification: for each candidate statement, the classifier's run method is called.
    # For Question 1:
    #   - Two calls for answer statements:
    #         Call 1 for "The sun is powered by nuclear fission." returns match: False.
    #         Call 2 for "Its primary function is to provide light to the solar system." returns match: True.
    #   - Three calls for ground truth statements:
    #         Call 1 for "The sun is powered by nuclear fusion." returns match: False.
    #         Call 2 for "Fusion releases energy." returns match: False.
    #         Call 3 for "The sun provides heat and light." returns match: True.
    #
    # For Question 2:
    #   - One call for answer statement returns match: True.
    #   - Two calls for ground truth statements:
    #         Call 1 for "The boiling point of water is 100 degrees Celsius at sea level." returns match: True.
    #         Call 2 for "Boiling point can change with altitude." returns match: False.
    evaluator._statement_classifier.run = MagicMock(
        side_effect=[
            # Q1, answer statement call 1:
            {"results": [{"match": False, "reasoning": "Mismatch: expected fusion but found fission."}]},
            # Q1, answer statement call 2:
            {"results": [{"match": True, "reasoning": "Supported: provides light."}]},
            # Q1, ground truth statement call 1:
            {"results": [{"match": False, "reasoning": "Mismatch: answer omits fusion."}]},
            # Q1, ground truth statement call 2:
            {"results": [{"match": False, "reasoning": "Answer does not mention energy release."}]},
            # Q1, ground truth statement call 3:
            {"results": [{"match": True, "reasoning": "Matches: answer supports the provision of light."}]},
            # Q2, answer statement call:
            {"results": [{"match": True, "reasoning": "Correct: statement matches."}]},
            # Q2, ground truth statement call 1:
            {"results": [{"match": True, "reasoning": "Exact match."}]},
            # Q2, ground truth statement call 2:
            {"results": [{"match": False, "reasoning": "Detail not mentioned in answer."}]},
        ]
    )

    # Run the evaluator
    correctness_scores = evaluator.run(
        questions=questions, answers=answers, ground_truth_answers=ground_truth_answers, verbose=False
    )

    # Explanation of expected metrics:
    #
    # For Question 1:
    #   Answer candidate classifications:
    #      - "The sun is powered by nuclear fission." => match: False
    #      - "Its primary function is to provide light to the solar system." => match: True
    #     => TP = 1, FP = 1, so Precision = 1/(1+1) = 0.50.
    #
    #   Ground truth candidate classifications (compared against answer text):
    #      - "The sun is powered by nuclear fusion." => match: False
    #      - "Fusion releases energy." => match: False
    #      - "The sun provides heat and light." => match: True
    #     => TP (present) = 1, FN = 2, so Recall = 1/(1+2) ≈ 0.33.
    #
    #   F1 Score = 2 * (0.5 * 0.33) / (0.5 + 0.33) ≈ 0.40.
    #
    # For Question 2:
    #   Answer candidate:
    #      - "The boiling point of water is 100 degrees Celsius at sea level." => match: True
    #     => TP = 1, FP = 0, so Precision = 1.0.
    #
    #   Ground truth candidate classifications:
    #      - "The boiling point of water is 100 degrees Celsius at sea level." => match: True
    #      - "Boiling point can change with altitude." => match: False
    #     => TP = 1, FN = 1, so Recall = 1/(1+1) = 0.50.
    #
    #   F1 Score = 2 * (1.0 * 0.50) / (1.0 + 0.50) ≈ 0.67.
    expected_scores = [0.40, 0.67]

    computed_scores = [result.score for result in correctness_scores.results]
    for computed, expected in zip(computed_scores, expected_scores):
        assert abs(computed - expected) < 0.01, f"Expected {expected}, got {computed}"
