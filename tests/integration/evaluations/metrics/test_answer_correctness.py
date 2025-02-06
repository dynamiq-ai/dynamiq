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
            "The boiling point of water is 100 degrees Celsius (212°F) at sea level. "
            "The boiling point can change with altitude."
        ),
    ]

    # Initialize evaluator with the provided openai_node
    evaluator = AnswerCorrectnessEvaluator(llm=openai_node)

    # Mock the LLMEvaluator's run method for statement extraction.
    # The first call returns extracted statements from the answers,
    # The second call returns extracted statements from the ground truth answers.
    evaluator._statement_extractor.run = MagicMock(
        side_effect=[
            # Extraction for answers (for two questions)
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
            # Extraction for ground truth answers (for two questions)
            {
                "results": [
                    {
                        "statements": [
                            "The sun is powered by nuclear fusion.",
                            "Fusion releases energy.",
                            "The sun provides heat and light.",
                        ]
                    },
                    {
                        "statements": [
                            "The boiling point of water is 100 degrees Celsius at sea level.",
                            "Boiling point can change with altitude.",
                        ]
                    },
                ]
            },
        ]
    )

    # For each candidate statement, the classification evaluator is called.
    # In our new implementation each call returns a dict with "results" list containing 'match'
    # and 'reasoning' keys.
    # For question 1, there are 2 answer statements and 3 ground truth statements.
    # For question 2, there is 1 answer statement and 2 ground truth statements.
    evaluator._statement_classifier.run = MagicMock(
        side_effect=[
            # Q1, answer statement call 1:
            {"results": [{"match": False, "reasoning": "Mismatch: expected fusion but found fission."}]},
            # Q1, answer statement call 2:
            {"results": [{"match": True, "reasoning": "The statement supports the core fact of providing light."}]},
            # Q1, ground truth statement call 1:
            {"results": [{"match": False, "reasoning": "Mismatch: the answer does not mention nuclear fusion."}]},
            # Q1, ground truth statement call 2:
            {"results": [{"match": False, "reasoning": "The answer does not mention energy release."}]},
            # Q1, ground truth statement call 3:
            {"results": [{"match": True, "reasoning": "Matches: the answer supports the provision of light."}]},
            # Q2, answer statement call:
            {"results": [{"match": True, "reasoning": "The statement matches the core fact."}]},
            # Q2, ground truth statement call 1:
            {"results": [{"match": True, "reasoning": "Matches: the statement is exactly present in the answer."}]},
            # Q2, ground truth statement call 2:
            {"results": [{"match": False, "reasoning": "This detail is not mentioned in the answer."}]},
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
    #     => TP (supported) = 1, FP = 1, Precision = 1/(1+1) = 0.5.
    #
    #   Ground truth candidate classifications (compared against answer text):
    #      - "The sun is powered by nuclear fusion." => match: False
    #      - "Fusion releases energy." => match: False
    #      - "The sun provides heat and light." => match: True
    #     => TP (present) = 1, FN = 2, Recall = 1/(1+2) ≈ 0.33.
    #
    #   F1 Score = 2*(0.5*0.33)/(0.5+0.33) ≈ 0.4.
    #
    # For Question 2:
    #   Answer candidate: "The boiling point of water is 100 degrees Celsius at sea level."
    #      => match: True, so TP = 1, FP = 0, Precision = 1.0.
    #
    #   Ground truth candidates:
    #      - "The boiling point of water is 100 degrees Celsius at sea level." => match: True
    #      - "Boiling point can change with altitude." => match: False
    #     => TP (present) = 1, FN = 1, Recall = 1/(1+1) = 0.5.
    #
    #   F1 Score = 2*(1*0.5)/(1+0.5) = 0.67 (approximately).
    #
    # Therefore, expected final scores are approximately [0.4, 0.67].
    expected_scores = [0.4, 0.67]

    for computed, expected in zip([result.score for result in correctness_scores.results], expected_scores):
        assert abs(computed - expected) < 0.01, f"Expected {expected}, got {computed}"
