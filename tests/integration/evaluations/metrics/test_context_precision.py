from unittest.mock import MagicMock

from dynamiq.evaluations.metrics import ContextPrecisionEvaluator


def test_context_precision_evaluator(openai_node):
    # Sample data
    questions = [
        "What can you tell me about Albert Einstein?",
        "Who won the 2020 ICC World Cup?",
        "What is the tallest mountain in the world?",
    ]
    answers = [
        (
            "Albert Einstein, born on 14 March 1879, was a German-born theoretical physicist, "
            "widely held to be one of the greatest and most influential scientists of all time. "
            "He received the 1921 Nobel Prize in Physics for his services to theoretical physics."
        ),
        "England",
        "Mount Everest.",
    ]
    context_list = [
        [
            # Contexts for the first question
            ("Albert Einstein was a German-born theoretical physicist. " "He developed the theory of relativity."),
            (
                "Albert Einstein's work is also known for its influence on the philosophy of "
                "science. His mass–energy equivalence formula E = mc^2 has been called 'the world's "
                "most famous equation'."
            ),
        ],
        [
            # Contexts for the second question
            (
                "The 2022 ICC Men's T20 World Cup, held from October 16 to November 13, 2022, in "
                "Australia, was the eighth edition of the tournament. Originally scheduled for "
                "2020, it was postponed due to the COVID-19 pandemic. England emerged victorious, "
                "defeating Pakistan by five wickets in the final to clinch their second ICC Men's "
                "T20 World Cup title."
            ),
            (
                "The 2016 ICC World Twenty20 was held in India from 8 March to 3 April 2016. The "
                "West Indies won the tournament, beating England in the final."
            ),
        ],
        [
            # Contexts for the third question
            (
                "The Andes is the longest continental mountain range in the world, located in "
                "South America. It features many high peaks but not the tallest in the world."
            ),
            ("Mount Kilimanjaro is the highest mountain in Africa, standing at 5,895 meters " "above sea level."),
        ],
    ]

    # Initialize evaluator with the provided openai_node
    evaluator = ContextPrecisionEvaluator(llm=openai_node)

    # Prepare the mocked results for each context evaluation
    mocked_run_results = [
        # First question, first context
        {
            "results": [
                {
                    "verdict": "1",
                    "reason": ("The context provides information about Albert Einstein relevant to the answer."),
                }
            ]
        },
        # First question, second context
        {
            "results": [
                {
                    "verdict": "1",
                    "reason": (
                        "The context includes details about Einstein's famous equation, which is "
                        "relevant to the answer."
                    ),
                }
            ]
        },
        # Second question, first context
        {
            "results": [
                {
                    "verdict": "1",
                    "reason": ("The context explains England won the tournament, which answers the question."),
                }
            ]
        },
        # Second question, second context
        {
            "results": [
                {
                    "verdict": "0",
                    "reason": ("The context is about the 2016 World Cup, not relevant to the 2020 World Cup."),
                }
            ]
        },
        # Third question, first context
        {
            "results": [
                {
                    "verdict": "0",
                    "reason": ("The context is about the Andes, not relevant to the tallest mountain."),
                }
            ]
        },
        # Third question, second context
        {
            "results": [
                {
                    "verdict": "0",
                    "reason": ("The context is about Mount Kilimanjaro, not the tallest mountain."),
                }
            ]
        },
    ]

    # Mock the run method of the evaluator's internal LLMEvaluator
    evaluator._context_precision_evaluator.run = MagicMock(side_effect=mocked_run_results)

    # Run the evaluator
    correctness_scores = evaluator.run(question=questions, answer=answers, context=context_list, verbose=False)

    # Expected scores based on the mocked data
    expected_scores = [
        1.0,  # For the first question: 2 out of 2 contexts marked as useful
        0.5,  # For the second question: 1 out of 2 contexts marked as useful
        0.0,  # For the third question: 0 out of 2 contexts marked as useful
    ]

    # Assert that the correctness scores are as expected
    for computed, expected in zip(correctness_scores, expected_scores):
        assert abs(computed - expected) < 0.01, f"Expected {expected}, got {computed}"
