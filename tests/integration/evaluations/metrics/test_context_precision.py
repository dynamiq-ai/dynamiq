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
    contexts_list = [
        [
            # Contexts for the first question
            (
                "Albert Einstein (14 March 1879 – 18 April 1955) was a German-born theoretical "
                "physicist, widely held to be one of the greatest and most influential scientists "
                "of all time. Best known for developing the theory of relativity, he also made "
                "important contributions to quantum mechanics."
            ),
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

    evaluator = ContextPrecisionEvaluator(llm=openai_node)

    # Prepare mocked results for each context evaluation (6 results total)
    mocked_run_results = [
        # First question, first context
        {
            "results": [
                {
                    "verdict": "1",
                    "reason": "The context provides information about Albert Einstein relevant to the answer.",
                }
            ]
        },
        # First question, second context
        {
            "results": [
                {
                    "verdict": "1",
                    "reason": (
                        "The context includes details about Einstein's famous equation, "
                        "which is relevant to the answer."
                    ),
                }
            ]
        },
        # Second question, first context
        {
            "results": [
                {
                    "verdict": "1",
                    "reason": "The context explains England won the tournament, which answers the question.",
                }
            ]
        },
        # Second question, second context
        {
            "results": [
                {
                    "verdict": "0",
                    "reason": "The context is about the 2016 World Cup, not relevant to the 2020 World Cup.",
                }
            ]
        },
        # Third question, first context
        {
            "results": [
                {
                    "verdict": "0",
                    "reason": "The context is about the Andes, not relevant to the tallest mountain.",
                }
            ]
        },
        # Third question, second context
        {
            "results": [
                {
                    "verdict": "0",
                    "reason": "The context is about Mount Kilimanjaro, not the tallest mountain.",
                }
            ]
        },
    ]

    evaluator._context_precision_evaluator.run = MagicMock(side_effect=mocked_run_results)

    output = evaluator.run(
        questions=questions,
        answers=answers,
        contexts_list=contexts_list,
        verbose=False,
    )

    # Extract scores from each run result.
    results = output.results
    expected_scores = [1.0, 1.0, 0.0]
    for result, expected in zip(results, expected_scores):
        assert abs(result.score - expected) < 0.01, f"Expected {expected}, got {result.score}"
        # Also assert that reasoning is not empty.
        assert result.reasoning.strip() != "", "Reasoning should not be empty"


def test_context_precision_evaluator_single_list_of_contexts(openai_node):
    """
    Test that the ContextPrecisionEvaluator can accept contexts_list as list[str].
    We'll provide one question, one answer, and multiple context strings in a single list.
    """
    question = ["Tell me about Python programming language."]
    answer = ["Python is a high-level, interpreted programming language known for its readability."]
    # Provide a single list of contexts (list[str])
    contexts_list = [
        "Python is an interpreted language created by Guido van Rossum, first released in 1991.",
        "It's dynamically typed and garbage-collected, and supports multiple programming paradigms.",
        "Python's design philosophy emphasizes code readability with its notable use of significant indentation.",
    ]

    evaluator = ContextPrecisionEvaluator(llm=openai_node)

    # Expected mocked results for each context string
    mocked_run_results = [
        {
            "results": [
                {"verdict": "1", "reason": "Context describes Python accurately and is relevant to the question."}
            ]
        },
        {"results": [{"verdict": "1", "reason": "Context also accurately describes Python and remains relevant."}]},
        {
            "results": [
                {
                    "verdict": "0",
                    "reason": (
                        "Context is focused on Python indentation only, which is only a part "
                        "of its design philosophy, hence less relevant."
                    ),
                }
            ]
        },
    ]

    evaluator._context_precision_evaluator.run = MagicMock(side_effect=mocked_run_results)

    output = evaluator.run(
        questions=question,
        answers=answer,
        contexts_list=contexts_list,
        verbose=False,
    )

    # Since we have three contexts for the single question,
    # compute expected average precision.
    # Our mocked verdicts are [1, 1, 0]:
    # For i=0: precision= 1/1=1, for i=1: precision=2/2=1; sum=2, total relevant=2 => avg=1.0.
    expected_final_score = 1.0

    # There is only one result.
    result = output.results[0]
    assert abs(result.score - expected_final_score) < 0.01, f"Expected {expected_final_score}, got {result.score}"
    assert result.reasoning.strip() != "", "Reasoning should not be empty"
