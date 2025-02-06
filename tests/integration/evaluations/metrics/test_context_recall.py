from unittest.mock import MagicMock

from dynamiq.evaluations.metrics import ContextRecallEvaluator


def test_context_recall_evaluator(openai_node):
    # Sample data
    questions = ["What can you tell me about Albert Einstein?", "Tell me about the Great Wall of China."]
    contexts = [
        (
            "Albert Einstein (14 March 1879 - 18 April 1955) was a German-born theoretical "
            "physicist, widely held to be one of the greatest and most influential scientists "
            "of all time. Best known for developing the theory of relativity, he also made "
            "important contributions to quantum mechanics."
        ),
        (
            "The Great Wall of China is a series of fortifications that were built across "
            "the historical northern borders of ancient Chinese states and Imperial China "
            "as protection against various nomadic groups."
        ),
    ]
    answers = [
        (
            "Albert Einstein was a theoretical physicist. He developed the theory of relativity "
            "and contributed to quantum mechanics. He was born in Germany and won the Nobel Prize "
            "in Physics in 1921. He loved playing the violin."
        ),
        (
            "The Great Wall of China is visible from space. It was built to protect against invasions. "
            "It stretches over 13,000 miles and is thousands of years old."
        ),
    ]

    evaluator = ContextRecallEvaluator(llm=openai_node)

    # Prepare the mocked results for each classification call.
    # There will be one call per question.
    mocked_run_results = [
        # For the first question:
        {
            "results": [
                {
                    "classifications": [
                        {
                            "statement": "Albert Einstein was a theoretical physicist.",
                            "reason": "This information is in the context.",
                            "attributed": "1",
                        },
                        {
                            "statement": "He developed the theory of relativity and contributed to quantum mechanics.",
                            "reason": "Both contributions are mentioned in the context.",
                            "attributed": "1",
                        },
                        {
                            "statement": "He was born in Germany and won the Nobel Prize in Physics in 1921.",
                            "reason": "His birthplace and Nobel Prize are in the context.",
                            "attributed": "1",
                        },
                        {
                            "statement": "He loved playing the violin.",
                            "reason": "This is not mentioned in the context.",
                            "attributed": "0",
                        },
                    ]
                }
            ]
        },
        # For the second question:
        {
            "results": [
                {
                    "classifications": [
                        {
                            "statement": "The Great Wall of China is visible from space.",
                            "reason": "This is a common myth not discussed in the context.",
                            "attributed": "0",
                        },
                        {
                            "statement": "It was built to protect against invasions.",
                            "reason": "The context mentions it was built as protection.",
                            "attributed": "1",
                        },
                        {
                            "statement": "It stretches over 13,000 miles and is thousands of years old.",
                            "reason": "Specific length and age are not provided in the context.",
                            "attributed": "0",
                        },
                    ]
                }
            ]
        },
    ]

    evaluator._classification_evaluator.run = MagicMock(side_effect=mocked_run_results)

    output = evaluator.run(
        questions=questions,
        contexts=contexts,
        answers=answers,
        verbose=False,
    )
    # Extract results from output; each result is a ContextRecallRunResult
    results = output.results

    # Expected scores:
    # For the first question: 3 attributed sentences out of 4 => 3/4 = 0.75
    # For the second question: 1 attributed out of 3 => approx 0.3333
    expected_scores = [0.75, 0.3333]

    for result, expected in zip(results, expected_scores):
        assert abs(result.score - expected) < 0.01, f"Expected {expected}, got {result.score}"
        # Also, the reasoning should never be empty.
        assert result.reasoning.strip() != "", "Reasoning should not be empty"


def test_context_recall_evaluator_with_list_of_lists_of_contexts(openai_node):
    """
    Test that the ContextRecallEvaluator can accept contexts as list[list[str]].
    The evaluator should join each sub-list into a single string.
    """
    questions = ["What can you tell me about Albert Einstein?", "Tell me about the Great Wall of China."]
    contexts = [
        [
            "Albert Einstein (14 March 1879 - 18 April 1955) was a German-born theoretical physicist.",
            "He developed the theory of relativity and also made contributions to quantum mechanics.",
        ],
        [
            "The Great Wall of China is a series of fortifications built across the historical northern borders.",
            "It was built to protect against various nomadic groups, but doesn't mention visibility from space.",
        ],
    ]
    answers = [
        (
            "Albert Einstein was a theoretical physicist. He developed the theory of relativity "
            "and contributed to quantum mechanics. He was born in Germany and won the Nobel Prize "
            "in Physics in 1921. He loved playing the violin."
        ),
        (
            "The Great Wall of China is visible from space. It was built to protect against invasions. "
            "It stretches over 13,000 miles and is thousands of years old."
        ),
    ]

    evaluator = ContextRecallEvaluator(llm=openai_node)

    # Prepare mocked results. When contexts are provided as a list[list[str]],
    # each sub-list is joined into a single string.
    mocked_run_results = [
        # For the first question, joined context:
        {
            "results": [
                {
                    "classifications": [
                        {
                            "statement": "Albert Einstein was a theoretical physicist.",
                            "reason": "Appears in the joined context.",
                            "attributed": "1",
                        },
                        {
                            "statement": "He developed the theory of relativity and contributed to quantum mechanics.",
                            "reason": "Both are mentioned in the joined context.",
                            "attributed": "1",
                        },
                        {
                            "statement": "He was born in Germany and won the Nobel Prize in Physics in 1921.",
                            "reason": "Birthplace and Nobel Prize are not explicitly in the joined context.",
                            "attributed": "0",
                        },
                        {
                            "statement": "He loved playing the violin.",
                            "reason": "This is not mentioned at all.",
                            "attributed": "0",
                        },
                    ]
                }
            ]
        },
        # For the second question, joined context:
        {
            "results": [
                {
                    "classifications": [
                        {
                            "statement": "The Great Wall of China is visible from space.",
                            "reason": "Not mentioned (common myth), so not in the joined context.",
                            "attributed": "0",
                        },
                        {
                            "statement": "It was built to protect against invasions.",
                            "reason": "Yes, that information is in the joined context.",
                            "attributed": "1",
                        },
                        {
                            "statement": "It stretches over 13,000 miles and is thousands of years old.",
                            "reason": "Not mentioned in the joined context.",
                            "attributed": "0",
                        },
                    ]
                }
            ]
        },
    ]

    evaluator._classification_evaluator.run = MagicMock(side_effect=mocked_run_results)

    output = evaluator.run(
        questions=questions,
        contexts=contexts,  # Passing list[list[str]]
        answers=answers,
        verbose=False,
    )
    results = output.results

    # Expected scores:
    # For the first question: 2 out of 4 => 0.5
    # For the second question: 1 out of 3 => approximately 0.3333
    expected_scores = [0.5, 0.3333]

    for result, expected in zip(results, expected_scores):
        assert abs(result.score - expected) < 0.01, f"Expected {expected}, got {result.score}"
        assert result.reasoning.strip() != "", "Reasoning should not be empty"
