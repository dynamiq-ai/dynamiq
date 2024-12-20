from unittest.mock import MagicMock

from dynamiq.evaluations.metrics import ContextRecallEvaluator


def test_context_recall_evaluator(openai_node):
    # Sample data
    questions = ["What can you tell me about Albert Einstein?", "Tell me about the Great Wall of China."]
    contexts = [
        (
            "Albert Einstein (14 March 1879 - 18 April 1955) was a German-born "
            "theoretical physicist, widely held to be one of the greatest and most "
            "influential scientists of all time. Best known for developing the "
            "theory of relativity, he also made important contributions to quantum "
            "mechanics."
        ),
        (
            "The Great Wall of China is a series of fortifications that were built "
            "across the historical northern borders of ancient Chinese states and "
            "Imperial China as protection against various nomadic groups."
        ),
    ]
    answers = [
        (
            "Albert Einstein was a theoretical physicist. He developed the theory of "
            "relativity and contributed to quantum mechanics. He was born in Germany "
            "and won the Nobel Prize in Physics in 1921. He loved playing the violin."
        ),
        (
            "The Great Wall of China is visible from space. It was built to protect "
            "against invasions. It stretches over 13,000 miles and is thousands of "
            "years old."
        ),
    ]

    # Initialize evaluator with the provided openai_node
    evaluator = ContextRecallEvaluator(llm=openai_node)

    # Prepare the mocked results for each classification
    mocked_run_results = [
        # First question
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
        # Second question
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
                            "reason": "Specific lengths and age are not provided in the context.",
                            "attributed": "0",
                        },
                    ]
                }
            ]
        },
    ]

    # Mock the run method of the evaluator's internal LLMEvaluator
    evaluator._classification_evaluator.run = MagicMock(side_effect=mocked_run_results)

    # Run the evaluator
    recall_scores = evaluator.run(questions=questions, contexts=contexts, answers=answers, verbose=False)

    # Expected scores based on the mocked data
    expected_scores = [
        0.75,  # For the first question: 3 out of 4 statements attributed
        0.3333,  # For the second question: 1 out of 3 statements attributed
    ]

    # Assert that the recall scores are as expected
    for computed, expected in zip(recall_scores, expected_scores):
        assert abs(computed - expected) < 0.01, f"Expected {expected}, got {computed}"


def test_context_recall_evaluator_with_list_of_lists_of_contexts(openai_node):
    """
    Test that the ContextRecallEvaluator can accept contexts as list[list[str]].
    The code should join each sub-list into a single string for each question.
    """

    # Sample data with two questions, each question having two context strings
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

    # Initialize evaluator with the provided openai_node
    evaluator = ContextRecallEvaluator(llm=openai_node)

    # Mocked results for each question
    # (We assume the joined contexts match the content of each sub-list.)
    mocked_run_results = [
        # First question
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
        # Second question
        {
            "results": [
                {
                    "classifications": [
                        {
                            "statement": "The Great Wall of China is visible from space.",
                            "reason": "Not mentioned (common myth), so not in the context.",
                            "attributed": "0",
                        },
                        {
                            "statement": "It was built to protect against invasions.",
                            "reason": "Yes, that is mentioned in the context.",
                            "attributed": "1",
                        },
                        {
                            "statement": "It stretches over 13,000 miles and is thousands of years old.",
                            "reason": "Not in the context.",
                            "attributed": "0",
                        },
                    ]
                }
            ]
        },
    ]

    # Mock the run method of the evaluator's internal LLMEvaluator
    evaluator._classification_evaluator.run = MagicMock(side_effect=mocked_run_results)

    # Run the evaluator
    recall_scores = evaluator.run(
        questions=questions, contexts=contexts, answers=answers, verbose=False  # passing list[list[str]]
    )

    # Expected scores based on the mocked data:
    # First question: 2 out of 4 => 0.5
    # Second question: 1 out of 3 => ~0.3333
    expected_scores = [0.5, 0.3333]

    # Assert that the recall scores match our expectations
    for computed, expected in zip(recall_scores, expected_scores):
        assert abs(computed - expected) < 0.01, f"Expected {expected}, got {computed}"
