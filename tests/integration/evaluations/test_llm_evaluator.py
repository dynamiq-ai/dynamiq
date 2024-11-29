from unittest.mock import MagicMock

from dynamiq.components.evaluators.llm_evaluator import LLMEvaluator
from dynamiq.nodes.llms import BaseLLM


def test_llm_evaluator_without_ground_truth():
    # Mock LLM
    mock_llm = MagicMock(spec=BaseLLM)

    # Define what the llm.execute method should return
    # It should return a dict with key 'content' containing the LLM's output as a string
    mock_llm.execute.side_effect = [
        {"content": '{"score": 0}'},
        {"content": '{"score": 0}'},
        {"content": '{"score": 1}'},
    ]

    evaluator = LLMEvaluator(
        instructions="Is this answer problematic for children?",
        inputs=[("answers", list[str])],  # Corrected inputs definition
        outputs=["score"],
        examples=[
            {
                "inputs": {"answers": "Damn, this is straight outta hell!!!"},
                "outputs": {"score": 1},
            },
            {
                "inputs": {"answers": "Football is the most popular sport."},
                "outputs": {"score": 0},
            },
        ],
        llm=mock_llm,
    )

    answers = [
        "Football is the most popular sport with around 4 billion followers worldwide",
        "Python language was created by Guido van Rossum.",
        "Damn, this is straight outta hell!!!",
    ]

    results = evaluator.run(answers=answers)
    expected_results = {"results": [{"score": 0}, {"score": 0}, {"score": 1}]}

    assert results == expected_results


def test_llm_evaluator_with_ground_truth():
    # Mock LLM
    mock_llm = MagicMock(spec=BaseLLM)
    mock_llm.execute.side_effect = [
        {"content": '{"score": 0}'},
        {"content": '{"score": 1}'},
    ]

    evaluator = LLMEvaluator(
        instructions="Is the answer correct compared to the ground truth answer?",
        inputs=[("answers", list[str]), ("ground_truth", list[str])],  # Corrected inputs definition
        outputs=["score"],
        examples=[
            {
                "inputs": {
                    "answers": "Lviv is a capital of Ukraine",
                    "ground_truth": "Kyiv is the capital of Ukraine",
                },
                "outputs": {"score": 0},
            },
            {
                "inputs": {
                    "answers": "Kyiv is a capital of Ukraine",
                    "ground_truth": "Kyiv is the capital of Ukraine",
                },
                "outputs": {"score": 1},
            },
        ],
        llm=mock_llm,
    )

    answers = [
        "Berlin is the capital of Great Britain",
        "Python language was created by Guido van Rossum.",
    ]

    ground_truth = [
        "London is the capital of Great Britain",
        "Python language was created by Guido van Rossum.",
    ]

    results = evaluator.run(answers=answers, ground_truth=ground_truth)
    expected_results = {"results": [{"score": 0}, {"score": 1}]}

    assert results == expected_results


def test_llm_evaluator_with_answer_correctness():
    # Mock LLM
    mock_llm = MagicMock(spec=BaseLLM)
    mock_llm.execute.side_effect = [
        {"content": '{"score": 0}'},
        {"content": '{"score": "1"}'},  # Testing with a string to ensure parsing works
    ]

    instruction_text = """
        Evaluate the "Answer Correctness". Firstly, read the <question> and <ground_truth_answer> and <answer_by_llm>.
        Then analyze both answers and evaluate if the answers are similar.
        - Score this metric from 0 to 1.
        - Use 1 if the score is positive, if the <answer_by_llm> can answer the <question> as <ground_truth_answer>
        - Use 0 if the <answer_by_llm> is very different than <ground_truth_answer> and
            the <question> cannot be answered completely only by <answer_by_llm>
    """

    evaluator = LLMEvaluator(
        instructions=instruction_text.strip(),
        inputs=[
            ("question", list[str]),  # Corrected inputs definition
            ("ground_truth_answer", list[str]),
            ("answer_by_llm", list[str]),
        ],
        outputs=["score"],
        examples=[
            {
                "inputs": {
                    "question": "What is the capital of Ukraine?",
                    "answer_by_llm": "Lviv is a capital of Ukraine",
                    "ground_truth_answer": "Kyiv is the capital of Ukraine",
                },
                "outputs": {"score": 0},
            },
            {
                "inputs": {
                    "question": "What is the capital of Ukraine?",
                    "answer_by_llm": "Kyiv is a capital of Ukraine",
                    "ground_truth_answer": "Kyiv is the capital of Ukraine",
                },
                "outputs": {"score": 1},
            },
        ],
        llm=mock_llm,
    )

    questions = [
        "What is the capital of Ukraine?",
        "Who created the Python language?",
    ]

    answers = [
        "Berlin is the capital of Great Britain",
        "Python language was created by Guido van Rossum.",
    ]

    ground_truth = [
        "London is the capital of Great Britain",
        "Python language was created by Guido van Rossum.",
    ]

    results = evaluator.run(question=questions, answer_by_llm=answers, ground_truth_answer=ground_truth)
    expected_results = {"results": [{"score": 0}, {"score": "1"}]}

    assert results == expected_results
