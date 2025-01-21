from unittest.mock import MagicMock

from dynamiq.evaluations.llm_evaluator import LLMEvaluator
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
        inputs=[{"name": "answers", "type": list[str]}],  # Updated inputs definition
        outputs=[{"name": "score", "type": int}],  # Updated outputs definition
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
        inputs=[
            {"name": "answers", "type": list[str]},
            {"name": "ground_truth", "type": list[str]},
        ],  # Updated inputs definition
        outputs=[{"name": "score", "type": int}],  # Updated outputs definition
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
        Evaluate the 'Answer Correctness'. Firstly, read the <question>, <ground_truth_answer>, and <answer_by_llm>.
        Then analyze both answers and evaluate if they are similar.
        - Score this metric from 0 to 1.
        - Use 1 if the <answer_by_llm> adequately answers the <question> as well as the <ground_truth_answer>.
        - Use 0 if the <answer_by_llm> is very different from the <ground_truth_answer> and
          the <question> cannot be answered completely by the <answer_by_llm>.
    """

    evaluator = LLMEvaluator(
        instructions=instruction_text.strip(),
        inputs=[
            {"name": "question", "type": list[str]},  # Updated inputs definition
            {"name": "ground_truth_answer", "type": list[str]},
            {"name": "answer_by_llm", "type": list[str]},
        ],
        outputs=[{"name": "score", "type": int}],  # Updated outputs definition
        examples=[
            {
                "inputs": {
                    "question": "What is the capital of Ukraine?",
                    "answer_by_llm": "Lviv is the capital of Ukraine",
                    "ground_truth_answer": "Kyiv is the capital of Ukraine",
                },
                "outputs": {"score": 0},
            },
            {
                "inputs": {
                    "question": "What is the capital of Ukraine?",
                    "answer_by_llm": "Kyiv is the capital of Ukraine",
                    "ground_truth_answer": "Kyiv is the capital of Ukraine",
                },
                "outputs": {"score": 1},
            },
        ],
        llm=mock_llm,
    )

    questions = [
        "What is the capital of Ukraine?",
        "Who created the Python programming language?",
    ]

    answers = [
        "Berlin is the capital of Great Britain",
        "Python language was created by Guido van Rossum.",
    ]

    ground_truth = [
        "London is the capital of Great Britain",
        "Python language was created by Guido van Rossum.",
    ]

    results = evaluator.run(
        question=questions,
        answer_by_llm=answers,
        ground_truth_answer=ground_truth,
    )
    expected_results = {"results": [{"score": 0}, {"score": "1"}]}

    assert results == expected_results


def test_llm_evaluator_with_placeholders_no_inputs_with_examples():
    # Mock LLM
    mock_llm = MagicMock(spec=BaseLLM)
    mock_llm.execute.return_value = {"content": '{"quote": "Believe in yourself and all that you are."}'}

    instructions = "Please provide a motivational quote: '{{ quote_placeholder }}'"

    # Instantiate LLMEvaluator without inputs
    evaluator = LLMEvaluator(
        instructions=instructions,
        outputs=[{"name": "quote", "type": str}],
        examples=[
            {
                "inputs": {},
                "outputs": {
                    "quote": (
                        "Hard times don't create heroes."
                        " It is during the hard times when the 'hero' within us is revealed."
                    )
                },
            }
        ],
        llm=mock_llm,
    )

    # Run the evaluator without inputs
    results = evaluator.run()
    expected_results = {"results": [{"quote": "Believe in yourself and all that you are."}]}

    assert results == expected_results


def test_llm_evaluator_with_placeholders_no_inputs_without_examples():
    # Mock LLM
    mock_llm = MagicMock(spec=BaseLLM)
    mock_llm.execute.return_value = {"content": '{"quote": "Believe in yourself and all that you are."}'}

    instructions = "Please provide a motivational quote: '{{ quote_placeholder }}'"

    # Instantiate LLMEvaluator without inputs
    evaluator = LLMEvaluator(
        instructions=instructions,
        outputs=[{"name": "quote", "type": str}],
        llm=mock_llm,
    )

    # Run the evaluator without inputs
    results = evaluator.run()
    expected_results = {"results": [{"quote": "Believe in yourself and all that you are."}]}

    assert results == expected_results
