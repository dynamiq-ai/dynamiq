from dynamiq.evaluations.llm_evaluator import LLMEvaluator
from dynamiq.nodes.llms import BaseLLM, OpenAI


def run_example_without_ground_truth(llm: BaseLLM):
    evaluator = LLMEvaluator(
        instructions="Is this answer problematic for children?",
        inputs=[{"name": "answers", "type": list[str]}],
        outputs=[
            {"name": "reasoning", "type": str},
            {"name": "score", "type": int},
        ],
        examples=[
            {
                "inputs": {"answers": "Damn, this is straight outta hell!!!"},
                "outputs": {
                    "reasoning": "The answer contains inappropriate language.",
                    "score": 1,
                },
            },
            {
                "inputs": {"answers": "Football is the most popular sport."},
                "outputs": {
                    "reasoning": "The answer is appropriate for children.",
                    "score": 0,
                },
            },
        ],
        llm=llm,
    )

    answers = [
        "Football is the most popular sport with around 4 billion followers worldwide",
        "Python language was created by Guido van Rossum.",
        "Damn, this is straight outta hell!!!",
    ]
    results = evaluator.run(answers=answers)
    return results


def run_example_with_ground_truth(llm: BaseLLM):
    evaluator = LLMEvaluator(
        instructions="Is the answer correct compared to the ground truth answer?",
        inputs=[
            {"name": "answers", "type": list[str]},
            {"name": "ground_truth", "type": list[str]},
        ],
        outputs=[
            {"name": "reasoning", "type": str},
            {"name": "score", "type": int},
        ],
        examples=[
            {
                "inputs": {
                    "answers": "Lviv is a capital of Ukraine",
                    "ground_truth": "Kyiv is the capital of Ukraine",
                },
                "outputs": {
                    "reasoning": "The answer provides an incorrect capital for Ukraine.",
                    "score": 0,
                },
            },
            {
                "inputs": {
                    "answers": "Kyiv is a capital of Ukraine",
                    "ground_truth": "Kyiv is the capital of Ukraine",
                },
                "outputs": {
                    "reasoning": "The answer correctly matches the ground truth.",
                    "score": 1,
                },
            },
        ],
        llm=llm,
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
    return results


def run_example_with_answer_correctness(llm: BaseLLM):
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
            {"name": "question", "type": list[str]},
            {"name": "ground_truth_answer", "type": list[str]},
            {"name": "answer_by_llm", "type": list[str]},
        ],
        outputs=[
            {"name": "reasoning", "type": str},
            {"name": "score", "type": int},
        ],
        examples=[
            {
                "inputs": {
                    "question": "What is the capital of Ukraine?",
                    "answer_by_llm": "Lviv is the capital of Ukraine.",
                    "ground_truth_answer": "Kyiv is the capital of Ukraine.",
                },
                "outputs": {
                    "reasoning": "The answer provides an incorrect capital for Ukraine.",
                    "score": 0,
                },
            },
            {
                "inputs": {
                    "question": "What is the capital of Ukraine?",
                    "answer_by_llm": "Kyiv is the capital of Ukraine.",
                    "ground_truth_answer": "Kyiv is the capital of Ukraine.",
                },
                "outputs": {
                    "reasoning": "The answer correctly matches the ground truth.",
                    "score": 1,
                },
            },
        ],
        llm=llm,
    )

    questions = [
        "What is the capital of Ukraine?",
        "Who created the Python programming language?",
    ]

    answers = [
        "Berlin is the capital of Great Britain.",
        "Python language was created by Guido van Rossum.",
    ]

    ground_truth = [
        "London is the capital of Great Britain.",
        "Python language was created by Guido van Rossum.",
    ]

    results = evaluator.run(question=questions, answer_by_llm=answers, ground_truth_answer=ground_truth)
    return results


def main():
    # Initialize the LLM (provide your LLM configuration)
    llm = OpenAI(
        name="OpenAI",
        model="gpt-4o-mini",
    )

    # Run the examples
    without_ground_truth_results = run_example_without_ground_truth(llm)
    print("Results without ground truth:")
    print(without_ground_truth_results)

    with_ground_truth_results = run_example_with_ground_truth(llm)
    print("\nResults with ground truth:")
    print(with_ground_truth_results)

    answer_correctness_results = run_example_with_answer_correctness(llm)
    print("\nAnswer correctness results:")
    print(answer_correctness_results)


if __name__ == "__main__":
    main()
