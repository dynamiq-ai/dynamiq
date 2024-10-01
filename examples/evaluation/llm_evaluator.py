from dynamiq.components.evaluators.llm_evaluator import LLMEvaluator
from dynamiq.nodes.llms import BaseLLM, OpenAI


def run_example_without_ground_truth(llm: BaseLLM):
    evaluator = LLMEvaluator(
        instructions="Is this answer problematic for children?",
        inputs=[("answers", list[str])],
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
        inputs=[("answers", list[str]), ("ground_truth", list[str])],
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
        Evaluate the "Answer Correctness". Firstly, read the <question> and <ground_truth_answer> and <answer_by_llm>.
        Then analyze both answers and evaluate if the answers are similar.
        - Score this metric from 0 to 1.
        - Use 1 if the score is positive, if the <answer_by_llm> can answer the <question> as <ground_truth_answer>
        - Use 0 if the <answer_by_llm> is very different than <ground_truth_answer> and
            the <question> cannot be answered completely only by <answer_by_llm>
    """

    evaluator = LLMEvaluator(
        instructions=instruction_text,
        inputs=[
            ("question", list[str]),
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
        llm=llm,
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

    results = evaluator.run(
        question=questions, answer_by_llm=answers, ground_truth_answer=ground_truth
    )
    return results


def main():
    llm = OpenAI(
        name="OpenAI",
        model="gpt-4o-mini",
        postponned_init=True,
    )
    withouth_ground_truth_results = run_example_without_ground_truth(llm)
    print(withouth_ground_truth_results)

    with_ground_truth_results = run_example_with_ground_truth(llm)
    print(with_ground_truth_results)

    answer_correctness_results = run_example_with_answer_correctness(llm)
    print(answer_correctness_results)


if __name__ == "__main__":
    main()
