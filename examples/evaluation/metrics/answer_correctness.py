from dotenv import find_dotenv, load_dotenv

from dynamiq.evaluations.metrics import AnswerCorrectnessEvaluator
from dynamiq.nodes.llms import OpenAI


def main():
    load_dotenv(find_dotenv())
    llm = OpenAI(model="gpt-4o-mini")

    questions = ["What powers the sun and what is its primary function?", "What is the boiling point of water?"]
    answers = [
        (
            "The sun is powered by nuclear fission, similar to nuclear reactors on Earth. "
            "Its primary function is to provide heat and light to the solar system."
        ),
        "The boiling point of water is 100 degrees Celsius at sea level.",
    ]
    ground_truth_answers = [
        (
            "The sun is powered by nuclear fusion, where hydrogen fuses to form helium. "
            "This fusion releases energy. The sun provides heat and light essential for life on Earth."
        ),
        (
            "The boiling point of water is 100 degrees Celsius (212°F) at sea level. "
            "Note that the boiling point changes with altitude."
        ),
    ]

    evaluator = AnswerCorrectnessEvaluator(llm=llm)
    results = evaluator.run(questions=questions, answers=answers, ground_truth_answers=ground_truth_answers)

    for idx, result in enumerate(results.results):
        print(f"Question {idx+1}: {questions[idx]}")
        print(f"Answer Correctness Score: {result.score}")
        print(result.reasoning)
        print("-" * 50)


if __name__ == "__main__":
    main()
