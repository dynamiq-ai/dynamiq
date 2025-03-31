from dotenv import find_dotenv, load_dotenv

from dynamiq.evaluations.metrics import FactualCorrectnessEvaluator
from dynamiq.nodes.llms import OpenAI


def main():
    load_dotenv(find_dotenv())

    llm = OpenAI(model="gpt-4o-mini")

    answers = [
        (
            "Albert Einstein was a German theoretical physicist. "
            "He developed the theory of relativity and contributed "
            "to quantum mechanics."
        ),
        ("The Eiffel Tower is located in Berlin, Germany. " "It was constructed in 1889."),
    ]
    contexts = [
        ("Albert Einstein was a German-born theoretical physicist. " "He developed the theory of relativity."),
        ("The Eiffel Tower is located in Paris, France. " "It was constructed in 1887 and opened in 1889."),
    ]

    evaluator = FactualCorrectnessEvaluator(llm=llm)
    results = evaluator.run(answers=answers, contexts=contexts)

    for idx, result in enumerate(results.results):
        print(f"Answer: {answers[idx]}")
        print(f"Factual Correctness Score: {result.score}")
        print("Reasoning:")
        print(result.reasoning)
        print("-" * 50)


if __name__ == "__main__":
    main()
