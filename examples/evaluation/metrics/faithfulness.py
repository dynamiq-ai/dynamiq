from dotenv import find_dotenv, load_dotenv
from dynamiq.evaluations.metrics import FaithfulnessEvaluator
from dynamiq.nodes.llms import OpenAI


def main():
    load_dotenv(find_dotenv())

    llm = OpenAI(model="gpt-4o-mini")

    questions = ["Who was Albert Einstein and what is he best known for?", "Tell me about the Great Wall of China."]
    answers = [
        (
            "He was a German-born theoretical physicist, widely acknowledged to be one of the "
            "greatest and most influential physicists of all time. He was best known for developing "
            "the theory of relativity; he also made important contributions to quantum mechanics."
        ),
        (
            "The Great Wall of China is a large wall in China. It was built to keep out invaders. "
            "It is visible from space."
        ),
    ]
    contexts = [
        ("Albert Einstein was a German-born theoretical physicist. He developed the theory of relativity."),
        (
            "The Great Wall of China is a series of fortifications built across the historical "
            "northern borders of ancient Chinese states and Imperial China as protection against "
            "various nomadic groups."
        ),
    ]

    evaluator = FaithfulnessEvaluator(llm=llm)
    scores = evaluator.run(questions=questions, answers=answers, contexts=contexts)

    for idx, result in enumerate(scores.results):
        print(f"Question: {questions[idx]}")
        print(f"Faithfulness Score: {result.score}")
        print(result.reasoning)
        print("-" * 50)


if __name__ == "__main__":
    main()
