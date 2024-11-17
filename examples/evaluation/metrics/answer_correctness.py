import logging
import sys

from dotenv import find_dotenv, load_dotenv

from dynamiq.evaluations.metrics.answer_correctness import AnswerCorrectnessEvaluator
from dynamiq.nodes.llms import OpenAI


def main():
    # Load environment variables for OpenAI API
    load_dotenv(find_dotenv())

    # Configure logging level (set to DEBUG to see verbose output)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    # Uncomment the following line to enable verbose logging
    # logging.getLogger().setLevel(logging.DEBUG)

    # Initialize the LLM (replace 'gpt-4o-mini' with your available model)
    llm = OpenAI(model="gpt-4")

    # Sample data (can be replaced with your data)
    questions = [
        "What powers the sun and what is its primary function?",
        "What is the boiling point of water?",
    ]
    answers = [
        (
            "The sun is powered by nuclear fission, similar to nuclear reactors on Earth."
            " Its primary function is to provide light to the solar system."
        ),
        "The boiling point of water is 100 degrees Celsius at sea level.",
    ]
    ground_truths = [
        (
            "The sun is powered by nuclear fusion, where hydrogen atoms fuse to form helium."
            " This fusion process releases a tremendous amount of energy. The sun provides"
            " heat and light, which are essential for life on Earth."
        ),
        (
            "The boiling point of water is 100 degrees Celsius (212 degrees Fahrenheit) at"
            " sea level. The boiling point can change with altitude."
        ),
    ]

    # Initialize evaluator
    evaluator = AnswerCorrectnessEvaluator(llm=llm)

    # Evaluate
    # Set verbose=True to enable detailed logging
    correctness_scores = evaluator.run(
        questions=questions,
        answers=answers,
        ground_truths=ground_truths,
        verbose=False,  # Set verbose=True to enable logging
    )

    # Print the results
    for idx, score in enumerate(correctness_scores):
        print(f"Question: {questions[idx]}")
        print(f"Answer Correctness Score: {score}")
        print("-" * 50)

    print("Answer Correctness Scores:")
    print(correctness_scores)


if __name__ == "__main__":
    main()
