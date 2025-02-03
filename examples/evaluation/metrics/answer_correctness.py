import logging
import sys
from dotenv import find_dotenv, load_dotenv

from dynamiq.evaluations.metrics import AnswerCorrectnessEvaluator
from dynamiq.nodes.llms import OpenAI


def main():
    # Load environment variables (e.g., for OpenAI API credentials)
    load_dotenv(find_dotenv())

    # Configure logging level; set to DEBUG for verbose output
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    # Uncomment this line for debug output:
    # logging.getLogger().setLevel(logging.DEBUG)

    # Initialize the LLM (replace 'gpt-4o-mini' with your available model)
    llm = OpenAI(model="gpt-4o-mini")

    # Sample data for evaluation
    questions = ["What powers the sun and what is its primary function?", "What is the boiling point of water?"]
    answers = [
        (
            "The sun is powered by nuclear fission, similar to nuclear reactors on Earth. "
            "Its primary function is to provide light to the solar system."
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

    # Evaluate the answers
    # Set verbose=True to see detailed logging and reasoning.
    results = evaluator.run(
        questions=questions, answers=answers, ground_truth_answers=ground_truth_answers, verbose=False
    )

    # Print detailed results.
    for idx, result in enumerate(results.results):
        print(f"Question {idx+1}: {questions[idx]}")
        print(f"Answer Correctness Score: {result.score}")
        print("Reasoning:")
        print(result.reasoning)
        print("-" * 50)

    print("Final Evaluation Scores:")
    print(results)


if __name__ == "__main__":
    main()
