import logging
import sys

from dotenv import find_dotenv, load_dotenv

from dynamiq.evaluations.metrics.context_precision import ContextPrecisionEvaluator
from dynamiq.nodes.llms import OpenAI


def main():
    # Load environment variables for OpenAI API
    load_dotenv(find_dotenv())

    # Configure logging level (set to DEBUG to see verbose output)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    # Uncomment the following line to enable verbose logging
    # logging.getLogger().setLevel(logging.DEBUG)

    # Initialize the LLM (replace 'gpt-4' with your available model)
    llm = OpenAI(model="gpt-4")

    # Sample data (can be replaced with your data)
    questions = [
        "What can you tell me about Albert Einstein?",
        "Who won the 2020 ICC World Cup?",
        "What is the tallest mountain in the world?",
    ]
    answers = [
        (
            "Albert Einstein, born on 14 March 1879, was a German-born theoretical physicist, "
            "widely held to be one of the greatest and most influential scientists of all time. "
            "He received the 1921 Nobel Prize in Physics for his services to theoretical physics."
        ),
        "England",
        "Mount Everest.",
    ]
    contexts_list = [
        [
            # Contexts for the first question
            (
                "Albert Einstein (14 March 1879 – 18 April 1955) was a German-born theoretical "
                "physicist, widely held to be one of the greatest and most influential scientists "
                "of all time. Best known for developing the theory of relativity, he also made "
                "important contributions to quantum mechanics."
            ),
            (
                "Albert Einstein's work is also known for its influence on the philosophy of "
                "science. His mass–energy equivalence formula E = mc^2 has been called 'the world's "
                "most famous equation'."
            ),
        ],
        [
            # Contexts for the second question
            (
                "The 2022 ICC Men's T20 World Cup, held from October 16 to November 13, 2022, in "
                "Australia, was the eighth edition of the tournament. Originally scheduled for "
                "2020, it was postponed due to the COVID-19 pandemic. England emerged victorious, "
                "defeating Pakistan by five wickets in the final to clinch their second ICC Men's "
                "T20 World Cup title."
            ),
            (
                "The 2016 ICC World Twenty20 was held in India from 8 March to 3 April 2016. The "
                "West Indies won the tournament, beating England in the final."
            ),
        ],
        [
            # Contexts for the third question
            (
                "The Andes is the longest continental mountain range in the world, located in "
                "South America. It features many high peaks but not the tallest in the world."
            ),
            ("Mount Kilimanjaro is the highest mountain in Africa, standing at 5,895 meters " "above sea level."),
        ],
    ]

    # Initialize evaluator
    evaluator = ContextPrecisionEvaluator(llm=llm)
    correctness_scores = evaluator.run(
        questions=questions,
        answers=answers,
        contexts_list=contexts_list,
        verbose=False,  # Set to True to enable verbose logging
    )

    # Print the results
    for idx, score in enumerate(correctness_scores):
        print(f"Question: {questions[idx]}")
        print(f"Context Precision Score: {score}")
        print("-" * 50)

    print("Context Precision Scores:")
    print(correctness_scores)


if __name__ == "__main__":
    main()
