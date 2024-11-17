import logging
import sys

from dotenv import find_dotenv, load_dotenv

from dynamiq.evaluations.metrics.context_recall import ContextRecallEvaluator
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

    # Sample data
    questions = ["What can you tell me about Albert Einstein?"]
    contexts = [
        (
            "Albert Einstein (14 March 1879 - 18 April 1955) was a German-born "
            "theoretical physicist, widely held to be one of the greatest and most "
            "influential scientists of all time. Best known for developing the "
            "theory of relativity, he also made important contributions to quantum "
            "mechanics, and was thus a central figure in the revolutionary reshaping "
            "of the scientific understanding of nature that modern physics accomplished "
            "in the first decades of the twentieth century. His mass-energy equivalence "
            "formula E = mc^2, which arises from relativity theory, has been called "
            "'the world's most famous equation'. He received the 1921 Nobel Prize in "
            "Physics 'for his services to theoretical physics, and especially for his "
            "discovery of the law of the photoelectric effect', a pivotal step in the "
            "development of quantum theory. His work is also known for its influence on "
            "the philosophy of science. In a 1999 poll of 130 leading physicists "
            "worldwide by the British journal Physics World, Einstein was ranked the "
            "greatest physicist of all time. His intellectual achievements and "
            "originality have made Einstein synonymous with genius."
        )
    ]
    answers = [
        (
            "Albert Einstein, born on 14 March 1879, was a German-born theoretical "
            "physicist, widely held to be one of the greatest and most influential "
            "scientists of all time. He received the 1921 Nobel Prize in Physics for "
            "his services to theoretical physics. He published 4 papers in 1905. "
            "Einstein moved to Switzerland in 1895."
        )
    ]

    # Initialize evaluator and evaluate
    evaluator = ContextRecallEvaluator(llm=llm)
    recall_scores = evaluator.run(
        questions=questions, contexts=contexts, answers=answers, verbose=True  # Set to False to disable verbose logging
    )

    # Print the results
    for idx, score in enumerate(recall_scores):
        print(f"Question: {questions[idx]}")
        print(f"Context Recall Score: {score}")
        print("-" * 50)

    print("Context Recall Scores:")
    print(recall_scores)


if __name__ == "__main__":
    main()
