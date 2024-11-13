import json
import logging

from dynamiq.components.evaluators.llm_evaluator import LLMEvaluator
from dynamiq.nodes.llms import BaseLLM, OpenAI

# Configure logging
logger = logging.getLogger(__name__)


class ContextRecallEvaluator:
    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self._initialize_evaluators()

    def _initialize_evaluators(self):
        # Instructions for context recall classification
        class_instructions = (
            'Given a "Question", "Context", and "Answer", analyze each sentence '
            "in the Answer and classify if the sentence can be attributed to the "
            "given Context or not.\n"
            "- Use '1' (Yes) or '0' (No) for classification.\n"
            '- Provide a brief "reason" for each classification.\n'
            '- Output as a JSON object with key "classifications", where the value '
            'is a list of dictionaries with keys "statement", "reason", and "attributed".\n'
            "- Ensure your response is valid JSON, using double quotes for all strings."
        )

        self.classification_evaluator = LLMEvaluator(
            instructions=class_instructions.strip(),
            inputs=[
                ("question", list[str]),
                ("context", list[str]),
                ("answer", list[str]),
            ],
            outputs=["classifications"],
            examples=[
                {
                    "inputs": {
                        "question": ["What can you tell me about Albert Einstein?"],
                        "context": [
                            (
                                "Albert Einstein (14 March 1879 - 18 April 1955) was a German-born "
                                "theoretical physicist, widely held to be one of the greatest and "
                                "most influential scientists of all time. Best known for developing "
                                "the theory of relativity, he also made important contributions to "
                                "quantum mechanics, and was thus a central figure in the revolutionary "
                                "reshaping of the scientific understanding of nature that modern "
                                "physics accomplished in the first decades of the twentieth century. "
                                "His mass-energy equivalence formula E = mc^2, which arises from "
                                "relativity theory, has been called 'the world's most famous equation'. "
                                "He received the 1921 Nobel Prize in Physics 'for his services to "
                                "theoretical physics, and especially for his discovery of the law of "
                                "the photoelectric effect', a pivotal step in the development of "
                                "quantum theory. His work is also known for its influence on the "
                                "philosophy of science. In a 1999 poll of 130 leading physicists "
                                "worldwide by the British journal Physics World, Einstein was ranked "
                                "the greatest physicist of all time. His intellectual achievements and "
                                "originality have made Einstein synonymous with genius."
                            )
                        ],
                        "answer": [
                            (
                                "Albert Einstein, born on 14 March 1879, was a German-born theoretical "
                                "physicist, widely held to be one of the greatest and most influential "
                                "scientists of all time. He received the 1921 Nobel Prize in Physics for "
                                "his services to theoretical physics. He published 4 papers in 1905. "
                                "Einstein moved to Switzerland in 1895."
                            )
                        ],
                    },
                    "outputs": {
                        "classifications": [
                            {
                                "statement": (
                                    "Albert Einstein, born on 14 March 1879, was a German-born "
                                    "theoretical physicist, widely held to be one of the greatest "
                                    "and most influential scientists of all time."
                                ),
                                "reason": ("The date of birth of Einstein is mentioned clearly in the context."),
                                "attributed": 1,
                            },
                            {
                                "statement": (
                                    "He received the 1921 Nobel Prize in Physics for his services to "
                                    "theoretical physics."
                                ),
                                "reason": "The exact sentence is present in the given context.",
                                "attributed": 1,
                            },
                            {
                                "statement": "He published 4 papers in 1905.",
                                "reason": ("There is no mention about papers he wrote in the given context."),
                                "attributed": 0,
                            },
                            {
                                "statement": "Einstein moved to Switzerland in 1895.",
                                "reason": ("There is no supporting evidence for this in the given context."),
                                "attributed": 0,
                            },
                        ]
                    },
                },
            ],
            llm=self.llm,
        )

    def evaluate(
        self, questions: list[str], contexts: list[str], answers: list[str], verbose: bool = False
    ) -> list[float]:
        if not (len(questions) == len(contexts) == len(answers)):
            raise ValueError("Questions, contexts, and answers must have the same length.")

        final_scores = []

        for idx in range(len(questions)):
            question = questions[idx]
            context = contexts[idx]
            answer = answers[idx]

            result = self.classification_evaluator.run(
                question=[question],
                context=[context],
                answer=[answer],
            )

            # Extract classifications
            classifications = result["results"][0]["classifications"]

            # Compute the score
            attributed_list = [int(item["attributed"]) for item in classifications]
            num_sentences = len(attributed_list)
            num_attributed = sum(attributed_list)
            score = num_attributed / num_sentences if num_sentences > 0 else 0.0
            final_scores.append(score)

            if verbose:
                logger.debug(f"Question: {question}")
                logger.debug(f"Answer: {answer}")
                logger.debug(f"Context: {context}")
                logger.debug("Classifications:")
                logger.debug(json.dumps(classifications, indent=2))
                logger.debug(f"Context Recall Score: {score}")
                logger.debug("-" * 50)

        return final_scores


# Example usage
if __name__ == "__main__":
    import sys

    from dotenv import find_dotenv, load_dotenv

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
            "Albert Einstein (14 March 1879 - 18 April 1955) was a German-born theoretical "
            "physicist, widely held to be one of the greatest and most influential scientists "
            "of all time. Best known for developing the theory of relativity, he also made "
            "important contributions to quantum mechanics, and was thus a central figure in "
            "the revolutionary reshaping of the scientific understanding of nature that modern "
            "physics accomplished in the first decades of the twentieth century. His mass-energy "
            "equivalence formula E = mc^2, which arises from relativity theory, has been called "
            "'the world's most famous equation'. He received the 1921 Nobel Prize in Physics 'for "
            "his services to theoretical physics, and especially for his discovery of the law of "
            "the photoelectric effect', a pivotal step in the development of quantum theory. His "
            "work is also known for its influence on the philosophy of science. In a 1999 poll of "
            "130 leading physicists worldwide by the British journal Physics World, Einstein was "
            "ranked the greatest physicist of all time. His intellectual achievements and "
            "originality have made Einstein synonymous with genius."
        )
    ]
    answers = [
        (
            "Albert Einstein, born on 14 March 1879, was a German-born theoretical physicist, "
            "widely held to be one of the greatest and most influential scientists of all time. "
            "He received the 1921 Nobel Prize in Physics for his services to theoretical physics. "
            "He published 4 papers in 1905. Einstein moved to Switzerland in 1895."
        )
    ]

    # Initialize evaluator and evaluate
    evaluator = ContextRecallEvaluator(llm)
    recall_scores = evaluator.evaluate(questions, contexts, answers, verbose=True)

    # Print the results
    for idx, score in enumerate(recall_scores):
        print(f"Question: {questions[idx]}")
        print(f"Context Recall Score: {score}")
        print("-" * 50)

    print("Context Recall Scores:")
    print(recall_scores)
