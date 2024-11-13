import logging

from dynamiq.components.evaluators.llm_evaluator import LLMEvaluator
from dynamiq.nodes.llms import BaseLLM, OpenAI

# Configure logging
logger = logging.getLogger(__name__)


class ContextPrecisionEvaluator:
    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self._initialize_evaluators()

    def _initialize_evaluators(self):
        # Initialize the LLMEvaluator for context precision

        context_precision_instructions = (
            'Given a "Question", "Answer", and "Context", verify if the Context was '
            "useful in arriving at the given Answer.\n"
            '- Provide a "verdict": 1 if useful, 0 if not.\n'
            '- Provide a brief "reason" for the verdict.\n'
            '- Output the result as a JSON object with keys "verdict" and "reason".\n'
            "- Ensure that your response is valid JSON, using double quotes for all "
            "strings."
        )

        self.context_precision_evaluator = LLMEvaluator(
            instructions=context_precision_instructions.strip(),
            inputs=[
                ("question", list[str]),
                ("answer", list[str]),
                ("context", list[str]),
            ],
            outputs=["verdict", "reason"],
            examples=[
                {
                    "inputs": {
                        "question": ["What can you tell me about Albert Einstein?"],
                        "answer": [
                            (
                                "Albert Einstein, born on 14 March 1879, was a German-born theoretical "
                                "physicist, widely held to be one of the greatest and most influential "
                                "scientists of all time. He received the 1921 Nobel Prize in Physics "
                                "for his services to theoretical physics."
                            )
                        ],
                        "context": [
                            (
                                "Albert Einstein (14 March 1879 – 18 April 1955) was a German-born "
                                "theoretical physicist, widely held to be one of the greatest and most "
                                "influential scientists of all time. Best known for developing the theory "
                                "of relativity, he also made important contributions to quantum mechanics, "
                                "and was thus a central figure in the revolutionary reshaping of the "
                                "scientific understanding of nature that modern physics accomplished in "
                                "the first decades of the twentieth century. His mass–energy equivalence "
                                "formula E = mc2, which arises from relativity theory, has been called "
                                "'the world's most famous equation'. He received the 1921 Nobel Prize in "
                                "Physics 'for his services to theoretical physics, and especially for his "
                                "discovery of the law of the photoelectric effect', a pivotal step in the "
                                "development of quantum theory. His work is also known for its influence on "
                                "the philosophy of science. In a 1999 poll of 130 leading physicists "
                                "worldwide by the British journal Physics World, Einstein was ranked the "
                                "greatest physicist of all time. His intellectual achievements and "
                                "originality have made Einstein synonymous with genius."
                            )
                        ],
                    },
                    "outputs": {
                        "verdict": 1,
                        "reason": (
                            "The context provides detailed information about Albert Einstein that is "
                            "reflected in the answer, including his birthdate, his contributions to "
                            "physics, and his Nobel Prize."
                        ),
                    },
                },
                {
                    "inputs": {
                        "question": ["Who won the 2020 ICC World Cup?"],
                        "answer": ["England"],
                        "context": [
                            (
                                "The 2022 ICC Men's T20 World Cup, held from October 16 to November 13, "
                                "2022, in Australia, was the eighth edition of the tournament. Originally "
                                "scheduled for 2020, it was postponed due to the COVID-19 pandemic. "
                                "England emerged victorious, defeating Pakistan by five wickets in the "
                                "final to clinch their second ICC Men's T20 World Cup title."
                            )
                        ],
                    },
                    "outputs": {
                        "verdict": 1,
                        "reason": (
                            "The context explains that the 2020 ICC World Cup was postponed to 2022, and "
                            "that England won the tournament, which is directly relevant to the answer."
                        ),
                    },
                },
                {
                    "inputs": {
                        "question": ["What is the tallest mountain in the world?"],
                        "answer": ["Mount Everest."],
                        "context": [
                            (
                                "The Andes is the longest continental mountain range in the world, located "
                                "in South America. It stretches across seven countries and features many of "
                                "the highest peaks in the Western Hemisphere. The range is known for its "
                                "diverse ecosystems, including the high-altitude Andean Plateau and the "
                                "Amazon rainforest."
                            )
                        ],
                    },
                    "outputs": {
                        "verdict": 0,
                        "reason": (
                            "The context discusses the Andes mountain range, which does not include Mount "
                            "Everest. Therefore, the context was not useful in arriving at the answer."
                        ),
                    },
                },
            ],
            llm=self.llm,
        )

    @staticmethod
    def calculate_average_precision(verdicts: list[int]) -> float:
        numerator = 0.0
        cumulative_hits = 0
        total_relevant = sum(verdicts)
        if total_relevant == 0:
            return 0.0
        for i, verdict in enumerate(verdicts):
            if verdict == 1:
                cumulative_hits += 1
                precision_at_i = cumulative_hits / (i + 1)
                numerator += precision_at_i
        average_precision = numerator / total_relevant
        return average_precision

    def evaluate(
        self,
        questions: list[str],
        answers: list[str],
        contexts_list: list[list[str]],
        verbose: bool = False,
    ) -> list[float]:
        if not (len(questions) == len(answers) == len(contexts_list)):
            raise ValueError("Questions, answers, and contexts_list must have the same length.")

        final_scores = []

        for idx in range(len(questions)):
            question = questions[idx]
            answer = answers[idx]
            contexts = contexts_list[idx]

            verdicts = []
            for context in contexts:
                # Prepare inputs for the evaluator
                result = self.context_precision_evaluator.run(
                    question=[question],
                    answer=[answer],
                    context=[context],
                )
                # Extract the verdict (ensure it's an int)
                verdict = int(result["results"][0]["verdict"])
                verdicts.append(verdict)

                if verbose:
                    reason = result["results"][0]["reason"]
                    # Use logging instead of print
                    logger.debug(f"Question: {question}")
                    logger.debug(f"Answer: {answer}")
                    logger.debug(f"Context: {context}")
                    logger.debug(f"Verdict: {verdict}")
                    logger.debug(f"Reason: {reason}")
                    logger.debug("-" * 50)

            # Calculate average precision for this set
            score = self.calculate_average_precision(verdicts)
            final_scores.append(score)

            if verbose:
                logger.debug(f"Average Precision Score: {score}")
                logger.debug("=" * 50)

        return final_scores


# Example usage
if __name__ == "__main__":
    import sys

    from dotenv import find_dotenv, load_dotenv

    # Load environment variables for OpenAI API
    load_dotenv(find_dotenv())

    # Configure logging level (set to DEBUG to see verbose output)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    # Set the logging level to DEBUG if verbose is desired
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

    # Initialize evaluator and evaluate
    evaluator = ContextPrecisionEvaluator(llm)
    correctness_scores = evaluator.evaluate(
        questions, answers, contexts_list, verbose=False  # Set to True to enable verbose logging
    )

    # Print the results
    for idx, score in enumerate(correctness_scores):
        print(f"Question: {questions[idx]}")
        print(f"Context Precision Score: {score}")
        print("-" * 50)

    print("Context Precision Scores:")
    print(correctness_scores)
