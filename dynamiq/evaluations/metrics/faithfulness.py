import logging

from dynamiq.components.evaluators.llm_evaluator import LLMEvaluator
from dynamiq.nodes.llms import BaseLLM, OpenAI

# Configure logging
logger = logging.getLogger(__name__)


class FaithfulnessEvaluator:
    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self._initialize_evaluators()

    def _initialize_evaluators(self):
        # Prompt to simplify sentences and remove pronouns
        simplify_instructions = (
            "Given a 'Question' and an 'Answer', break down each sentence in the "
            "Answer into one or more fully understandable statements.\n"
            "- Ensure no pronouns are used in each statement.\n"
            "- Output as a JSON object with key 'statements', where the value is a "
            "list of statements.\n"
            "- Ensure your response is valid JSON, using double quotes for all strings."
        )

        self.statement_simplifier = LLMEvaluator(
            instructions=simplify_instructions.strip(),
            inputs=[
                ("question", list[str]),
                ("answer", list[str]),
            ],
            outputs=["statements"],
            examples=[
                {
                    "inputs": {
                        "question": ["Who was Albert Einstein and what is he best known for?"],
                        "answer": [
                            (
                                "He was a German-born theoretical physicist, widely "
                                "acknowledged to be one of the greatest and most influential "
                                "physicists of all time. He was best known for developing "
                                "the theory of relativity, he also made important contributions "
                                "to the development of quantum mechanics."
                            )
                        ],
                    },
                    "outputs": {
                        "statements": [
                            "Albert Einstein was a German-born theoretical physicist.",
                            "Albert Einstein is recognized as one of the greatest and "
                            "most influential physicists of all time.",
                            "Albert Einstein was best known for developing the theory " "of relativity.",
                            "Albert Einstein also made important contributions to the "
                            "development of quantum mechanics.",
                        ]
                    },
                },
            ],
            llm=self.llm,
        )

        # Prompt to check faithfulness of statements
        nli_instructions = (
            "Your task is to judge the faithfulness of a series of statements based "
            "on a given Context.\n"
            "- For each statement, return 'verdict': 1 if it can be directly inferred "
            "from the Context, or 0 if not.\n"
            "- Provide a brief 'reason' for the verdict.\n"
            "- Output as a JSON object with key 'results', where the value is a list "
            "of dictionaries with keys 'statement', 'verdict', and 'reason'.\n"
            "- Ensure your response is valid JSON, using double quotes for all strings."
        )

        self.nli_evaluator = LLMEvaluator(
            instructions=nli_instructions.strip(),
            inputs=[
                ("context", list[str]),
                ("statements", list[list[str]]),
            ],
            outputs=["results"],
            examples=[
                {
                    "inputs": {
                        "context": [
                            (
                                "John is a student at XYZ University. He is pursuing a "
                                "degree in Computer Science. He is enrolled in several "
                                "courses this semester, including Data Structures, Algorithms, "
                                "and Database Management. John is a diligent student and "
                                "spends a significant amount of time studying and completing "
                                "assignments. He often stays late in the library to work on "
                                "his projects."
                            )
                        ],
                        "statements": [
                            [
                                "John is majoring in Biology.",
                                "John is taking a course on Artificial Intelligence.",
                                "John is a dedicated student.",
                                "John has a part-time job.",
                            ]
                        ],
                    },
                    "outputs": {
                        "results": [
                            {
                                "statement": "John is majoring in Biology.",
                                "verdict": 0,
                                "reason": (
                                    "The context states that John is pursuing a degree "
                                    "in Computer Science, not Biology."
                                ),
                            },
                            {
                                "statement": "John is taking a course on Artificial Intelligence.",
                                "verdict": 0,
                                "reason": (
                                    "The context lists his courses, and Artificial " "Intelligence is not mentioned."
                                ),
                            },
                            {
                                "statement": "John is a dedicated student.",
                                "verdict": 1,
                                "reason": (
                                    "The context mentions he spends significant time studying "
                                    "and stays late to work on projects."
                                ),
                            },
                            {
                                "statement": "John has a part-time job.",
                                "verdict": 0,
                                "reason": (
                                    "There is no information in the context about John " "having a part-time job."
                                ),
                            },
                        ]
                    },
                },
            ],
            llm=self.llm,
        )

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
            context = "\n".join(contexts)

            # Simplify statements
            simplify_result = self.statement_simplifier.run(
                question=[question],
                answer=[answer],
            )
            statements = simplify_result["results"][0]["statements"]

            # Check faithfulness of statements
            nli_result = self.nli_evaluator.run(
                context=[context],
                statements=[statements],
            )
            results = nli_result["results"][0]["results"]

            # Compute faithfulness score
            num_statements = len(results)
            num_faithful = sum(int(item["verdict"]) for item in results)
            score = num_faithful / num_statements if num_statements > 0 else 0.0
            final_scores.append(score)

            if verbose:
                logger.debug(f"Question: {question}")
                logger.debug(f"Answer: {answer}")
                logger.debug(f"Context: {context}")
                logger.debug("Simplified Statements:")
                logger.debug(statements)
                logger.debug("Faithfulness Results:")
                logger.debug(results)
                logger.debug(f"Faithfulness Score: {score}")
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
    questions = ["Who was Albert Einstein and what is he best known for?"]
    answers = [
        (
            "He was a German-born theoretical physicist, widely acknowledged to be one "
            "of the greatest and most influential physicists of all time. He was best "
            "known for developing the theory of relativity, he also made important "
            "contributions to the development of the theory of quantum mechanics."
        )
    ]
    contexts_list = [
        [
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
                "worldwide by the British journal Physics World, Einstein was "
                "ranked the greatest physicist of all time. His intellectual "
                "achievements and originality have made Einstein synonymous with genius."
            )
        ]
    ]

    # Initialize evaluator and evaluate
    evaluator = FaithfulnessEvaluator(llm)
    faithfulness_scores = evaluator.evaluate(questions, answers, contexts_list, verbose=True)

    # Print the results
    for idx, score in enumerate(faithfulness_scores):
        print(f"Question: {questions[idx]}")
        print(f"Faithfulness Score: {score}")
        print("-" * 50)

    print("Faithfulness Scores:")
    print(faithfulness_scores)
