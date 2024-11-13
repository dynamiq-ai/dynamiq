import logging

from dynamiq.components.evaluators.llm_evaluator import LLMEvaluator
from dynamiq.nodes.llms import BaseLLM, OpenAI

# Configure logging
logger = logging.getLogger(__name__)


class FactualCorrectnessEvaluator:
    def __init__(
        self,
        llm: BaseLLM,
        mode: str = "f1",
        beta: float = 1.0,
        atomicity: str = "low",
        coverage: str = "low",
    ):
        self.llm = llm
        self.mode = mode
        self.beta = beta
        self.atomicity = atomicity
        self.coverage = coverage
        self._initialize_evaluators()

    def _initialize_evaluators(self):
        # Claim Decomposition Evaluator
        decomposition_instructions = (
            "Decompose the 'Input Text' into standalone factual claims.\n"
            "- Each claim should be a simple, verifiable statement.\n"
            "- Do not include personal opinions or interpretations.\n"
            "- Output a JSON object with key 'claims' containing the list of claims.\n"
            "- Ensure your response is valid JSON, using double quotes for all strings."
        )

        self.claim_decomposer = LLMEvaluator(
            instructions=decomposition_instructions.strip(),
            inputs=[("input_text", list[str])],
            outputs=["claims"],
            examples=[
                {
                    "inputs": {
                        "input_text": [
                            "Albert Einstein was a German theoretical physicist. "
                            "He developed the theory of relativity and contributed "
                            "to quantum mechanics."
                        ]
                    },
                    "outputs": {
                        "claims": [
                            "Albert Einstein was a German theoretical physicist.",
                            "Albert Einstein developed the theory of relativity.",
                            "Albert Einstein contributed to quantum mechanics.",
                        ]
                    },
                },
            ],
            llm=self.llm,
        )

        # NLI Evaluator
        nli_instructions = (
            "For each 'Claim', determine if it is supported by the 'Premise'.\n"
            "- Return 'verdict': 1 for supported, 0 for unsupported claims.\n"
            "- Provide a brief 'reason' for each verdict.\n"
            "- Output a JSON object with key 'results' containing a list of verdicts.\n"
            "- Each item should have keys 'claim', 'verdict', and 'reason'.\n"
            "- Ensure your response is valid JSON, using double quotes for all strings."
        )

        self.nli_evaluator = LLMEvaluator(
            instructions=nli_instructions.strip(),
            inputs=[
                ("premise", list[str]),
                ("claims", list[list[str]]),
            ],
            outputs=["results"],
            examples=[
                {
                    "inputs": {
                        "premise": [
                            "Albert Einstein was a German-born theoretical physicist. "
                            "He developed the theory of relativity."
                        ],
                        "claims": [
                            [
                                "Albert Einstein was a German theoretical physicist.",
                                "Albert Einstein developed the theory of relativity.",
                                "Albert Einstein contributed to quantum mechanics.",
                            ]
                        ],
                    },
                    "outputs": {
                        "results": [
                            {
                                "claim": "Albert Einstein was a German theoretical physicist.",
                                "verdict": 1,
                                "reason": ("The premise states he was a German-born theoretical physicist."),
                            },
                            {
                                "claim": "Albert Einstein developed the theory of relativity.",
                                "verdict": 1,
                                "reason": "This is explicitly mentioned in the premise.",
                            },
                            {
                                "claim": "Albert Einstein contributed to quantum mechanics.",
                                "verdict": 0,
                                "reason": ("The premise does not mention contributions to quantum mechanics."),
                            },
                        ]
                    },
                },
            ],
            llm=self.llm,
        )

    def decompose_claims(self, texts: list[str]) -> list[list[str]]:
        # Decompose each text into claims
        results = self.claim_decomposer.run(input_text=texts)
        claims_list = [result["claims"] for result in results["results"]]
        return claims_list

    def verify_claims(self, premises: list[str], claims_list: list[list[str]]) -> list[list[int]]:
        # Verify the claims
        results = self.nli_evaluator.run(premise=premises, claims=claims_list)
        verdicts_list = []
        for result in results["results"]:
            verdicts = [int(item["verdict"]) for item in result["results"]]
            verdicts_list.append(verdicts)
        return verdicts_list

    def fbeta_score(self, tp: int, fp: int, fn: int) -> float:
        beta = self.beta
        precision = tp / (tp + fp + 1e-8) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn + 1e-8) if (tp + fn) > 0 else 0.0
        if (precision + recall) == 0:
            return 0.0
        score = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-8)
        return score

    def evaluate(
        self,
        responses: list[str],
        references: list[str],
        mode: str = None,
        beta: float = None,
        verbose: bool = False,
    ) -> list[float]:
        if not (len(responses) == len(references)):
            raise ValueError("Responses and references must have the same length.")
        if mode is None:
            mode = self.mode
        if beta is None:
            beta = self.beta

        final_scores = []

        for idx in range(len(responses)):
            response = responses[idx]
            reference = references[idx]

            # Decompose claims
            response_claims_list = self.decompose_claims([response])
            reference_claims_list = self.decompose_claims([reference])

            response_claims = response_claims_list[0]
            reference_claims = reference_claims_list[0]

            # Verify response claims against reference (precision)
            reference_response_verdicts_list = self.verify_claims(premises=[reference], claims_list=[response_claims])
            reference_response_verdicts = reference_response_verdicts_list[0]

            tp = sum(reference_response_verdicts)
            fp = len(reference_response_verdicts) - tp

            if mode != "precision":
                # Verify reference claims against response (recall)
                response_reference_verdicts_list = self.verify_claims(
                    premises=[response], claims_list=[reference_claims]
                )
                response_reference_verdicts = response_reference_verdicts_list[0]
                fn = sum(1 - v for v in response_reference_verdicts)
            else:
                fn = 0

            if mode == "precision":
                score = tp / (tp + fp + 1e-8)
            elif mode == "recall":
                score = tp / (tp + fn + 1e-8)
            else:  # F1 or F-beta
                score = self.fbeta_score(tp, fp, fn)

            final_scores.append(score)

            if verbose:
                logger.debug(f"Response: {response}")
                logger.debug(f"Reference: {reference}")
                logger.debug(f"Response Claims: {response_claims}")
                logger.debug(f"Reference Claims: {reference_claims}")
                logger.debug(f"TP: {tp}, FP: {fp}, FN: {fn}")
                logger.debug(f"Score: {score}")
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

    # Sample data (can be replaced with your data)
    responses = [
        (
            "Albert Einstein was a German theoretical physicist. "
            "He developed the theory of relativity and contributed "
            "to quantum mechanics."
        ),
        ("The Eiffel Tower is located in Berlin, Germany. " "It was constructed in 1889."),
    ]
    references = [
        ("Albert Einstein was a German-born theoretical physicist. " "He developed the theory of relativity."),
        ("The Eiffel Tower is located in Paris, France. " "It was constructed in 1887 and opened in 1889."),
    ]

    # Initialize evaluator and evaluate
    evaluator = FactualCorrectnessEvaluator(llm)
    correctness_scores = evaluator.evaluate(responses, references, verbose=True)

    # Print the results
    for idx, score in enumerate(correctness_scores):
        print(f"Response: {responses[idx]}")
        print(f"Factual Correctness Score: {score}")
        print("-" * 50)

    print("Factual Correctness Scores:")
    print(correctness_scores)
