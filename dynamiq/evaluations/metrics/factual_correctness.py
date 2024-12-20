import logging
from functools import cached_property
from typing import Any

from pydantic import BaseModel, PrivateAttr, computed_field, model_validator

from dynamiq.components.evaluators.llm_evaluator import LLMEvaluator
from dynamiq.nodes.llms import BaseLLM

# Configure logging
logger = logging.getLogger(__name__)


class DecomposeClaimsInput(BaseModel):
    """
    Input model for decomposing texts into claims.

    Attributes:
        texts (List[str]): List of texts to decompose.
    """

    texts: list[str]


class DecomposeClaimsOutput(BaseModel):
    """
    Output model for claim decomposition.

    Attributes:
        claims_list (List[List[str]]): List of lists of claims.
    """

    claims_list: list[list[str]]


class VerifyClaimsInput(BaseModel):
    """
    Input model for verifying claims against premises.

    Attributes:
        premises (List[str]): List of premises.
        claims_list (List[List[str]]): List of lists of claims.
    """

    premises: list[str]
    claims_list: list[list[str]]


class VerifyClaimsOutput(BaseModel):
    """
    Output model for claim verification.

    Attributes:
        verdicts_list (List[List[int]]): List of lists of verdicts (0 or 1).
    """

    verdicts_list: list[list[int]]


class RunInput(BaseModel):
    """
    Input model for running factual correctness evaluation.

    Attributes:
        answer (List[str]): List of response texts.
        context (List[str]): List of reference texts.
        mode (Optional[str]): Evaluation mode ('precision', 'recall', or 'f1').
        beta (Optional[float]): Beta value for F-beta score.
        verbose (bool): Flag to enable verbose logging.
    """

    answers: list[str]
    contexts: list[str]
    mode: str | None = None
    beta: float | None = None
    verbose: bool = False

    @model_validator(mode="after")
    def check_equal_length(self):
        if len(self.answers) != len(self.contexts):
            raise ValueError("Answer and context must have the same length.")
        return self


class RunOutput(BaseModel):
    """
    Output model for factual correctness evaluation.

    Attributes:
        final_scores (List[float]): List of factual correctness scores.
    """

    final_scores: list[float]


class FactualCorrectnessEvaluator(BaseModel):
    """
    Evaluator class for factual correctness metric.

    Attributes:
        llm (BaseLLM): The language model to use for evaluation.
        mode (str): Evaluation mode ('precision', 'recall', or 'f1').
        beta (float): Beta value for F-beta score.
        atomicity (str): Level of atomicity ('low' or 'high').
        coverage (str): Level of coverage ('low' or 'high').
    """

    llm: BaseLLM
    mode: str = "f1"
    beta: float = 1.0
    atomicity: str = "low"
    coverage: str = "low"

    _claim_decomposer: LLMEvaluator = PrivateAttr()
    _nli_evaluator: LLMEvaluator = PrivateAttr()

    class Config:
        arbitrary_types_allowed = True  # Allow arbitrary types

    @computed_field
    @cached_property
    def type(self) -> str:
        return f"{self.__module__.rsplit('.', 1)[0]}.{self.__class__.__name__}"

    def __init__(self, **data):
        super().__init__(**data)
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

        self._claim_decomposer = LLMEvaluator(
            instructions=decomposition_instructions.strip(),
            inputs=[{"name": "input_text", "type": list[str]}],
            outputs=[{"name": "claims", "type": list[str]}],
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

        self._nli_evaluator = LLMEvaluator(
            instructions=nli_instructions.strip(),
            inputs=[
                {"name": "premise", "type": list[str]},
                {"name": "claims", "type": list[list[str]]},
            ],
            outputs=[{"name": "results", "type": list[dict[str, Any]]}],
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
                                "He developed the theory of relativity.",
                                "He contributed to quantum mechanics.",
                            ]
                        ],
                    },
                    "outputs": {
                        "results": [
                            {
                                "claim": "Albert Einstein was a German theoretical physicist.",
                                "verdict": 1,
                                "reason": "The premise states he was a German-born theoretical physicist.",
                            },
                            {
                                "claim": "He developed the theory of relativity.",
                                "verdict": 1,
                                "reason": "This is explicitly mentioned in the premise.",
                            },
                            {
                                "claim": "He contributed to quantum mechanics.",
                                "verdict": 0,
                                "reason": "The premise does not mention contributions to quantum mechanics.",
                            },
                        ]
                    },
                },
            ],
            llm=self.llm,
        )

    def decompose_claims(self, texts: list[str]) -> list[list[str]]:
        """
        Decompose each text into claims.

        Args:
            texts (List[str]): List of texts to decompose.

        Returns:
            List[List[str]]: List of lists of claims.
        """
        input_data = DecomposeClaimsInput(texts=texts)
        results = self._claim_decomposer.run(input_text=input_data.texts)
        claims_list = []
        for result in results["results"]:
            claims = result.get("claims")
            if isinstance(claims, list):
                claims_list.append(claims)
            else:
                # If claims is a single string, wrap it in a list
                claims_list.append([claims])
        output_data = DecomposeClaimsOutput(claims_list=claims_list)
        return output_data.claims_list

    def verify_claims(self, premises: list[str], claims_list: list[list[str]]) -> list[list[int]]:
        """
        Verify the claims against the premises.

        Args:
            premises (List[str]): List of premises.
            claims_list (List[List[str]]): List of lists of claims.

        Returns:
            List[List[int]]: List of lists of verdicts.
        """
        input_data = VerifyClaimsInput(premises=premises, claims_list=claims_list)
        results = self._nli_evaluator.run(
            premise=input_data.premises,
            claims=input_data.claims_list,
        )
        verdicts_list = []
        for result in results["results"]:
            verdicts_raw = result["results"]
            verdicts = []
            for item in verdicts_raw:
                verdict = int(item["verdict"])
                verdicts.append(verdict)
            verdicts_list.append(verdicts)
        output_data = VerifyClaimsOutput(verdicts_list=verdicts_list)
        return output_data.verdicts_list

    def fbeta_score(self, tp: int, fp: int, fn: int, beta: float) -> float:
        """
        Calculate the F-beta score.

        Args:
            tp (int): True positives.
            fp (int): False positives.
            fn (int): False negatives.
            beta (float): Beta value.

        Returns:
            float: F-beta score.
        """
        precision = tp / (tp + fp + 1e-8) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn + 1e-8) if (tp + fn) > 0 else 0.0
        if (precision + recall) == 0:
            return 0.0
        score = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-8)
        return score

    def run(
        self,
        answers: list[str],
        contexts: list[str],
        mode: str | None = None,
        beta: float | None = None,
        verbose: bool = False,
    ) -> list[float]:
        """
        Evaluate the factual correctness of answers against contexts.

        Args:
            answer (List[str]): List of response texts.
            context (List[str]): List of reference texts.
            mode (Optional[str]): Evaluation mode ('precision', 'recall', or 'f1').
            beta (Optional[float]): Beta value for F-beta score.
            verbose (bool): Flag to enable verbose logging.

        Returns:
            List[float]: List of factual correctness scores.
        """
        input_data = RunInput(
            answers=answers,
            contexts=contexts,
            mode=mode,
            beta=beta,
            verbose=verbose,
        )
        mode = input_data.mode or self.mode
        beta = input_data.beta or self.beta

        final_scores = []

        for idx in range(len(input_data.answers)):
            single_answer = input_data.answers[idx]
            single_context = input_data.contexts[idx]

            # Decompose claims from answer and context
            answer_claims_list = self.decompose_claims([single_answer])
            context_claims_list = self.decompose_claims([single_context])

            answer_claims = answer_claims_list[0]
            context_claims = context_claims_list[0]

            # Verify answer claims against context (precision)
            context_verdicts_list = self.verify_claims(
                premises=[single_context],
                claims_list=[answer_claims],
            )
            context_verdicts = context_verdicts_list[0]

            tp = sum(context_verdicts)
            fp = len(context_verdicts) - tp

            if mode != "precision":
                # Verify context claims against answer (recall)
                answer_verdicts_list = self.verify_claims(
                    premises=[single_answer],
                    claims_list=[context_claims],
                )
                answer_verdicts = answer_verdicts_list[0]
                fn = sum(1 - v for v in answer_verdicts)
            else:
                fn = 0

            if mode == "precision":
                score = tp / (tp + fp + 1e-8)
            elif mode == "recall":
                score = tp / (tp + fn + 1e-8)
            else:  # 'f1' or F-beta
                score = self.fbeta_score(tp, fp, fn, beta)

            final_scores.append(score)

            if input_data.verbose:
                logger.debug(f"Answer: {single_answer}")
                logger.debug(f"Context: {single_context}")
                logger.debug(f"Answer Claims: {answer_claims}")
                logger.debug(f"Context Claims: {context_claims}")
                logger.debug(f"TP: {tp}, FP: {fp}, FN: {fn}")
                logger.debug(f"Score: {score}")
                logger.debug("-" * 50)

        output_data = RunOutput(final_scores=final_scores)
        return output_data.final_scores
