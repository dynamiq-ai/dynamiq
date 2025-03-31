from typing import Any

from pydantic import BaseModel, PrivateAttr, field_validator, model_validator
from dynamiq.evaluations import BaseEvaluator
from dynamiq.evaluations.llm_evaluator import LLMEvaluator
from dynamiq.nodes.llms import BaseLLM
from dynamiq.utils.logger import logger


class DecomposeClaimsInput(BaseModel):
    """
    Input model for decomposing texts into claims.

    Attributes:
        texts (list[str]): List of texts to decompose.
    """
    texts: list[str]


class DecomposeClaimsOutput(BaseModel):
    """
    Output model for claim decomposition.

    Attributes:
        claims_list (list[list[str]]): List of lists of claims.
    """
    claims_list: list[list[str]]


class VerifyClaimsInput(BaseModel):
    """
    Input model for verifying claims against premises.

    Attributes:
        premises (list[str]): List of premises.
        claims_list (list[list[str]]): List of lists of claims.
    """
    premises: list[str]
    claims_list: list[list[str]]


class VerifyClaimsOutput(BaseModel):
    """
    Output model for claim verification.

    Attributes:
        verdicts_list (list[list[int]]): List of lists of verdicts (0 or 1).
    """
    verdicts_list: list[list[int]]


class RunInput(BaseModel):
    """
    Input model for running factual correctness evaluation.

    Attributes:
        answers (list[str]): List of response texts.
        contexts (list[str] | list[list[str]]): List of reference texts, or list of lists of
            reference texts.
        mode (str | None): Evaluation mode ('precision', 'recall', or 'f1').
        beta (float | None): Beta value for F-beta score.
        verbose (bool): Flag to enable verbose logging.
    """
    answers: list[str]
    contexts: list[str] | list[list[str]]
    mode: str | None = None
    beta: float | None = None
    verbose: bool = False

    @field_validator("contexts", mode="before")
    def unify_contexts(cls, value):
        """
        Allow contexts to be either list[str] or list[list[str]]. If list[list[str]],
        each sub-list is joined into one string. Otherwise, leave as-is.
        """
        if not isinstance(value, list):
            raise ValueError("contexts must be a list of strings or a list of lists of strings.")
        if all(isinstance(item, list) and all(isinstance(element, str) for element in item) for item in value):
            return [" ".join(sublist) for sublist in value]
        if all(isinstance(item, str) for item in value):
            return value
        raise ValueError("contexts must be either a list of strings or a list of lists of strings.")

    @model_validator(mode="after")
    def check_equal_length(self):
        """
        Confirm that answers and contexts have the same length.
        """
        if len(self.answers) != len(self.contexts):
            raise ValueError("answers and contexts must have the same length.")
        return self


class FactualCorrectnessRunResult(BaseModel):
    """
    Result model for factual correctness evaluation.

    Attributes:
        score (float): The computed factual correctness score.
        reasoning (str): Detailed reasoning explaining the evaluation.
    """
    score: float
    reasoning: str


class RunOutput(BaseModel):
    """
    Output model for factual correctness evaluation.

    Attributes:
        results (list[FactualCorrectnessRunResult]): List of results with score and reasoning.
    """
    results: list[FactualCorrectnessRunResult]


class FactualCorrectnessEvaluator(BaseEvaluator):
    """
    Evaluator class for factual correctness metric.

    Pipeline:
      1) Claim Decomposition: The answer and context are decomposed into standalone,
         verifiable claims.
      2) Claim Verification: The answer claims are verified against the context to compute
         precision (TP vs. FP). Optionally, context claims are verified against answer for
         recall (FN).
      3) Score Computation: Depending on mode, evaluate precision, recall, or F-beta score.
      4) Detailed Reasoning: Generates a user-friendly explanation describing each step,
         including claim lists, TP, FP, FN, and metric computations with emojis.

    Attributes:
        llm (BaseLLM): The language model to use for evaluation.
        mode (str): Evaluation mode ('precision', 'recall', or 'f1').
        beta (float): Beta value for F-beta score.
        atomicity (str): Level of atomicity ('low' or 'high').
        coverage (str): Level of coverage ('low' or 'high').
    """
    name: str = "FactualCorrectness"
    llm: BaseLLM
    mode: str = "f1"
    beta: float = 1.0
    atomicity: str = "low"
    coverage: str = "low"

    _claim_decomposer: LLMEvaluator = PrivateAttr()
    _nli_evaluator: LLMEvaluator = PrivateAttr()

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
                                "reason": "The premise states he was a German-born theoretical physicist.",
                            },
                            {
                                "claim": "Albert Einstein developed the theory of relativity.",
                                "verdict": 1,
                                "reason": "This is explicitly mentioned in the premise.",
                            },
                            {
                                "claim": "Albert Einstein contributed to quantum mechanics.",
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
            texts (list[str]): List of texts to decompose.

        Returns:
            list[list[str]]: List of lists of claims.
        """
        input_data = DecomposeClaimsInput(texts=texts)
        results = self._claim_decomposer.run(input_text=input_data.texts)
        claims_list = []
        for result in results["results"]:
            claims = result.get("claims")
            if isinstance(claims, list):
                claims_list.append(claims)
            else:
                claims_list.append([claims])
        output_data = DecomposeClaimsOutput(claims_list=claims_list)
        return output_data.claims_list

    def verify_claims(self, premises: list[str], claims_list: list[list[str]]) -> list[list[int]]:
        """
        Verify the claims against the premises.

        Args:
            premises (list[str]): List of premises.
            claims_list (list[list[str]]): List of lists of claims.

        Returns:
            list[list[int]]: List of lists of verdicts.
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

    def _build_reasoning(
        self,
        answer_claims: list[str],
        context_claims: list[str],
        answer_verdicts: list[int],
        context_verdicts: list[int],
        tp: int,
        fp: int,
        fn: int,
        score: float,
        mode: str,
        beta: float,
    ) -> str:
        """
        Build a detailed reasoning string for factual correctness evaluation.

        Explains:
        • How the answer and context were decomposed into claims.
        • How claim verification produced verdicts (TP, FP, FN) with emojis.
        • The calculation of the final score depending on the mode.

        Args:
            answer_claims (list[str]): Claims from the answer.
            context_claims (list[str]): Claims from the context.
            answer_verdicts (list[int]): Verdicts from verifying context claims against answer.
            context_verdicts (list[int]): Verdicts from verifying answer claims against context.
            tp (int): True positive count.
            fp (int): False positive count.
            fn (int): False negative count.
            score (float): Computed score.
            mode (str): Evaluation mode.
            beta (float): Beta value.

        Returns:
            str: Detailed reasoning.
        """
        lines = []
        lines.extend(["Reasoning:", "", "1. Claim Decomposition:", "   Answer was decomposed into claims:"])
        for claim in answer_claims:
            lines.append(f"     - {claim}")
        lines.extend(["   Context was decomposed into claims:"])
        for claim in context_claims:
            lines.append(f"     - {claim}")
        lines.extend(["", "2. Claim Verification:"])
        # Map verdicts to emojis: 1 -> ✅, 0 -> ❌
        mapped_context = [("✅" if v == 1 else "❌") for v in context_verdicts]
        lines.extend(
            [
                "   Verification of answer claims against context yields:",
                f"     Verdicts: {mapped_context}   (✅ = supported, ❌ = unsupported)",
                f"     TP (supported): {tp}",
                f"     FP (unsupported): {fp}",
            ]
        )
        if mode != "precision":
            mapped_answer = [("✅" if v == 1 else "❌") for v in answer_verdicts]
            lines.extend(
                [
                    "",
                    "   Verification of context claims against answer yields:",
                    f"     Verdicts: {mapped_answer}",
                    f"     FN (not supported): {fn}",
                ]
            )
        lines.append("")
        if mode == "precision":
            precision = tp / (tp + fp + 1e-8)
            lines.extend([f"Precision = TP/(TP+FP) = {precision:.2f}"])
        elif mode == "recall":
            recall = tp / (tp + fn + 1e-8)
            lines.extend([f"Recall = TP/(TP+FN) = {recall:.2f}"])
        else:
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8) if (tp + fn) > 0 else 0.0
            lines.extend(
                [
                    f"Precision = TP/(TP+FP) = {precision:.2f}",
                    f"Recall = TP/(TP+FN) = {recall:.2f}",
                    f"F-beta Score (beta={beta:.2f}) = {score:.2f}",
                ]
            )
        lines.extend(["", f"Final Score = {score:.2f}"])
        return "\n".join(lines)

    def run_single(
        self, answer: str, context: str, mode: str | None = None, beta: float | None = None, verbose: bool = False
    ) -> FactualCorrectnessRunResult:
        """
        Evaluate the factual correctness for a single sample.

        Args:
            answer (str): The response text.
            context (str): The reference text.
            mode (str | None): Evaluation mode ('precision', 'recall', or 'f1').
            beta (float | None): Beta value for F-beta score.
            verbose (bool): Flag for verbose logging.

        Returns:
            FactualCorrectnessRunResult: The computed factual correctness score and detailed reasoning.
        """
        evaluation_mode = mode or self.mode
        beta_value = beta or self.beta

        answer_claims_list = self.decompose_claims([answer])
        if not answer_claims_list or answer_claims_list[0] is None:
            if verbose:
                logger.debug(f"No claims decomposed for answer: {answer}. Using empty list.")
            answer_claims = []
        else:
            answer_claims = answer_claims_list[0]

        context_claims_list = self.decompose_claims([context])
        if not context_claims_list or context_claims_list[0] is None:
            if verbose:
                logger.debug(f"No claims decomposed for context: {context}. Using empty list.")
            context_claims = []
        else:
            context_claims = context_claims_list[0]

        # Verify answer claims against context (precision part).
        context_verdicts_list = self.verify_claims(premises=[context], claims_list=[answer_claims])
        if not context_verdicts_list or context_verdicts_list[0] is None:
            if verbose:
                logger.debug(f"No verdicts returned when verifying answer claims against context for answer: {answer}")
            context_verdicts = []
        else:
            context_verdicts = context_verdicts_list[0]
        tp = sum(context_verdicts)
        fp = len(context_verdicts) - tp

        # For recall or F1, verify context claims against answer.
        if evaluation_mode not in ("precision", "PRECISION"):
            answer_verdicts_list = self.verify_claims(premises=[answer], claims_list=[context_claims])
            if not answer_verdicts_list or answer_verdicts_list[0] is None:
                if verbose:
                    logger.debug(
                        f"No verdicts returned when verifying context claims against answer for answer: {answer}"
                    )
                answer_verdicts = []
                fn = 0
            else:
                answer_verdicts = answer_verdicts_list[0]
                fn = sum(1 - v for v in answer_verdicts)
        else:
            answer_verdicts = []
            fn = 0

        if evaluation_mode == "precision":
            computed_score = tp / (tp + fp + 1e-8)
        elif evaluation_mode == "recall":
            computed_score = tp / (tp + fn + 1e-8)
        else:
            computed_score = self.fbeta_score(tp, fp, fn, beta_value)

        reasoning_text = self._build_reasoning(
            answer_claims=answer_claims,
            context_claims=context_claims,
            answer_verdicts=answer_verdicts,
            context_verdicts=context_verdicts,
            tp=tp,
            fp=fp,
            fn=fn,
            score=computed_score,
            mode=evaluation_mode,
            beta=beta_value,
        )

        if verbose:
            logger.debug(f"Answer: {answer}")
            logger.debug(f"Context: {context}")
            logger.debug(f"Answer Claims: {answer_claims}")
            logger.debug(f"Context Claims: {context_claims}")
            logger.debug(f"TP: {tp}, FP: {fp}, FN: {fn}")
            logger.debug(f"Score: {computed_score}")
            logger.debug(reasoning_text)
            logger.debug("-" * 50)

        return FactualCorrectnessRunResult(score=round(computed_score, 2), reasoning=reasoning_text)

    def run(
        self,
        answers: list[str],
        contexts: list[str] | list[list[str]],
        mode: str | None = None,
        beta: float | None = None,
        verbose: bool = False,
    ) -> RunOutput:
        """
        Evaluate the factual correctness of answers against contexts.

        Pipeline:
        1) Decompose both answer and context into claims.
        2) Verify answer claims against context to compute precision.
        3) If mode is recall or F1, verify context claims against answer
           to compute false negatives.
        4) Compute the final score based on the selected mode.
        5) Generate detailed reasoning regarding the claim decomposition,
           verification, and final metric calculations with emojis.

        Args:
            answers (list[str]): List of response texts.
            contexts (list[str] | list[list[str]]): List of context texts.
            mode (str | None): Evaluation mode ('precision', 'recall', or 'f1').
            beta (float | None): Beta value for F-beta score.
            verbose (bool): Flag for verbose logging.

        Returns:
            RunOutput: Contains a list of FactualCorrectnessRunResult.
        """
        run_input = RunInput(answers=answers, contexts=contexts, mode=mode, beta=beta, verbose=verbose)
        evaluation_mode = run_input.mode or self.mode
        beta_value = run_input.beta or self.beta

        results_output = []
        for index in range(len(run_input.answers)):
            answer_sample = run_input.answers[index]
            context_sample = run_input.contexts[index]
            result_single = self.run_single(
                answer=answer_sample,
                context=context_sample,
                mode=evaluation_mode,
                beta=beta_value,
                verbose=run_input.verbose,
            )
            results_output.append(result_single)
        return RunOutput(results=results_output)
