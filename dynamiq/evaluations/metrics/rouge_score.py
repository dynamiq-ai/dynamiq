from enum import Enum
from typing import Callable

from pydantic import BaseModel, PrivateAttr, model_validator
from dynamiq.evaluations import BaseEvaluator


class RougeType(str, Enum):
    """
    Enumeration of supported ROUGE types.
    """
    rouge1 = "rouge1"
    rouge2 = "rouge2"
    rougeL = "rougeL"


class MeasureType(str, Enum):
    """
    Enumeration of measurement types for ROUGE scores.
    """
    fmeasure = "fmeasure"
    precision = "precision"
    recall = "recall"


class RunInput(BaseModel):
    """
    Input model for batch ROUGE score evaluation.

    Attributes:
        ground_truth_answers (list[str]): List of reference strings.
        answers (list[str]): List of candidate response strings.
    """
    ground_truth_answers: list[str]
    answers: list[str]

    @model_validator(mode="after")
    def check_equal_length(self) -> "RunInput":
        """
        Validate that the number of ground truth answers matches the number of answers.

        Raises:
            ValueError: If the lengths of `ground_truth_answers` and `answers` do not match.

        Returns:
            RunInput: The validated instance.
        """
        if len(self.ground_truth_answers) != len(self.answers):
            raise ValueError("ground_truth_answers and answers must have the same length.")
        return self


class RunOutput(BaseModel):
    """
    Output model for ROUGE score evaluation.

    Attributes:
        scores (list[float]): List of computed ROUGE scores.
    """

    scores: list[float]


class RunSingleInput(BaseModel):
    """
    Single-run input model for ROUGE score evaluation.

    Attributes:
        ground_truth_answer (str): The reference string.
        answer (str): The candidate string.
    """

    ground_truth_answer: str
    answer: str


class RunSingleOutput(BaseModel):
    """
    Single-run output model for ROUGE score evaluation.

    Attributes:
        score (float): The computed ROUGE score.
    """

    score: float


class RougeScoreEvaluator(BaseEvaluator):
    """
    Evaluates ROUGE scores using the rouge_score library.

    Attributes:
        name (str): Name of the evaluator. Defaults to "RougeScore".
        rouge_type (RougeType): ROUGE variant to compute. Defaults to RougeType.rougeL.
        measure_type (MeasureType): The field of the metric to retrieve. Defaults to MeasureType.fmeasure.
    """
    name: str = "RougeScore"
    rouge_type: RougeType = RougeType.rougeL
    measure_type: MeasureType = MeasureType.fmeasure

    _scorer: Callable = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self._initialize_rouge()

    def _initialize_rouge(self) -> None:
        from rouge_score import rouge_scorer

        self._scorer = rouge_scorer.RougeScorer([self.rouge_type.value], use_stemmer=True)

    def run_single(self, ground_truth_answer: str, answer: str) -> float:
        """
        Compute the ROUGE score for a single pair of ground truth (reference) and answer.

        Args:
            ground_truth_answer (str): The reference string.
            answer (str): The candidate string.

        Returns:
            float: The computed ROUGE score.
        """
        # Validate input.
        single_input = RunSingleInput(ground_truth_answer=ground_truth_answer, answer=answer)
        rouge_result = self._scorer.score(single_input.ground_truth_answer, single_input.answer)
        metric_value = getattr(rouge_result[self.rouge_type.value], self.measure_type.value)
        score = round(float(metric_value), 2)

        output = RunSingleOutput(score=score)
        return output.score

    def run(self, ground_truth_answers: list[str], answers: list[str]) -> list[float]:
        """
        Compute ROUGE scores for each reference-response pair in batch.

        Args:
            ground_truth_answers (list[str]): List of reference strings.
            answers (list[str]): List of candidate strings.

        Returns:
            list[float]: List of computed ROUGE scores.
        """
        input_data = RunInput(ground_truth_answers=ground_truth_answers, answers=answers)
        scores = []
        for gt, ans in zip(input_data.ground_truth_answers, input_data.answers):
            score = self.run_single(ground_truth_answer=gt, answer=ans)
            scores.append(score)
        output_data = RunOutput(scores=scores)
        return output_data.scores
