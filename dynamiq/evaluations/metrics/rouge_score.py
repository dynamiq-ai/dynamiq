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
    Input model for ROUGE score evaluation.

    Attributes:
        ground_truth_answers (list[str]): List of reference strings, one per example.
        answers (list[str]): List of response strings, one per example.
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
        scores (list[float]): List of ROUGE scores, one per reference/response pair.
    """

    scores: list[float]


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

    # Private attribute to store the rouge_scorer.RougeScorer instance
    _scorer: Callable = PrivateAttr()

    def __init__(self, **data):
        """
        Initialize the RougeScoreEvaluator instance and set up the rouge_scorer.RougeScorer.

        Args:
            **data: Arbitrary keyword arguments for the BaseModel.
        """
        super().__init__(**data)
        self._initialize_rouge()

    def _initialize_rouge(self) -> None:
        """
        Initialize the RougeScorer from rouge_score.

        Raises:
            ImportError: If rouge_score is not installed.
        """
        from rouge_score import rouge_scorer

        self._scorer = rouge_scorer.RougeScorer([self.rouge_type.value], use_stemmer=True)

    def run(self, ground_truth_answers: list[str], answers: list[str]) -> list[float]:
        """
        Compute ROUGE scores for each reference-response pair.

        Args:
            ground_truth_answers (list[str]): List of reference strings.
            answers (list[str]): List of response strings.

        Returns:
            list[float]: ROUGE scores, one per pair.
        """
        input_data = RunInput(ground_truth_answers=ground_truth_answers, answers=answers)
        scores: list[float] = []

        for ref, resp in zip(input_data.ground_truth_answers, input_data.answers):
            rouge_result = self._scorer.score(ref, resp)
            # e.g., rouge_result["rougeL"].fmeasure / precision / recall
            metric_value = getattr(rouge_result[self.rouge_type.value], self.measure_type.value)
            rouge_score = round(float(metric_value), 2)
            scores.append(rouge_score)

        output_data = RunOutput(scores=scores)
        return output_data.scores
