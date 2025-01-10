import logging
from functools import cached_property
from typing import Literal

from pydantic import BaseModel, PrivateAttr, model_validator

logger = logging.getLogger(__name__)

try:
    from rouge_score import rouge_scorer
except ImportError as e:
    raise ImportError(f"{e.name} is required for RougeScore. Please install it using: pip install {e.name}")


class SingleTurnMetricInput(BaseModel):
    """
    Input model for a single-turn ROUGE metric.

    Attributes:
        references (List[str]): List of reference strings, one per example.
        responses (List[str]): List of response strings, one per example.
    """

    references: list[str]
    responses: list[str]

    @model_validator(mode="after")
    def check_equal_length(self):
        if len(self.references) != len(self.responses):
            raise ValueError("References and responses must have the same length.")
        return self


class SingleTurnMetricOutput(BaseModel):
    """
    Output model for a single-turn ROUGE metric.

    Attributes:
        scores (List[float]): List of ROUGE scores, one per reference/response pair.
    """

    scores: list[float]


class BaseStringMetric(BaseModel):
    """
    Base class for string metrics.

    Attributes:
        name (str): Name of the metric.
    """

    name: str

    class Config:
        arbitrary_types_allowed = True

    @cached_property
    def type(self) -> str:
        """
        Returns a string indicating the fully qualified name of this class.
        """
        return f"{self.__module__}.{self.__class__.__name__}"

    def run(self, references: list[str], responses: list[str]) -> list[float]:
        """
        Runs the metric on the provided references and responses.
        Must be overridden by subclasses.

        Args:
            references (List[str]): Ground-truth/reference strings.
            responses (List[str]): System-generated/response strings.

        Returns:
            List[float]: Score for each reference/response pair.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class RougeScoreMetric(BaseStringMetric):
    """
    Implements a ROUGE score metric using rouge_score's RougeScorer.

    Attributes:
        name (str): Name of the metric. Defaults to "rouge_score".
        rouge_type (Literal["rouge1", "rougeL"]): ROUGE variant to compute. Defaults to "rougeL".
        measure_type (Literal["fmeasure", "precision", "recall"]): The field of the metric to retrieve.
            Defaults to "fmeasure".
    """

    name: str = "rouge_score"
    rouge_type: Literal["rouge1", "rougeL"] = "rougeL"
    measure_type: Literal["fmeasure", "precision", "recall"] = "fmeasure"

    # Private attribute to store the RougeScorer initialization
    _scorer_class = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self._initialize_rouge()

    def _initialize_rouge(self):
        """
        Initialize the rouge_scorer with the requested rouge_type and a stemmer.
        """
        self._scorer_class = rouge_scorer.RougeScorer

    def run(self, references: list[str], responses: list[str]) -> list[float]:
        """
        Compute the ROUGE scores for each reference/response pair.

        Args:
            references (List[str]): List of reference strings.
            responses (List[str]): List of response/hypothesis strings.

        Returns:
            List[float]: ROUGE scores, one per pair, extracted from the measure_type (fmeasure, precision, recall).
        """
        input_data = SingleTurnMetricInput(references=references, responses=responses)
        scores = []

        for ref, resp in zip(input_data.references, input_data.responses):
            scorer = self._scorer_class([self.rouge_type], use_stemmer=True)
            rouge_result = scorer.score(ref, resp)
            # e.g. rouge_result["rouge1"].fmeasure / precision / recall
            metric_value = getattr(rouge_result[self.rouge_type], self.measure_type)
            scores.append(metric_value)

        output_data = SingleTurnMetricOutput(scores=scores)
        return output_data.scores
