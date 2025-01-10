import logging
from enum import Enum
from functools import cached_property

from pydantic import BaseModel, PrivateAttr, model_validator
from rapidfuzz import distance

logger = logging.getLogger(__name__)


class DistanceMeasure(str, Enum):
    LEVENSHTEIN = "levenshtein"
    HAMMING = "hamming"
    JARO = "jaro"
    JARO_WINKLER = "jaro_winkler"


class SingleTurnMetricInput(BaseModel):
    """
    Input model for a single-turn string metric.

    Attributes:
        references (List[str]): List of reference strings.
        responses (List[str]): List of response strings.
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
    Output model for a single-turn string metric.

    Attributes:
        scores (List[float]): List of metric scores, one per reference/response pair.
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
        arbitrary_types_allowed = True  # If you need to store anything that isn't a built-in type

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
            references (List[str]): The ground-truth/reference strings.
            responses (List[str]): The system-generated/response strings.

        Returns:
            List[float]: A score for each reference/response pair.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class ExactMatchMetric(BaseStringMetric):
    """
    A metric that checks for exact string matches.

    Attributes:
        name (str): Name of the metric. Defaults to "exact_match".
    """

    name: str = "exact_match"

    def run(self, references: list[str], responses: list[str]) -> list[float]:
        input_data = SingleTurnMetricInput(references=references, responses=responses)
        scores = []
        for ref, resp in zip(input_data.references, input_data.responses):
            scores.append(float(ref == resp))
        output_data = SingleTurnMetricOutput(scores=scores)
        return output_data.scores


class StringPresenceMetric(BaseStringMetric):
    """
    A metric that checks if the reference string is present (substring) in the response.

    Attributes:
        name (str): Name of the metric. Defaults to "string_presence".
    """

    name: str = "string_presence"

    def run(self, references: list[str], responses: list[str]) -> list[float]:
        input_data = SingleTurnMetricInput(references=references, responses=responses)
        scores = []
        for ref, resp in zip(input_data.references, input_data.responses):
            scores.append(float(ref in resp))
        output_data = SingleTurnMetricOutput(scores=scores)
        return output_data.scores


class NonLLMStringSimilarityMetric(BaseStringMetric):
    """
    A metric that calculates a similarity score ((1 - normalized_distance) using rapidfuzz)
    between reference and response strings.

    Attributes:
        name (str): Name of the metric. Defaults to "non_llm_string_similarity".
        distance_measure (DistanceMeasure): Which distance measure to use.
    """

    name: str = "non_llm_string_similarity"
    distance_measure: DistanceMeasure = DistanceMeasure.LEVENSHTEIN

    # Store the distance functions in a private attribute (initialized once).
    _distance_map = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self._initialize_distance_map()

    def _initialize_distance_map(self):
        """
        Sets up a map from the enum DistanceMeasure to the actual rapidfuzz distance classes.
        """
        self._distance_map = {
            DistanceMeasure.LEVENSHTEIN: distance.Levenshtein,
            DistanceMeasure.HAMMING: distance.Hamming,
            DistanceMeasure.JARO: distance.Jaro,
            DistanceMeasure.JARO_WINKLER: distance.JaroWinkler,
        }

    def run(self, references: list[str], responses: list[str]) -> list[float]:
        input_data = SingleTurnMetricInput(references=references, responses=responses)
        dist_func = self._distance_map[self.distance_measure]
        scores = []
        for ref, resp in zip(input_data.references, input_data.responses):
            # 1 - normalized_distance => similarity
            sim_score = 1 - dist_func.normalized_distance(ref, resp)
            scores.append(float(sim_score))
        output_data = SingleTurnMetricOutput(scores=scores)
        return output_data.scores
