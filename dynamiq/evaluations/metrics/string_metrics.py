import logging
from enum import Enum
from functools import cached_property
from typing import Callable

from pydantic import BaseModel, ConfigDict, PrivateAttr, computed_field, model_validator

logger = logging.getLogger(__name__)


class DistanceMeasure(str, Enum):
    """
    Enumeration of supported distance measures for string similarity.
    """

    LEVENSHTEIN = "levenshtein"
    HAMMING = "hamming"
    JARO = "jaro"
    JARO_WINKLER = "jaro_winkler"


class RunInput(BaseModel):
    """
    Input model for string evaluators.

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
    Output model for string evaluators.

    Attributes:
        scores (list[float]): List of evaluator scores, one per reference/answer pair.
    """

    scores: list[float]


class BaseStringEvaluator(BaseModel):
    """
    Base class for string evaluators.

    Attributes:
        name (str): Name of the evaluator.
    """

    name: str

    # Configuration using ConfigDict for Pydantic v2 compliance
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @computed_field
    @cached_property
    def type(self) -> str:
        """
        Compute the type identifier for the evaluator.

        Returns:
            str: A string representing the module and class name.
        """
        return f"{self.__module__.rsplit('.', 1)[0]}.{self.__class__.__name__}"

    def run(self, ground_truth_answers: list[str], answers: list[str]) -> list[float]:
        """
        Executes the evaluator on the provided ground truth answers and answers.
        Must be overridden by subclasses.

        Args:
            ground_truth_answers (list[str]): The ground-truth/reference strings.
            answers (list[str]): The system-generated/response strings.

        Returns:
            list[float]: Scores for each reference/answer pair.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class ExactMatchEvaluator(BaseStringEvaluator):
    """
    An evaluator that checks for exact string matches.

    Attributes:
        name (str): Name of the evaluator. Defaults to "exact_match".
    """

    name: str = "exact_match"

    def run(self, ground_truth_answers: list[str], answers: list[str]) -> list[float]:
        """
        Compute exact match scores for each reference/answer pair.

        Args:
            ground_truth_answers (list[str]): List of reference strings.
            answers (list[str]): List of answer strings.

        Returns:
            list[float]: Exact match scores (1.0 for match, 0.0 otherwise) for each pair.
        """
        input_data = RunInput(ground_truth_answers=ground_truth_answers, answers=answers)
        scores: list[float] = []

        for idx, (ref, ans) in enumerate(zip(input_data.ground_truth_answers, input_data.answers)):
            exact_match = float(ref == ans)
            logger.debug(f"Processing pair {idx + 1}: Exact Match = {exact_match}")
            scores.append(exact_match)

        output_data = RunOutput(scores=scores)
        return output_data.scores


class StringPresenceEvaluator(BaseStringEvaluator):
    """
    An evaluator that checks if the reference string is present (as a substring) in the answer.

    Attributes:
        name (str): Name of the evaluator. Defaults to "string_presence".
    """

    name: str = "string_presence"

    def run(self, ground_truth_answers: list[str], answers: list[str]) -> list[float]:
        """
        Compute string presence scores for each reference/answer pair.

        Args:
            ground_truth_answers (list[str]): List of reference strings.
            answers (list[str]): List of answer strings.

        Returns:
            list[float]: String presence scores
            (1.0 if reference is a substring of the answer, 0.0 otherwise) for each pair.
        """
        input_data = RunInput(ground_truth_answers=ground_truth_answers, answers=answers)
        scores: list[float] = []

        for idx, (ref, ans) in enumerate(zip(input_data.ground_truth_answers, input_data.answers)):
            presence = float(ref in ans)
            logger.debug(f"Processing pair {idx + 1}: String Presence = {presence}")
            scores.append(presence)

        output_data = RunOutput(scores=scores)
        return output_data.scores


class StringSimilarityEvaluator(BaseStringEvaluator):
    """
    An evaluator that calculates a similarity score (1 - normalized_distance) using RapidFuzz
    between reference and answer strings.

    Attributes:
        name (str): Name of the evaluator. Defaults to "non_llm_string_similarity".
        distance_measure (DistanceMeasure): Which distance measure to use. Defaults to DistanceMeasure.LEVENSHTEIN.
    """

    name: str = "non_llm_string_similarity"
    distance_measure: DistanceMeasure = DistanceMeasure.LEVENSHTEIN

    # Private attribute to store the distance functions map
    _distance_map: dict[DistanceMeasure, Callable] = PrivateAttr()

    # Configuration using ConfigDict for Pydantic v2 compliance
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @computed_field
    @cached_property
    def type(self) -> str:
        """
        Compute the type identifier for the evaluator.

        Returns:
            str: A string representing the module and class name.
        """
        return f"{self.__module__.rsplit('.', 1)[0]}.{self.__class__.__name__}"

    def __init__(self, **data):
        """
        Initialize the StringSimilarityEvaluator instance and set up the distance functions map.

        Args:
            **data: Arbitrary keyword arguments for the BaseModel.
        """
        super().__init__(**data)
        self._initialize_distance_map()

    def _initialize_distance_map(self) -> None:
        """
        Set up a map from the DistanceMeasure enum to the actual RapidFuzz distance classes.

        Raises:
            ImportError: If rapidfuzz is not installed.
        """
        try:
            from rapidfuzz import distance as rapidfuzz_distance
        except ImportError:
            raise ImportError(
                "rapidfuzz is required for StringSimilarityEvaluator. "
                "Please install it using `pip install rapidfuzz`."
            )
        self._distance_map = {
            DistanceMeasure.LEVENSHTEIN: rapidfuzz_distance.Levenshtein,
            DistanceMeasure.HAMMING: rapidfuzz_distance.Hamming,
            DistanceMeasure.JARO: rapidfuzz_distance.Jaro,
            DistanceMeasure.JARO_WINKLER: rapidfuzz_distance.JaroWinkler,
        }
        logger.debug(f"Initialized distance map: {self._distance_map}")

    def run(self, ground_truth_answers: list[str], answers: list[str]) -> list[float]:
        """
        Compute similarity scores for each reference/answer pair using the specified distance measure.

        Args:
            ground_truth_answers (list[str]): List of reference strings.
            answers (list[str]): List of answer strings.

        Returns:
            list[float]: Similarity scores for each pair.
        """
        input_data = RunInput(ground_truth_answers=ground_truth_answers, answers=answers)
        scores: list[float] = []
        distance_function = self._distance_map[self.distance_measure]

        for idx, (ref, ans) in enumerate(zip(input_data.ground_truth_answers, input_data.answers)):
            normalized_dist = distance_function.normalized_distance(ref, ans)
            similarity = 1 - float(normalized_dist)
            similarity_rounded = round(similarity, 2)
            logger.debug(
                f"Processing pair {idx + 1}: "
                f"Distance Measure = {self.distance_measure}, "
                f"Normalized Distance = {normalized_dist}, "
                f"Similarity = {similarity_rounded}"
            )
            scores.append(similarity_rounded)

        output_data = RunOutput(scores=scores)
        return output_data.scores
