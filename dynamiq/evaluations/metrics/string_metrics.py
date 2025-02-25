from enum import Enum
from typing import Callable

from pydantic import BaseModel, PrivateAttr, model_validator
from dynamiq.evaluations import BaseEvaluator
from dynamiq.utils.logger import logger


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
    Input model for batch string evaluations.

    Attributes:
        ground_truth_answers (list[str]): List of reference strings, one per example.
        answers (list[str]): List of candidate response strings, one per example.
    """
    ground_truth_answers: list[str]
    answers: list[str]

    @model_validator(mode="after")
    def check_equal_length(self) -> "RunInput":
        if len(self.ground_truth_answers) != len(self.answers):
            raise ValueError("ground_truth_answers and answers must have the same length.")
        return self


class RunOutput(BaseModel):
    """
    Output model for string evaluations.

    Attributes:
        scores (list[float]): List of evaluator scores, one per reference/answer pair.
    """

    scores: list[float]


class RunSingleInput(BaseModel):
    """
    Single-run input model for string evaluations.

    Attributes:
        ground_truth_answer (str): The reference string.
        answer (str): The candidate string.
    """

    ground_truth_answer: str
    answer: str


class RunSingleOutput(BaseModel):
    """
    Single-run output model for string evaluations.

    Attributes:
        score (float): The computed evaluator score.
    """

    score: float


class ExactMatchEvaluator(BaseEvaluator):
    """
    An evaluator that checks for exact string matches.

    Attributes:
        name (str): Name of the evaluator. Defaults to "exact_match".
    """
    name: str = "exact_match"

    def run_single(self, ground_truth_answer: str, answer: str) -> float:
        input_data = RunSingleInput(ground_truth_answer=ground_truth_answer, answer=answer)
        exact_match = float(input_data.ground_truth_answer == input_data.answer)
        logger.debug(f"Single pair: Exact Match = {exact_match}")
        return exact_match

    def run(self, ground_truth_answers: list[str], answers: list[str]) -> list[float]:
        input_data = RunInput(ground_truth_answers=ground_truth_answers, answers=answers)
        scores = []
        for gt, ans in zip(input_data.ground_truth_answers, input_data.answers):
            score = self.run_single(ground_truth_answer=gt, answer=ans)
            scores.append(score)
        output_data = RunOutput(scores=scores)
        return output_data.scores


class StringPresenceEvaluator(BaseEvaluator):
    """
    An evaluator that checks if the reference string is present (as a substring) in the answer.

    Attributes:
        name (str): Name of the evaluator. Defaults to "string_presence".
    """
    name: str = "string_presence"

    def run_single(self, ground_truth_answer: str, answer: str) -> float:
        input_data = RunSingleInput(ground_truth_answer=ground_truth_answer, answer=answer)
        presence = float(input_data.ground_truth_answer in input_data.answer)
        output = RunSingleOutput(score=presence)
        return output.score

    def run(self, ground_truth_answers: list[str], answers: list[str]) -> list[float]:
        input_data = RunInput(ground_truth_answers=ground_truth_answers, answers=answers)
        scores = []
        for gt, ans in zip(input_data.ground_truth_answers, input_data.answers):
            score = self.run_single(ground_truth_answer=gt, answer=ans)
            scores.append(score)
        output_data = RunOutput(scores=scores)
        return output_data.scores


class StringSimilarityEvaluator(BaseEvaluator):
    """
    An evaluator that calculates a similarity score (1 - normalized_distance) using RapidFuzz
    between reference and answer strings.

    Attributes:
        name (str): Name of the evaluator. Defaults to "non_llm_string_similarity".
        distance_measure (DistanceMeasure): Which distance measure to use. Defaults to DistanceMeasure.LEVENSHTEIN.
    """
    name: str = "non_llm_string_similarity"
    distance_measure: DistanceMeasure = DistanceMeasure.LEVENSHTEIN

    _distance_map: dict[DistanceMeasure, Callable] = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self._initialize_distance_map()

    def _initialize_distance_map(self) -> None:
        from rapidfuzz import distance as rapidfuzz_distance
        self._distance_map = {
            DistanceMeasure.LEVENSHTEIN: rapidfuzz_distance.Levenshtein,
            DistanceMeasure.HAMMING: rapidfuzz_distance.Hamming,
            DistanceMeasure.JARO: rapidfuzz_distance.Jaro,
            DistanceMeasure.JARO_WINKLER: rapidfuzz_distance.JaroWinkler,
        }
        logger.debug(f"Initialized distance map: {self._distance_map}")

    def run_single(self, ground_truth_answer: str, answer: str) -> float:
        input_data = RunSingleInput(ground_truth_answer=ground_truth_answer, answer=answer)
        distance_function = self._distance_map[self.distance_measure]
        normalized_dist = distance_function.normalized_distance(input_data.ground_truth_answer, input_data.answer)
        similarity = 1 - float(normalized_dist)
        similarity_rounded = round(similarity, 2)
        logger.debug(
            f"Single pair: Distance Measure = {self.distance_measure}, "
            f"Normalized Distance = {normalized_dist}, "
            f"Similarity = {similarity_rounded}"
        )
        output = RunSingleOutput(score=similarity_rounded)
        return output.score

    def run(self, ground_truth_answers: list[str], answers: list[str]) -> list[float]:
        input_data = RunInput(ground_truth_answers=ground_truth_answers, answers=answers)
        scores = []
        for gt, ans in zip(input_data.ground_truth_answers, input_data.answers):
            score = self.run_single(ground_truth_answer=gt, answer=ans)
            scores.append(score)
        output_data = RunOutput(scores=scores)
        return output_data.scores
