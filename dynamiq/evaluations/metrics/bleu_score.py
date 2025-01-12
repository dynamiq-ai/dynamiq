import logging
from functools import cached_property
from typing import Callable

from pydantic import BaseModel, PrivateAttr, computed_field, model_validator

logger = logging.getLogger(__name__)


class RunInput(BaseModel):
    """
    Input model for a single-turn BLEU metric.

    Attributes:
        references (List[str]): List of reference strings, one per example.
        responses (List[str]): List of response strings, one per example.
    """

    ground_truth_answers: list[str]
    answers: list[str]

    @model_validator(mode="after")
    def check_equal_length(self):
        if len(self.ground_truth_answers) != len(self.answers):
            raise ValueError("Answers and ground truth answers must have the same length.")
        return self


class RunOutput(BaseModel):
    """
    Output model for a single-turn BLEU metric.

    Attributes:
        scores (List[float]): List of BLEU scores, one per reference/response pair.
    """

    scores: list[float]


class BleuScoreMetric(BaseModel):
    """
    Implements a BLEU score metric using sacrebleu.

    Attributes:
        name (str): Name of the metric. Defaults to "bleu_score".
        language (str): Language for BLEU scoring. Here for demonstration purposes.
    """

    name: str = "bleu_score"
    language: str = "english"  # Not used directly in sacrebleu, but can be stored

    # Private attribute to store the sacrebleu function
    _corpus_bleu: Callable = PrivateAttr()

    class Config:
        arbitrary_types_allowed = True

    @computed_field
    @cached_property
    def type(self) -> str:
        return f"{self.__module__.rsplit('.', 1)[0]}.{self.__class__.__name__}"

    def __init__(self, **data):
        super().__init__(**data)
        self._initialize_bleu()

    def _initialize_bleu(self):
        """
        Initialize the corpus_bleu function from sacrebleu.
        """
        try:
            from sacrebleu import corpus_bleu
        except ImportError:
            raise ImportError("sacrebleu is required for BleuScore. Please install it using `pip install sacrebleu`")

        self._corpus_bleu = corpus_bleu

    def run(self, references: list[str], responses: list[str]) -> list[float]:
        """
        Compute BLEU scores for each ground_truth/answer pair.

        The reference and response are each split into sentences by '. ' to mimic
        a multi-sentence scenario. The original reference is then wrapped into the
        format needed by sacrebleu, i.e. List[List[str]] for multiple references.

        Args:
            ground_truth_answers (List[str]): List of reference strings.
            answers (List[str]): List of response strings.

        Returns:
            List[float]: BLEU scores, one per pair.
        """
        input_data = RunInput(ground_truth_answers=references, answers=responses)
        scores = []

        for ref, resp in zip(input_data.ground_truth_answers, input_data.answers):
            # Split each into sentences
            ref_sentences = ref.split(". ")
            resp_sentences = resp.split(". ")

            # sacrebleu expects multiple references in the format of List[List[str]]
            # Here we treat each sentence as a reference.
            # e.g. [['This is sentence one.'], ['This is sentence two.']]
            structured_refs = [[sent] for sent in ref_sentences]

            # For the hypothesis, we just pass the sentence list
            # e.g. ['This is sentence one.', 'This is sentence two.']
            hypothesis = resp_sentences

            # Compute BLEU score (scale down by 100, as sacrebleu returns a percentage)
            bleu_result = self._corpus_bleu(hypothesis, structured_refs).score / 100.0
            scores.append(float(bleu_result))

        output_data = RunOutput(scores=scores)
        return output_data.scores
