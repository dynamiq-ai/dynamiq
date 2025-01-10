import logging
from functools import cached_property

from pydantic import BaseModel, PrivateAttr, model_validator

logger = logging.getLogger(__name__)

try:
    from sacrebleu import corpus_bleu
except ImportError:
    raise ImportError("sacrebleu is required for BleuScore. Please install it using `pip install sacrebleu`")


class SingleTurnMetricInput(BaseModel):
    """
    Input model for a single-turn BLEU metric.

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
    Output model for a single-turn BLEU metric.

    Attributes:
        scores (List[float]): List of BLEU scores, one per reference/response pair.
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
        arbitrary_types_allowed = True  # If you want to store custom objects

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


class BleuScoreMetric(BaseStringMetric):
    """
    Implements a BLEU score metric using sacrebleu.

    Attributes:
        name (str): Name of the metric. Defaults to "bleu_score".
        language (str): Language for BLEU scoring. Here for demonstration purposes.
    """

    name: str = "bleu_score"
    language: str = "english"  # Not used directly in sacrebleu, but can be stored

    # Private attribute to store the sacrebleu function
    _corpus_bleu = PrivateAttr(default=None)

    def __init__(self, **data):
        super().__init__(**data)
        self._initialize_bleu()

    def _initialize_bleu(self):
        """
        Initialize the corpus_bleu function from sacrebleu.
        """
        self._corpus_bleu = corpus_bleu

    def run(self, references: list[str], responses: list[str]) -> list[float]:
        """
        Compute BLEU scores for each reference/response pair.

        The reference and response are each split into sentences by '. ' to mimic
        a multi-sentence scenario. The original reference is then wrapped into the
        format needed by sacrebleu, i.e. List[List[str]] for multiple references.

        Args:
            references (List[str]): List of reference strings.
            responses (List[str]): List of response strings.

        Returns:
            List[float]: BLEU scores, one per pair.
        """
        input_data = SingleTurnMetricInput(references=references, responses=responses)
        scores = []

        for ref, resp in zip(input_data.references, input_data.responses):
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

        output_data = SingleTurnMetricOutput(scores=scores)
        return output_data.scores
