from typing import Callable

from pydantic import BaseModel, PrivateAttr, model_validator

from dynamiq.evaluations import BaseEvaluator


class RunInput(BaseModel):
    """
    Input model for the BLEU score evaluation.

    Attributes:
        ground_truth_answers (list[str]): List of reference strings, one per example.
        answers (list[str]): List of response strings, one per example.
    """

    ground_truth_answers: list[str]
    answers: list[str]

    @model_validator(mode="after")
    def check_equal_length(self) -> "RunInput":
        """
        Validate that the number of ground truth answers matches the number of responses.

        Raises:
            ValueError: If the lengths of `ground_truth_answers` and `answers` do not match.

        Returns:
            RunInput: The validated instance.
        """
        if len(self.ground_truth_answers) != len(self.answers):
            raise ValueError("Answers and ground truth answers must have the same length.")
        return self


class RunOutput(BaseModel):
    """
    Output model for the BLEU score evaluation.

    Attributes:
        scores (list[float]): List of BLEU scores, one per reference/response pair.
    """

    scores: list[float]


class BleuScoreEvaluator(BaseEvaluator):
    """
    Evaluates BLEU scores using the sacrebleu library.

    Attributes:
        name (str): Name of the metric. Defaults to "BleuScore".
    """

    name: str = "BleuScore"

    # Private attribute to store the sacrebleu corpus_bleu function
    _corpus_bleu: Callable = PrivateAttr()

    def __init__(self, **data):
        """
        Initialize the BleuScoreEvaluator instance and set up the sacrebleu corpus_bleu function.

        Args:
            **data: Arbitrary keyword arguments for the BaseModel.
        """
        super().__init__(**data)
        self._initialize_bleu()

    def _initialize_bleu(self) -> None:
        """
        Initialize the corpus_bleu function from sacrebleu.

        Raises:
            ImportError: If sacrebleu is not installed.
        """
        from sacrebleu import corpus_bleu

        self._corpus_bleu = corpus_bleu

    def run(self, ground_truth_answers: list[str], answers: list[str]) -> list[float]:
        """
        Compute BLEU scores for each reference-response pair.

        Each reference and response is split into sentences by '. ' to simulate a multi-sentence scenario.
        The references are formatted as required by sacrebleu, i.e., a list of lists for multiple references.

        Args:
            references (list[str]): List of reference strings.
            responses (list[str]): List of response strings.

        Returns:
            list[float]: BLEU scores, one per pair.
        """
        input_data = RunInput(ground_truth_answers=ground_truth_answers, answers=answers)
        scores: list[float] = []

        for ref, resp in zip(input_data.ground_truth_answers, input_data.answers):
            # Split each into sentences
            ref_sentences = ref.split(". ")
            resp_sentences = resp.split(". ")

            # sacrebleu expects multiple references in the format of list[list[str]]
            # Here we treat each sentence as a separate reference.
            # Example: [['This is sentence one.'], ['This is sentence two.']]
            structured_refs: list[list[str]] = [[sent] for sent in ref_sentences]

            # Hypothesis is the list of response sentences
            # Example: ['This is sentence one.', 'This is sentence two.']
            hypothesis: list[str] = resp_sentences

            # Compute BLEU score (scale down by 100, as sacrebleu returns a percentage)
            bleu_result = self._corpus_bleu(hypothesis, structured_refs).score / 100.0
            bleu_score = round(float(bleu_result), 2)
            scores.append(bleu_score)

        output_data = RunOutput(scores=scores)
        return output_data.scores
