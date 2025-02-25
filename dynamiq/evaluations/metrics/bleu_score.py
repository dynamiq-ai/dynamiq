import re
from typing import Callable

from pydantic import BaseModel, PrivateAttr, model_validator

from dynamiq.evaluations import BaseEvaluator


class RunInput(BaseModel):
    """
    Input model for batch BLEU score evaluation.

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
            ValueError: If the lengths differ.

        Returns:
            RunInput: The validated instance.
        """
        if len(self.ground_truth_answers) != len(self.answers):
            raise ValueError("The number of ground truth answers must equal the number of answers.")
        return self


class RunOutput(BaseModel):
    """
    Output model for batch BLEU score evaluation.

    Attributes:
        scores (list[float]): List of computed BLEU scores.
    """

    scores: list[float]


class RunSingleInput(BaseModel):
    """
    Single-run input model for BLEU score evaluation.

    Attributes:
        ground_truth_answer (str): The reference answer.
        answer (str): The candidate answer.
    """

    ground_truth_answer: str
    answer: str


class RunSingleOutput(BaseModel):
    """
    Single-run output model for BLEU score evaluation.

    Attributes:
        score (float): The computed BLEU score.
    """

    score: float


class BleuScoreEvaluator(BaseEvaluator):
    """
    Evaluates BLEU scores using the sacrebleu library.

    Attributes:
        name (str): Name of the metric. Defaults to "BleuScore".
    """
    name: str = "BleuScore"

    # Private attribute to store the sacrebleu corpus_bleu function.
    _corpus_bleu: Callable = PrivateAttr()

    def __init__(self, **data):
        """
        Initialize the BleuScoreEvaluator and load the sacrebleu corpus_bleu function.
        """
        super().__init__(**data)
        self._initialize_bleu()

    def _initialize_bleu(self) -> None:
        """
        Initialize the corpus_bleu function from the sacrebleu library.

        Raises:
            ImportError: If sacrebleu is not installed.
        """
        from sacrebleu import corpus_bleu

        self._corpus_bleu = corpus_bleu

    def run_single(self, ground_truth_answer: str, answer: str) -> float:
        """
        Compute the BLEU score for a single pair of ground truth (reference) and answer.

        The input strings are into sentences. The reference is provided
        in a format expected by sacrebleu (a list of lists) and the candidate is provided
        as a list of sentences.

        Args:
            ground_truth_answer (str): The reference answer.
            answer (str): The candidate answer.

        Returns:
            float: The computed BLEU score (as a fraction, e.g., 0.75 for 75%).
        """
        # Validate inputs using the Pydantic model.
        single_input = RunSingleInput(ground_truth_answer=ground_truth_answer, answer=answer)

        # Process text into clean sentences
        ref_sentences = self._process_text_for_bleu(single_input.ground_truth_answer)
        resp_sentences = self._process_text_for_bleu(single_input.answer)

        # Format the reference as a list of lists (one per sentence)
        structured_refs = [[sent] for sent in ref_sentences]
        hypothesis = resp_sentences

        # Compute the BLEU score; sacrebleu returns a percentage, so we scale it by 1/100.
        bleu_result = self._corpus_bleu(hypothesis, structured_refs).score / 100.0
        score = round(float(bleu_result), 2)

        output = RunSingleOutput(score=score)
        return output.score

    def _process_text_for_bleu(self, text: str) -> list[str]:
        """
        Process text into clean sentences for BLEU score computation.

        Args:
            text (str): The text to process.

        Returns:
            list[str]: List of cleaned sentences.
        """
        # First split the text into sentences
        raw_sentences = self._split_text_into_sentences(text)

        # Then clean each sentence by removing punctuation
        cleaned_sentences = [self._clean_sentence(sentence) for sentence in raw_sentences]

        # Filter out empty or very short sentences (likely fragments)
        return [s for s in cleaned_sentences if s and len(s.split()) > 1]

    def _split_text_into_sentences(self, text: str) -> list[str]:
        """
        Split text into sentences based on punctuation boundaries.

        Args:
            text (str): The text to split.

        Returns:
            list[str]: List of sentences.
        """
        # Split on ., !, or ? followed by whitespace or end of string
        sentences = re.split(r"(?<=[.!?])\s+|(?<=[.!?])$", text)
        return [s.strip() for s in sentences if s.strip()]

    def _clean_sentence(self, sentence: str) -> str:
        """
        Clean a sentence by removing all punctuation.

        Args:
            sentence (str): The sentence to clean.

        Returns:
            str: Cleaned sentence with punctuation removed.
        """
        # Remove leading/trailing whitespace
        cleaned = sentence.strip()

        # Remove all punctuation
        cleaned = re.sub(r"[^\w\s]", "", cleaned)

        return cleaned.strip()

    def run(self, ground_truth_answers: list[str], answers: list[str]) -> list[float]:
        """
        Compute BLEU scores for each ground_truth_answer/answer pair in batch.

        Args:
            ground_truth_answers (list[str]): List of reference answers.
            answers (list[str]): List of candidate answers.

        Returns:
            list[float]: List of computed BLEU scores.
        """
        # Validate batch input.
        input_data = RunInput(ground_truth_answers=ground_truth_answers, answers=answers)
        scores = []

        for gt, ans in zip(input_data.ground_truth_answers, input_data.answers):
            score = self.run_single(ground_truth_answer=gt, answer=ans)
            scores.append(score)

        output_data = RunOutput(scores=scores)
        return output_data.scores
