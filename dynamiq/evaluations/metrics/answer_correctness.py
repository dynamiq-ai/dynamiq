import json
import logging
from functools import cached_property

from pydantic import BaseModel, Field, PrivateAttr, computed_field, field_validator, model_validator

from dynamiq.components.evaluators.llm_evaluator import LLMEvaluator
from dynamiq.nodes.llms import BaseLLM

# Configure logging
logger = logging.getLogger(__name__)


class ExtractStatementsInput(BaseModel):
    """
    Input model for extracting statements.

    Attributes:
        texts (List[str]): The texts from which to extract key statements.
    """

    texts: list[str]


class ExtractStatementsOutput(BaseModel):
    """
    Output model for extracted statements.

    Attributes:
        statements_list (List[List[str]]): A list of lists containing the
        extracted statements.
    """

    statements_list: list[list[str]]


class ClassifyStatementsInput(BaseModel):
    """
    Input model for classifying statements.

    Attributes:
        questions (List[str]): The list of questions.
        answer_statements_list (List[List[str]]): The list of answer statements.
        ground_truth_statements_list (List[List[str]]): The list of ground
        truth statements.
    """

    questions: list[str]
    answer_statements_list: list[list[str]]
    ground_truth_statements_list: list[list[str]]


class ClassificationResult(BaseModel):
    """
    Model for classification results.

    Attributes:
        TP (List[str]): True Positives.
        FP (List[str]): False Positives.
        FN (List[str]): False Negatives.
    """

    TP: list[str] = Field(default_factory=list)
    FP: list[str] = Field(default_factory=list)
    FN: list[str] = Field(default_factory=list)

    @field_validator("TP", "FP", "FN", mode="before")
    @classmethod
    def set_default_list(cls, v):
        return v or []


class ClassifyStatementsOutput(BaseModel):
    """
    Output model for classification of statements.

    Attributes:
        classifications_list (List[ClassificationResult]): The list of
        classification results.
    """

    classifications_list: list[ClassificationResult]


class ComputeSimilarityScoresInput(BaseModel):
    """
    Input model for computing similarity scores.

    Attributes:
        answers (List[str]): The list of answers.
        ground_truths (List[str]): The list of ground truth texts.
    """

    answers: list[str]
    ground_truths: list[str]


class ComputeSimilarityScoresOutput(BaseModel):
    """
    Output model for similarity scores.

    Attributes:
        similarity_scores (List[float]): The list of similarity scores.
    """

    similarity_scores: list[float]


class RunInput(BaseModel):
    """
    Input model for running the evaluator.

    Attributes:
        questions (List[str]): The list of questions.
        answers (List[str]): The list of answers.
        ground_truths (List[str]): The list of ground truth texts.
        verbose (bool): Flag to enable verbose logging.
    """

    questions: list[str]
    answers: list[str]
    ground_truths: list[str]
    verbose: bool = False

    @model_validator(mode="after")
    def check_equal_length(self):
        if len(self.questions) != len(self.answers) or len(self.questions) != len(self.ground_truths):
            raise ValueError("Questions, answers, and ground truths must have the same length.")
        return self


class RunOutput(BaseModel):
    """
    Output model for the final scores.

    Attributes:
        final_scores (List[float]): The list of final correctness scores.
    """

    final_scores: list[float]


class AnswerCorrectnessEvaluator(BaseModel):
    """
    Evaluator class for computing the correctness of answers given
    questions and ground truths.

    Attributes:
        llm (BaseLLM): The language model to use for evaluation.
        weights (List[float]): The weights to combine F1 and similarity scores.
    """

    llm: BaseLLM
    weights: list[float] = Field(default_factory=lambda: [0.75, 0.25])

    # Private attributes (not part of the Pydantic model fields)
    _statement_extractor: LLMEvaluator = PrivateAttr()
    _statement_classifier: LLMEvaluator = PrivateAttr()
    _similarity_evaluator: LLMEvaluator = PrivateAttr()

    class Config:
        arbitrary_types_allowed = True  # Allow arbitrary types like LLMEvaluator and BaseLLM

    @computed_field
    @cached_property
    def type(self) -> str:
        return f"{self.__module__.rsplit('.', 1)[0]}.{self.__class__.__name__}"

    def __init__(self, **data):
        super().__init__(**data)
        self._initialize_evaluators()

    def _initialize_evaluators(self):
        # Initialize the LLMEvaluators and store them as class attributes

        # Extract Statements Evaluator
        extract_instructions = (
            "For each input text, extract the key statements.\n"
            "- Keep statements concise and focused.\n"
            "- Return the statements as a JSON array of strings.\n"
            "- Ensure that your response is valid JSON, using double quotes "
            "for all strings."
        )
        self._statement_extractor = LLMEvaluator(
            instructions=extract_instructions.strip(),
            inputs=[{"name": "texts", "type": list[str]}],
            outputs=[{"name": "statements", "type": list[str]}],
            examples=[
                {
                    "inputs": {"texts": ["The sun is powered by nuclear fusion. It provides heat and light."]},
                    "outputs": {
                        "statements": [
                            "The sun is powered by nuclear fusion.",
                            "It provides heat and light.",
                        ]
                    },
                },
            ],
            llm=self.llm,
        )

        # Classify Statements Evaluator
        classify_instructions = (
            'Given a "Question", "Answer Statements", and "Ground Truth '
            'Statements", classify the statements:\n'
            '- "TP" (True Positive): Statements present in both Answer and '
            "Ground Truth.\n"
            '- "FP" (False Positive): Statements in Answer but not in Ground '
            "Truth.\n"
            '- "FN" (False Negative): Statements in Ground Truth but not in '
            "Answer.\n"
            "Each statement should only belong to one category.\n"
            'Provide the classifications as a JSON object with keys "TP", '
            '"FP", "FN", and values as lists of statements.\n'
            "Ensure that your response is valid JSON, using double quotes "
            "for all strings."
        )
        self._statement_classifier = LLMEvaluator(
            instructions=classify_instructions.strip(),
            inputs=[
                {"name": "question", "type": list[str]},
                {"name": "answer_statements", "type": list[list[str]]},
                {"name": "ground_truth_statements", "type": list[list[str]]},
            ],
            outputs=[{"name": "classifications", "type": dict[str, list[str]]}],
            examples=[
                {
                    "inputs": {
                        "question": ["What powers the sun and what is its primary function?"],
                        "answer_statements": [
                            "The sun is powered by nuclear fission.",
                            "The sun's primary function is to provide light to the solar system.",
                        ],
                        "ground_truth_statements": [
                            "The sun is powered by nuclear fusion.",
                            "The sun provides heat and light essential for life on Earth.",
                        ],
                    },
                    "outputs": {
                        "classifications": {
                            "TP": ["The sun's primary function is to provide light to the solar system."],
                            "FP": ["The sun is powered by nuclear fission."],
                            "FN": [
                                "The sun is powered by nuclear fusion.",
                                "The sun provides heat and light essential for life on Earth.",
                            ],
                        }
                    },
                },
            ],
            llm=self.llm,
        )

        # Compute Similarity Evaluator
        similarity_instructions = (
            'For each pair of "Answer" and "Ground Truth", evaluate their '
            "semantic similarity.\n"
            "- Score the similarity from 0 to 1.\n"
            "- Use 1 if the Answer is semantically identical to the Ground "
            "Truth.\n"
            "- Use 0 if the Answer is completely dissimilar to the Ground "
            "Truth.\n"
            "- Provide the similarity score as a single number between 0 "
            "and 1.\n"
            "Ensure that your response is valid JSON, using double quotes "
            "for all strings."
        )
        self._similarity_evaluator = LLMEvaluator(
            instructions=similarity_instructions.strip(),
            inputs=[{"name": "answers", "type": list[str]}, {"name": "ground_truths", "type": list[str]}],
            outputs=[{"name": "similarity_score", "type": float}],
            examples=[
                {
                    "inputs": {
                        "answers": ["Paris is the capital of France."],
                        "ground_truths": ["The capital of France is Paris."],
                    },
                    "outputs": {"similarity_score": 1},
                },
                {
                    "inputs": {
                        "answers": ["Berlin is the capital of Germany."],
                        "ground_truths": ["The capital of France is Paris."],
                    },
                    "outputs": {"similarity_score": 0},
                },
            ],
            llm=self.llm,
        )

    def extract_statements(self, texts: list[str]) -> list[list[str]]:
        input_data = ExtractStatementsInput(texts=texts)
        results = self._statement_extractor.run(texts=input_data.texts)
        # Extract the 'statements' from the results and ensure proper structure
        statements_list = []
        for result in results["results"]:
            statements = result.get("statements")
            # Ensure 'statements' is a list of strings
            if isinstance(statements, list):
                statements_list.append(statements)
            else:
                # If not, wrap it in a list
                statements_list.append([statements])
        output_data = ExtractStatementsOutput(statements_list=statements_list)
        return output_data.statements_list

    def classify_statements(
        self,
        questions: list[str],
        answer_statements_list: list[list[str]],
        ground_truth_statements_list: list[list[str]],
    ) -> list[ClassificationResult]:
        input_data = ClassifyStatementsInput(
            questions=questions,
            answer_statements_list=answer_statements_list,
            ground_truth_statements_list=ground_truth_statements_list,
        )
        results = self._statement_classifier.run(
            question=input_data.questions,
            answer_statements=input_data.answer_statements_list,
            ground_truth_statements=input_data.ground_truth_statements_list,
        )
        classifications_list = []
        for result in results["results"]:
            classification = ClassificationResult(**result["classifications"])
            classifications_list.append(classification)
        output_data = ClassifyStatementsOutput(classifications_list=classifications_list)
        return output_data.classifications_list

    def compute_similarity_scores(self, answers: list[str], ground_truths: list[str]) -> list[float]:
        input_data = ComputeSimilarityScoresInput(answers=answers, ground_truths=ground_truths)
        results = self._similarity_evaluator.run(
            answers=input_data.answers,
            ground_truths=input_data.ground_truths,
        )
        similarity_scores = [float(result["similarity_score"]) for result in results["results"]]
        output_data = ComputeSimilarityScoresOutput(similarity_scores=similarity_scores)
        return output_data.similarity_scores

    @staticmethod
    def compute_f1_score(classifications: ClassificationResult) -> float:
        tp = len(classifications.TP)
        fp = len(classifications.FP)
        fn = len(classifications.FN)
        if tp == 0 and (fp > 0 or fn > 0):
            return 0.0
        if tp == 0 and fp == 0 and fn == 0:
            return 1.0  # No statements to compare.
        precision_denom = tp + fp
        recall_denom = tp + fn
        precision = tp / precision_denom if precision_denom > 0 else 1.0
        recall = tp / recall_denom if recall_denom > 0 else 1.0
        if (precision + recall) == 0.0:
            return 0.0
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1

    def run(
        self,
        questions: list[str],
        answers: list[str],
        ground_truths: list[str],
        verbose: bool = False,
    ) -> list[float]:
        input_data = RunInput(
            questions=questions,
            answers=answers,
            ground_truths=ground_truths,
            verbose=verbose,
        )

        # Extract statements
        answer_statements_list = self.extract_statements(input_data.answers)
        ground_truth_statements_list = self.extract_statements(input_data.ground_truths)

        # Classify statements
        classifications_list = self.classify_statements(
            input_data.questions,
            answer_statements_list,
            ground_truth_statements_list,
        )

        # Compute F1 scores
        f1_scores = [self.compute_f1_score(classifications) for classifications in classifications_list]

        # Compute similarity scores
        similarity_scores = self.compute_similarity_scores(input_data.answers, input_data.ground_truths)

        # Combine scores
        final_scores = []
        for i in range(len(input_data.questions)):
            f1_score = f1_scores[i]
            sim_score = similarity_scores[i]
            final_score = self.weights[0] * f1_score + self.weights[1] * sim_score
            final_scores.append(final_score)

            if input_data.verbose:
                logger.debug(f"Question: {input_data.questions[i]}")
                logger.debug(f"Answer: {input_data.answers[i]}")
                logger.debug(f"Ground Truth: {input_data.ground_truths[i]}")
                logger.debug("Classifications:")
                logger.debug(json.dumps(classifications_list[i].dict(), indent=2))
                logger.debug(f"F1 Score: {f1_score}")
                logger.debug(f"Similarity Score: {sim_score}")
                logger.debug(f"Final Score: {final_score}")
                logger.debug("-" * 50)

        output_data = RunOutput(final_scores=final_scores)
        return output_data.final_scores
