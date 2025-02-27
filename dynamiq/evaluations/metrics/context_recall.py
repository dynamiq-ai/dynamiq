import json
from typing import Any

from pydantic import BaseModel, PrivateAttr, field_validator, model_validator
from dynamiq.evaluations import BaseEvaluator
from dynamiq.evaluations.llm_evaluator import LLMEvaluator
from dynamiq.nodes.llms import BaseLLM
from dynamiq.utils.logger import logger


class ContextRecallInput(BaseModel):
    """
    Input model for context recall evaluation.

    Attributes:
        questions (list[str]): List of questions.
        contexts (list[str]): List of corresponding contexts (can also accept list[list[str]]).
        answers (list[str]): List of answers.
        verbose (bool): Flag to enable verbose logging.
    """
    questions: list[str]
    contexts: list[str] | list[list[str]]
    answers: list[str]
    verbose: bool = False

    @field_validator("contexts", mode="before")
    def unify_contexts(cls, value):
        """
        If we receive a list of lists of strings, join each sublist into a single string.
        Otherwise, if it's already list[str], do nothing.
        """
        if not isinstance(value, list):
            raise ValueError("contexts must be either a list[str] or a list[list[str]].")
        if all(isinstance(item, list) and all(isinstance(x, str) for x in item) for item in value):
            return [" ".join(sublist) for sublist in value]
        if all(isinstance(item, str) for item in value):
            return value
        raise ValueError("contexts must be either a list[str] or a list[list[str]].")

    @model_validator(mode="after")
    def check_equal_length(self):
        if not (len(self.questions) == len(self.contexts) == len(self.answers)):
            raise ValueError("Questions, contexts, and answers must have the same length.")
        return self


class ClassificationItem(BaseModel):
    """
    Model for individual classification result.

    Attributes:
        statement (str): The statement being classified.
        reason (str): Reason for the classification.
        attributed (int): 1 if attributed to context, 0 otherwise.
    """
    statement: str
    reason: str
    attributed: int

    @field_validator("attributed")
    @classmethod
    def validate_attributed(cls, value):
        if value not in (0, 1):
            raise ValueError("Attributed must be either 0 or 1.")
        return value


class ContextRecallRunResult(BaseModel):
    """
    Result model for the context recall evaluation.

    Attributes:
        score (float): The computed context recall score.
        reasoning (str): Detailed reasoning explaining how the score was derived.
    """
    score: float
    reasoning: str


class ContextRecallOutput(BaseModel):
    """
    Output model for context recall evaluation.

    Attributes:
        results (list[ContextRecallRunResult]): Detailed run results.
    """
    results: list[ContextRecallRunResult]


class ContextRecallEvaluator(BaseEvaluator):
    """
    Evaluator class for context recall metric.

    Attributes:
        llm (BaseLLM): The language model to use for evaluation.
    """
    name: str = "ContextRecall"
    llm: BaseLLM

    _classification_evaluator: LLMEvaluator = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self._initialize_evaluator()

    def _initialize_evaluator(self):
        class_instructions = (
            'Given a "Question", "Context", and "Answer", analyze each sentence in the '
            "Answer and classify if the sentence can be attributed to the given Context "
            "or not.\n"
            "- Use '1' (Yes) or '0' (No) for classification.\n"
            '- Provide a brief "reason" for each classification.\n'
            '- Output as a JSON object with key "classifications", where the value is a list '
            'of dictionaries with keys "statement", "reason", and "attributed".\n'
            "- Ensure your response is valid JSON, using double quotes for all strings."
        )

        self._classification_evaluator = LLMEvaluator(
            instructions=class_instructions.strip(),
            inputs=[
                {"name": "question", "type": list[str]},
                {"name": "context", "type": list[str]},
                {"name": "answer", "type": list[str]},
            ],
            outputs=[
                {"name": "classifications", "type": list[dict[str, Any]]},
            ],
            examples=[
                {
                    "inputs": {
                        "question": ["What can you tell me about Albert Einstein?"],
                        "context": [
                            (
                                "Albert Einstein (14 March 1879 - 18 April 1955) was a German-born "
                                "theoretical physicist, widely held to be one of the greatest and most "
                                "influential scientists of all time. Best known for developing the theory "
                                "of relativity, he also made important contributions to quantum mechanics, "
                                "and was thus a pivotal figure in modern physics."
                            )
                        ],
                        "answer": [
                            (
                                "Albert Einstein, born on 14 March 1879, was a German-born theoretical "
                                "physicist, widely held to be one of the greatest and most influential "
                                "scientists of all time. He received the 1921 Nobel Prize in Physics for his "
                                "services to theoretical physics. He published 4 papers in 1905. Einstein "
                                "moved to Switzerland in 1895."
                            )
                        ],
                    },
                    "outputs": {
                        "classifications": [
                            {
                                "statement": (
                                    "Albert Einstein, born on 14 March 1879, was a German-born theoretical "
                                    "physicist, widely held to be one of the greatest and most influential "
                                    "scientists of all time."
                                ),
                                "reason": "The birth date and status as a theoretical physicist are mentioned.",
                                "attributed": 1,
                            },
                            {
                                "statement": (
                                    "He received the 1921 Nobel Prize in Physics for his services to theoretical "
                                    "physics."
                                ),
                                "reason": "The sentence is present in the context.",
                                "attributed": 1,
                            },
                            {
                                "statement": "He published 4 papers in 1905.",
                                "reason": "There is no mention of his papers in the context.",
                                "attributed": 0,
                            },
                            {
                                "statement": "Einstein moved to Switzerland in 1895.",
                                "reason": "There is no supporting evidence for this in the context.",
                                "attributed": 0,
                            },
                        ]
                    },
                },
            ],
            llm=self.llm,
        )

    def _build_reasoning(self, classifications: list[ClassificationItem], score: float) -> str:
        """
        Build a detailed reasoning string for context recall evaluation.

        Explains:
        • Each sentence in the answer is classified (using emojis: ✅ for attributed, ❌ for not).
        • A corresponding explanation is provided for each classification.
        • The final context recall score is computed as the ratio of attributable sentences.

        Args:
            classifications (list[ClassificationItem]): List of classification results.
            score (float): The computed recall score.

        Returns:
            str: Detailed reasoning.
        """
        lines = []
        lines.extend(["Reasoning:", "", "Classifications:"])
        for item in classifications:
            mark = "✅" if item.attributed == 1 else "❌"
            lines.extend(
                [
                    f" - Statement: {item.statement}",
                    f"   Verdict: {mark} (value: {item.attributed})",
                    f"   Explanation: {item.reason}",
                    "",
                ]
            )
        lines.append(f"Context Recall Score = {score:.2f}")
        return "\n".join(lines)

    def run_single(self, question: str, context: str, answer: str, verbose: bool = False) -> ContextRecallRunResult:
        """
        Evaluate the context recall for a single sample.

        Args:
            question (str): The question.
            context (str): The context (already normalized as a single string).
            answer (str): The answer.
            verbose (bool): Flag to enable verbose logging.

        Returns:
            ContextRecallRunResult: The computed context recall score and detailed reasoning.
        """
        result = self._classification_evaluator.run(
            question=[question],
            context=[context],
            answer=[answer],
        )

        classifications = []
        if "results" not in result or not result["results"]:
            if verbose:
                logger.debug(f"No results returned for question: {question}, context: {context}.")
        else:
            first_result = result["results"][0]
            if "classifications" not in first_result or not first_result["classifications"]:
                if verbose:
                    logger.debug(f"No classifications returned for question: {question}, context: {context}.")
            else:
                classifications_raw = first_result["classifications"]
                for item in classifications_raw:
                    classification_item = ClassificationItem(
                        statement=item["statement"],
                        reason=item["reason"],
                        attributed=int(item["attributed"]),
                    )
                    classifications.append(classification_item)

        attributed_list = [item.attributed for item in classifications]
        num_sentences = len(attributed_list)
        num_attributed = sum(attributed_list)
        score = num_attributed / num_sentences if num_sentences > 0 else 0.0
        score = round(float(score), 2)

        reasoning_str = self._build_reasoning(classifications, score)

        if verbose:
            logger.debug(f"Question: {question}")
            logger.debug(f"Answer: {answer}")
            logger.debug(f"Context: {context}")
            logger.debug("Classifications:")
            logger.debug(json.dumps([item.dict() for item in classifications], indent=2))
            logger.debug(f"Context Recall Score: {score}")
            logger.debug("-" * 50)

        return ContextRecallRunResult(score=score, reasoning=reasoning_str)

    def run(
        self,
        questions: list[str],
        contexts: list[str] | list[list[str]],
        answers: list[str],
        verbose: bool = False,
    ) -> ContextRecallOutput:
        """
        Evaluate the context recall for each question.

        Args:
            questions (list[str]): List of questions.
            contexts (list[str] or list[list[str]]): Either a single list of context strings or a list
                of context strings (one per question).
            answers (list[str]): List of answers.
            verbose (bool): Flag to enable verbose logging (for internal logging only).

        Returns:
            ContextRecallOutput: Contains a list of context recall run results.
        """
        run_input = ContextRecallInput(
            questions=questions,
            contexts=contexts,
            answers=answers,
            verbose=verbose,
        )
        results_output = []
        for i in range(len(run_input.questions)):
            question = run_input.questions[i]
            context = run_input.contexts[i]
            answer = run_input.answers[i]
            result_single = self.run_single(
                question=question, context=context, answer=answer, verbose=run_input.verbose
            )
            results_output.append(result_single)
        return ContextRecallOutput(results=results_output)
