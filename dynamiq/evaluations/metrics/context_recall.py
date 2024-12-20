import json
import logging
from functools import cached_property
from typing import Any

from pydantic import BaseModel, PrivateAttr, computed_field, field_validator, model_validator

from dynamiq.components.evaluators.llm_evaluator import LLMEvaluator
from dynamiq.nodes.llms import BaseLLM

# Configure logging
logger = logging.getLogger(__name__)


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

    # mode="before" -> run this validator before creating the model
    @field_validator("contexts", mode="before")
    def unify_contexts(cls, v):
        """
        If we receive a list of lists of strings, join each sublist into a single string.
        Otherwise, if it's already list[str], do nothing.
        """
        # Ensure "contexts" is at least a list
        if not isinstance(v, list):
            raise ValueError("contexts must be either a list[str] or a list[list[str]].")

        # Check if list[list[str]] (all items are lists, each sublist all strings)
        if all(isinstance(item, list) and all(isinstance(x, str) for x in item) for item in v):
            # Convert each sublist into a single string by joining with a space
            return [" ".join(sublist) for sublist in v]

        # Check if already list[str]
        if all(isinstance(item, str) for item in v):
            return v

        # Otherwise, invalid data structure
        raise ValueError("contexts must be either a list[str] or a list[list[str]].")

    @model_validator(mode="after")
    def check_equal_length(self):
        """
        After the model is created (and 'contexts' is normalized),
        ensure that questions, contexts, and answers are the same length.
        """
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
    def validate_attributed(cls, v):
        if v not in (0, 1):
            raise ValueError("Attributed must be either 0 or 1.")
        return v


class ContextRecallOutput(BaseModel):
    """
    Output model for context recall evaluation.

    Attributes:
        final_scores (List[float]): List of context recall scores.
    """

    final_scores: list[float]


class ContextRecallEvaluator(BaseModel):
    """
    Evaluator class for context recall metric.

    Attributes:
        llm (BaseLLM): The language model to use for evaluation.
    """

    llm: BaseLLM

    # Private attribute
    _classification_evaluator: LLMEvaluator = PrivateAttr()

    class Config:
        arbitrary_types_allowed = True  # Allow arbitrary types

    @computed_field
    @cached_property
    def type(self) -> str:
        return f"{self.__module__.rsplit('.', 1)[0]}.{self.__class__.__name__}"

    def __init__(self, **data):
        super().__init__(**data)
        self._initialize_evaluator()

    def _initialize_evaluator(self):
        class_instructions = (
            'Given a "Question", "Context", and "Answer", analyze each sentence '
            "in the Answer and classify if the sentence can be attributed to the "
            "given Context or not.\n"
            "- Use '1' (Yes) or '0' (No) for classification.\n"
            '- Provide a brief "reason" for each classification.\n'
            '- Output as a JSON object with key "classifications", where the value '
            'is a list of dictionaries with keys "statement", "reason", and '
            '"attributed".\n'
            "- Ensure your response is valid JSON, using double quotes for "
            "all strings."
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
                                "Albert Einstein (14 March 1879 - 18 April 1955) "
                                "was a German-born theoretical physicist, widely "
                                "held to be one of the greatest and most influential "
                                "scientists of all time. Best known for developing "
                                "the theory of relativity, he also made important "
                                "contributions to quantum mechanics, and was thus a "
                                "central figure in the revolutionary reshaping of the "
                                "scientific understanding of nature that modern "
                                "physics accomplished in the first decades of the "
                                "twentieth century. His mass-energy equivalence "
                                "formula E = mc^2, which arises from relativity "
                                "theory, has been called 'the world's most famous "
                                "equation'. He received the 1921 Nobel Prize in "
                                "Physics 'for his services to theoretical physics, "
                                "and especially for his discovery of the law of the "
                                "photoelectric effect', a pivotal step in the "
                                "development of quantum theory. His work is also "
                                "known for its influence on the philosophy of science. "
                                "In a 1999 poll of 130 leading physicists worldwide by "
                                "the British journal Physics World, Einstein was ranked "
                                "the greatest physicist of all time. His intellectual "
                                "achievements and originality have made Einstein "
                                "synonymous with genius."
                            )
                        ],
                        "answer": [
                            (
                                "Albert Einstein, born on 14 March 1879, was a "
                                "German-born theoretical physicist, widely held to be "
                                "one of the greatest and most influential scientists "
                                "of all time. He received the 1921 Nobel Prize in "
                                "Physics for his services to theoretical physics. He "
                                "published 4 papers in 1905. Einstein moved to "
                                "Switzerland in 1895."
                            )
                        ],
                    },
                    "outputs": {
                        "classifications": [
                            {
                                "statement": (
                                    "Albert Einstein, born on 14 March 1879, was a "
                                    "German-born theoretical physicist, widely held to "
                                    "be one of the greatest and most influential "
                                    "scientists of all time."
                                ),
                                "reason": "The date of birth of Einstein is mentioned in the context.",
                                "attributed": 1,
                            },
                            {
                                "statement": (
                                    "He received the 1921 Nobel Prize in Physics for "
                                    "his services to theoretical physics."
                                ),
                                "reason": "The exact sentence is present in the given context.",
                                "attributed": 1,
                            },
                            {
                                "statement": "He published 4 papers in 1905.",
                                "reason": "There is no mention about papers he wrote in the context.",
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

    def run(
        self,
        questions: list[str],
        contexts: list[str] | list[list[str]],
        answers: list[str],
        verbose: bool = False,
    ) -> list[float]:
        """
        Evaluate the context recall for each question.

        Args:
            questions (list[str]): List of questions.
            contexts (list[str] or list[list[str]]): Could be a single list of context strings
                or a list of sublists (each sublist is for one question).
            answers (list[str]): List of answers.
            verbose (bool): Flag to enable verbose logging.

        Returns:
            list[float]: List of context recall scores for each question.
        """
        # Pass everything to the Pydantic model, which will unify "contexts".
        input_data = ContextRecallInput(
            questions=questions,
            contexts=contexts,
            answers=answers,
            verbose=verbose,
        )

        final_scores = []

        for idx in range(len(input_data.questions)):
            question = input_data.questions[idx]
            context = input_data.contexts[idx]  # Now guaranteed to be a single string
            answer = input_data.answers[idx]

            # Evaluate classification
            result = self._classification_evaluator.run(
                question=[question],
                context=[context],
                answer=[answer],
            )

            # Extract classifications
            classifications_raw = result["results"][0]["classifications"]
            classifications = []
            for item in classifications_raw:
                classification_item = ClassificationItem(
                    statement=item["statement"],
                    reason=item["reason"],
                    attributed=int(item["attributed"]),
                )
                classifications.append(classification_item)

            # Compute the score
            attributed_list = [item.attributed for item in classifications]
            num_sentences = len(attributed_list)
            num_attributed = sum(attributed_list)
            score = num_attributed / num_sentences if num_sentences > 0 else 0.0
            final_scores.append(score)

            if input_data.verbose:
                logger.debug(f"Question: {question}")
                logger.debug(f"Answer: {answer}")
                logger.debug(f"Context: {context}")
                logger.debug("Classifications:")
                logger.debug(json.dumps([item.dict() for item in classifications], indent=2))
                logger.debug(f"Context Recall Score: {score}")
                logger.debug("-" * 50)

        output_data = ContextRecallOutput(final_scores=final_scores)
        return output_data.final_scores
