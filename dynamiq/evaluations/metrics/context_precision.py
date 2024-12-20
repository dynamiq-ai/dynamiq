import logging
from functools import cached_property

from pydantic import BaseModel, PrivateAttr, computed_field, field_validator, model_validator

from dynamiq.components.evaluators.llm_evaluator import LLMEvaluator
from dynamiq.nodes.llms import BaseLLM

# Configure logging
logger = logging.getLogger(__name__)


class ContextPrecisionInput(BaseModel):
    """
    Input model for context precision evaluation.

    Attributes:
        questions (list[str]): List of questions.
        answers (list[str]): List of corresponding answers.
        contexts_list (list[list[str]] | list[str]): Either a list of lists of
            strings or a list of strings; it will be normalized to a list of lists.
        verbose (bool): Flag to enable verbose logging.
    """

    questions: list[str]
    answers: list[str]
    contexts_list: list[list[str]] | list[str]
    verbose: bool = False

    @field_validator("contexts_list", mode="before")
    def normalize_contexts_list(cls, v):
        # If the user provides a list[str], wrap it into [list[str]].
        # If the user provides a list[list[str]], leave as-is.
        # If neither, raise an error.
        if isinstance(v, list):
            if all(isinstance(item, str) for item in v):
                return [v]  # e.g. ["foo", "bar"] -> [["foo", "bar"]]
            if all(isinstance(item, list) and all(isinstance(x, str) for x in item) for item in v):
                return v
        raise ValueError("contexts_list must be either a list of strings or a list of list of strings.")

    @model_validator(mode="after")
    def check_equal_length(self):
        # Now self.contexts_list will always be a list of lists of strings
        if not (len(self.questions) == len(self.answers) == len(self.contexts_list)):
            raise ValueError("questions, answers, and contexts_list must have the same length.")
        return self


class ContextPrecisionOutput(BaseModel):
    """
    Output model for context precision evaluation.

    Attributes:
        final_scores (List[float]): List of context precision scores.
    """

    final_scores: list[float]


class VerdictResult(BaseModel):
    """
    Model for the verdict result from the evaluator.

    Attributes:
        verdict (int): 1 if the context was useful, 0 otherwise.
        reason (str): Reason for the verdict.
    """

    verdict: int
    reason: str

    @field_validator("verdict")
    @classmethod
    def validate_verdict(cls, v):
        if v not in (0, 1):
            raise ValueError("Verdict must be either 0 or 1.")
        return v


class ContextPrecisionEvaluator(BaseModel):
    """
    Evaluator class for context precision metric.

    Attributes:
        llm (BaseLLM): The language model to use for evaluation.
    """

    llm: BaseLLM

    # Private attribute (not part of the Pydantic model fields)
    _context_precision_evaluator: LLMEvaluator = PrivateAttr()

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
        context_precision_instructions = (
            'Given a "Question", "Answer", and "Context", verify if the Context was '
            "useful in arriving at the given Answer.\n"
            '- Provide a "verdict": 1 if useful, 0 if not.\n'
            '- Provide a brief "reason" for the verdict.\n'
            '- Output the result as a JSON object with keys "verdict" and "reason".\n'
            "- Ensure that your response is valid JSON, using double quotes for all "
            "strings."
        )

        self._context_precision_evaluator = LLMEvaluator(
            instructions=context_precision_instructions.strip(),
            inputs=[
                {"name": "question", "type": list[str]},
                {"name": "answer", "type": list[str]},
                {"name": "context", "type": list[str]},
            ],
            outputs=[
                {"name": "verdict", "type": int},
                {"name": "reason", "type": str},
            ],
            examples=[
                {
                    "inputs": {
                        "question": ["What can you tell me about Albert Einstein?"],
                        "answer": [
                            (
                                "Albert Einstein, born on 14 March 1879, was a German-born theoretical "
                                "physicist, widely held to be one of the greatest and most influential "
                                "scientists of all time. He received the 1921 Nobel Prize in Physics "
                                "for his services to theoretical physics."
                            )
                        ],
                        "context": [
                            (
                                "Albert Einstein (14 March 1879 – 18 April 1955) was a German-born "
                                "theoretical physicist, widely held to be one of the greatest and most "
                                "influential scientists of all time. Best known for developing the theory "
                                "of relativity, he also made important contributions to quantum mechanics, "
                                "and was thus a central figure in the revolutionary reshaping of the "
                                "scientific understanding of nature that modern physics accomplished in "
                                "the first decades of the twentieth century. His mass–energy equivalence "
                                "formula E = mc2, which arises from relativity theory, has been called "
                                "'the world's most famous equation'. He received the 1921 Nobel Prize in "
                                "Physics 'for his services to theoretical physics, and especially for his "
                                "discovery of the law of the photoelectric effect', a pivotal step in the "
                                "development of quantum theory. His work is also known for its influence on "
                                "the philosophy of science. In a 1999 poll of 130 leading physicists "
                                "worldwide by the British journal Physics World, Einstein was ranked the "
                                "greatest physicist of all time. His intellectual achievements and "
                                "originality have made Einstein synonymous with genius."
                            )
                        ],
                    },
                    "outputs": {
                        "verdict": 1,
                        "reason": (
                            "The context provides detailed information about Albert Einstein that is "
                            "reflected in the answer, including his birthdate, his contributions to "
                            "physics, and his Nobel Prize."
                        ),
                    },
                },
                {
                    "inputs": {
                        "question": ["Who won the 2020 ICC World Cup?"],
                        "answer": ["England"],
                        "context": [
                            (
                                "The 2022 ICC Men's T20 World Cup, held from October 16 to November 13, "
                                "2022, in Australia, was the eighth edition of the tournament. Originally "
                                "scheduled for 2020, it was postponed due to the COVID-19 pandemic. "
                                "England emerged victorious, defeating Pakistan by five wickets in the "
                                "final to clinch their second ICC Men's T20 World Cup title."
                            )
                        ],
                    },
                    "outputs": {
                        "verdict": 1,
                        "reason": (
                            "The context explains that the 2020 ICC World Cup was postponed to 2022, and "
                            "that England won the tournament, which is directly relevant to the answer."
                        ),
                    },
                },
                {
                    "inputs": {
                        "question": ["What is the tallest mountain in the world?"],
                        "answer": ["Mount Everest."],
                        "context": [
                            (
                                "The Andes is the longest continental mountain range in the world, located "
                                "in South America. It stretches across seven countries and features many of "
                                "the highest peaks in the Western Hemisphere. The range is known for its "
                                "diverse ecosystems, including the high-altitude Andean Plateau and the "
                                "Amazon rainforest."
                            )
                        ],
                    },
                    "outputs": {
                        "verdict": 0,
                        "reason": (
                            "The context discusses the Andes mountain range, which does not include Mount "
                            "Everest. Therefore, the context was not useful in arriving at the answer."
                        ),
                    },
                },
            ],
            llm=self.llm,
        )

    @staticmethod
    def calculate_average_precision(verdicts: list[int]) -> float:
        """
        Calculate the average precision based on verdicts.

        Args:
            verdicts (List[int]): List of verdicts (1 for relevant, 0 for not relevant).

        Returns:
            float: The average precision score.
        """
        numerator = 0.0
        cumulative_hits = 0
        total_relevant = sum(verdicts)
        if total_relevant == 0:
            return 0.0
        for i, verdict in enumerate(verdicts):
            if verdict == 1:
                cumulative_hits += 1
                precision_at_i = cumulative_hits / (i + 1)
                numerator += precision_at_i
        average_precision = numerator / total_relevant
        return average_precision

    def run(
        self,
        questions: list[str],
        answers: list[str],
        contexts_list: list[list[str]] | list[str],
        verbose: bool = False,
    ) -> list[float]:
        """
        Evaluate the context precision for each question.

        Args:
            questions (list[str]): List of questions.
            answers (list[str]): List of corresponding answers.
            contexts_list (list[list[str]] | list[str]): Either a list of contexts
                per question (list[list[str]]) or a single list of context strings (list[str]).
            verbose (bool): Flag to enable verbose logging.

        Returns:
            list[float]: List of context precision scores for each question.
        """
        # Pass everything to the Pydantic model
        input_data = ContextPrecisionInput(
            questions=questions,
            answers=answers,
            contexts_list=contexts_list,
            verbose=verbose,
        )

        final_scores = []

        for idx in range(len(input_data.questions)):
            question = input_data.questions[idx]
            answer = input_data.answers[idx]
            contexts = input_data.contexts_list[idx]  # This is now a list[str]

            verdicts = []
            for context in contexts:
                # Prepare inputs for the evaluator
                result = self._context_precision_evaluator.run(
                    question=[question],
                    answer=[answer],
                    context=[context],
                )
                # Extract the verdict (ensure it's an int)
                verdict_raw = result["results"][0]["verdict"]
                verdict = int(verdict_raw) if not isinstance(verdict_raw, str) else int(verdict_raw.strip())
                verdicts.append(verdict)

                if input_data.verbose:
                    reason = result["results"][0]["reason"]
                    logger.debug(f"Question: {question}")
                    logger.debug(f"Answer: {answer}")
                    logger.debug(f"Context: {context}")
                    logger.debug(f"Verdict: {verdict}")
                    logger.debug(f"Reason: {reason}")
                    logger.debug("-" * 50)

            # Calculate average precision for this question
            score = self.calculate_average_precision(verdicts)
            final_scores.append(score)

            if input_data.verbose:
                logger.debug(f"Average Precision Score: {score}")
                logger.debug("=" * 50)

        output_data = ContextPrecisionOutput(final_scores=final_scores)
        return output_data.final_scores
