from pydantic import BaseModel, PrivateAttr, field_validator, model_validator
from dynamiq.evaluations import BaseEvaluator
from dynamiq.evaluations.llm_evaluator import LLMEvaluator
from dynamiq.nodes.llms import BaseLLM
from dynamiq.utils.logger import logger


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
    def normalize_contexts_list(cls, value):
        # If the user provides a list[str], wrap it into [list[str]].
        # If the user provides a list[list[str]], leave as-is.
        # Otherwise, raise an error.
        if isinstance(value, list):
            if all(isinstance(item, str) for item in value):
                return [value]  # e.g. ["foo", "bar"] becomes [["foo", "bar"]]
            if all(isinstance(item, list) and all(isinstance(x, str) for x in item) for item in value):
                return value
        raise ValueError("contexts_list must be either a list of strings or a list of list of strings.")

    @model_validator(mode="after")
    def check_equal_length(self):
        # Now self.contexts_list will always be a list of lists of strings.
        if not (len(self.questions) == len(self.answers) == len(self.contexts_list)):
            raise ValueError("questions, answers, and contexts_list must have the same length.")
        return self


class ContextPrecisionRunResult(BaseModel):
    """
    Result model for the context precision evaluation.

    Attributes:
        score (float): The computed context precision score.
        reasoning (str): Detailed reasoning explaining how the score was derived.
    """
    score: float
    reasoning: str


class ContextPrecisionOutput(BaseModel):
    """
    Output model for context precision evaluation.

    Attributes:
        results (list[ContextPrecisionRunResult]): List of evaluation results.
    """
    results: list[ContextPrecisionRunResult]


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
    def validate_verdict(cls, value):
        if value not in (0, 1):
            raise ValueError("Verdict must be either 0 or 1.")
        return value


class ContextPrecisionEvaluator(BaseEvaluator):
    """
    Evaluator class for context precision metric.

    Attributes:
        llm (BaseLLM): The language model to use for evaluation.
    """
    name: str = "ContextPrecision"
    llm: BaseLLM

    # Private attribute (not a Pydantic model field)
    _context_precision_evaluator: LLMEvaluator = PrivateAttr()

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
                                "and was thus a central figure in modern physics. His mass–energy equivalence "
                                "formula E = mc2 has been called 'the world's most famous equation'."
                            )
                        ],
                    },
                    "outputs": {
                        "reason": (
                            "The context provides detailed info about Einstein that is reflected in the "
                            "answer (e.g. his contributions and Nobel Prize)."
                        ),
                        "verdict": 1,
                    },
                },
                {
                    "inputs": {
                        "question": ["Who won the 2020 ICC World Cup?"],
                        "answer": ["England"],
                        "context": [
                            (
                                "The 2022 ICC Men's T20 World Cup was postponed from 2020 due to COVID-19. "
                                "England won the tournament, defeating Pakistan in the final."
                            )
                        ],
                    },
                    "outputs": {
                        "reason": (
                            "The context explains the tournament details and mentions England's victory, "
                            "which is directly relevant."
                        ),
                        "verdict": 1,
                    },
                },
                {
                    "inputs": {
                        "question": ["What is the tallest mountain in the world?"],
                        "answer": ["Mount Everest."],
                        "context": [
                            (
                                "The Andes is the longest continental mountain range, but it does not "
                                "contain Mount Everest."
                            )
                        ],
                    },
                    "outputs": {
                        "reason": ("The context discusses the Andes, which is unrelated to Mount Everest."),
                        "verdict": 0,
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
            verdicts (list[int]): List of verdicts (1 for useful, 0 for not useful).

        Returns:
            float: The average precision score.
        """
        numerator = 0.0
        cumulative_hits = 0
        total_relevant = sum(verdicts)
        if total_relevant == 0:
            return 0.0
        for index, verdict in enumerate(verdicts):
            if verdict == 1:
                cumulative_hits += 1
                precision_at_index = cumulative_hits / (index + 1)
                numerator += precision_at_index
        average_precision = numerator / total_relevant
        return round(float(average_precision), 2)

    def _build_reasoning(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        verdicts: list[int],
        verdict_details: list[str],
        average_precision: float,
    ) -> str:
        """
        Build a detailed reasoning string for context precision evaluation.

        Explains:
        • Each context is evaluated with a verdict (emojis used: ✅ for supported, ❌ for not).
        • The corresponding explanation for each verdict.
        • How the average precision is calculated.

        Args:
            question (str): The evaluation question.
            answer (str): The answer text.
            contexts (list[str]): List of contexts evaluated.
            verdicts (list[int]): List of verdicts (1 or 0) for each context.
            verdict_details (list[str]): List of explanations for each verdict.
            average_precision (float): The average precision score.

        Returns:
            str: Detailed reasoning.
        """
        reasoning_strings = ["Reasoning:", "", f"Question: {question}", f"Answer: {answer}", "", "Context Evaluations:"]
        for context_text, verdict_value, detail in zip(contexts, verdicts, verdict_details):
            verdict_mark = "✅" if verdict_value == 1 else "❌"
            reasoning_strings.extend(
                [
                    f" - Context: {context_text}",
                    f"   Verdict: {verdict_mark} (value: {verdict_value})",
                    f"   Explanation: {detail}",
                    "",
                ]
            )
        reasoning_strings.append(f"Average Precision Score = {average_precision:.2f}")
        return "\n".join(reasoning_strings)

    def run_single(
        self, question: str, answer: str, contexts: list[str], verbose: bool = False
    ) -> ContextPrecisionRunResult:
        """
        Evaluate the context precision for a single sample.

        Args:
            question (str): The question.
            answer (str): The corresponding answer.
            contexts (list[str]): A list of contexts for this question.
            verbose (bool): Flag to enable verbose logging.

        Returns:
            ContextPrecisionRunResult: Contains the computed average precision score and detailed reasoning.
        """
        verdicts = []
        verdict_details = []
        for context in contexts:
            evaluation_result = self._context_precision_evaluator.run(
                question=[question], answer=[answer], context=[context]
            )
            if ("results" not in evaluation_result) or (not evaluation_result["results"]):
                default_verdict = 0
                verdicts.append(default_verdict)
                verdict_details.append("No results returned from evaluator.")
                if verbose:
                    logger.debug(f"Missing results for context: {context}. Defaulting verdict to {default_verdict}.")
                continue

            result_item = evaluation_result["results"][0]
            verdict_raw = result_item.get("verdict", "0")
            try:
                verdict = int(verdict_raw) if not isinstance(verdict_raw, str) else int(verdict_raw.strip())
            except (ValueError, AttributeError):
                verdict = 0
            verdicts.append(verdict)
            verdict_details.append(result_item.get("reason", "No reasoning provided"))

            if verbose:
                logger.debug(f"Question: {question}")
                logger.debug(f"Answer: {answer}")
                logger.debug(f"Context: {context}")
                logger.debug(f"Verdict: {verdict}")
                logger.debug(f"Reason: {result_item.get('reason', 'No reasoning provided')}")
                logger.debug("-" * 50)

        average_precision = self.calculate_average_precision(verdicts)
        reasoning_text = self._build_reasoning(question, answer, contexts, verdicts, verdict_details, average_precision)
        if verbose:
            logger.debug(f"Average Precision Score: {average_precision}")
            logger.debug("=" * 50)
        return ContextPrecisionRunResult(score=average_precision, reasoning=reasoning_text)

    def run(
        self,
        questions: list[str],
        answers: list[str],
        contexts_list: list[list[str]] | list[str],
        verbose: bool = False,
    ) -> ContextPrecisionOutput:
        """
        Evaluate the context precision for each question.

        Args:
            questions (list[str]): List of questions.
            answers (list[str]): List of corresponding answers.
            contexts_list (list[list[str]] | list[str]): Either a list of contexts per question
                (list[list[str]]) or a single list of context strings (list[str]).
            verbose (bool): Flag to enable verbose logging (for internal logging only).

        Returns:
            ContextPrecisionOutput: Contains a list of context precision scores and reasoning.
        """
        run_input = ContextPrecisionInput(
            questions=questions,
            answers=answers,
            contexts_list=contexts_list,
            verbose=verbose,
        )
        results_output = []
        for index in range(len(run_input.questions)):
            question = run_input.questions[index]
            answer = run_input.answers[index]
            contexts = run_input.contexts_list[index]
            result_single = self.run_single(
                question=question, answer=answer, contexts=contexts, verbose=run_input.verbose
            )
            results_output.append(result_single)
        return ContextPrecisionOutput(results=results_output)
