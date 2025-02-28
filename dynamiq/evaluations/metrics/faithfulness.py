from typing import Any

from pydantic import BaseModel, PrivateAttr, field_validator, model_validator
from dynamiq.evaluations import BaseEvaluator
from dynamiq.evaluations.llm_evaluator import LLMEvaluator
from dynamiq.nodes.llms import BaseLLM
from dynamiq.utils.logger import logger


class SimplifyStatementsInput(BaseModel):
    """
    Input model for simplifying statements.

    Attributes:
        questions (list[str]): List of questions.
        answers (list[str]): List of corresponding answers.
    """
    questions: list[str]
    answers: list[str]

    @model_validator(mode="after")
    def check_equal_length(self):
        if len(self.questions) != len(self.answers):
            raise ValueError("Questions and answers must have the same length.")
        return self


class SimplifyStatementsOutput(BaseModel):
    """
    Output model for simplified statements.

    Attributes:
        statements_list (list[list[str]]): List of lists of simplified statements.
    """
    statements_list: list[list[str]]


class NLIInput(BaseModel):
    """
    Input model for NLI evaluation.

    Attributes:
        contexts (list[str]): List of contexts.
        statements_list (list[list[str]]): List of lists of statements.
    """
    contexts: list[str]
    statements_list: list[list[str]]

    @model_validator(mode="after")
    def check_equal_length(self):
        if len(self.contexts) != len(self.statements_list):
            raise ValueError("Contexts and statements_list must have the same length.")
        return self


class NLIResultItem(BaseModel):
    """
    Model for individual NLI result.

    Attributes:
        statement (str): The statement being evaluated.
        verdict (int): 1 if faithful, 0 otherwise.
        reason (str): Reason for the verdict.
    """
    statement: str
    verdict: int
    reason: str

    @field_validator("verdict")
    @classmethod
    def validate_verdict(cls, v):
        if v not in (0, 1):
            raise ValueError("Verdict must be either 0 or 1.")
        return v


class NLIOutput(BaseModel):
    """
    Output model for NLI evaluation.

    Attributes:
        results_list (list[list[NLIResultItem]]): List of lists of NLI results.
    """
    results_list: list[list[NLIResultItem]]


class RunInput(BaseModel):
    """
    Input model for running the faithfulness evaluation.

    Attributes:
        questions (list[str]): List of questions.
        answers (list[str]): List of corresponding answers.
        contexts (list[str] | list[list[str]]): List of context texts for each question,
            which can be either one string per question or multiple strings per question.
        verbose (bool): Flag to enable verbose logging.
    """
    questions: list[str]
    answers: list[str]
    contexts: list[str] | list[list[str]]
    verbose: bool = False

    @field_validator("contexts", mode="before")
    def unify_contexts(cls, value):
        """
        If contexts is list[list[str]], join each sublist with a space.
        Otherwise, if list[str], leave as-is.
        """
        if not isinstance(value, list):
            raise ValueError("contexts must be either a list of strings or a list of list of strings")
        if all(isinstance(item, list) and all(isinstance(x, str) for x in item) for item in value):
            return [" ".join(sublist) for sublist in value]
        if all(isinstance(item, str) for item in value):
            return value
        raise ValueError("contexts must be either a list[str] or a list[list[str]]")

    @model_validator(mode="after")
    def check_equal_length(self):
        if not (len(self.questions) == len(self.answers) == len(self.contexts)):
            raise ValueError("Questions, answers, and contexts must have the same length.")
        return self


class FaithfulnessRunResult(BaseModel):
    """
    Result model for faithfulness evaluation.

    Attributes:
        score (float): The computed faithfulness score.
        reasoning (str): Detailed reasoning explaining the evaluation.
    """
    score: float
    reasoning: str


class RunOutput(BaseModel):
    """
    Output model for faithfulness evaluation.

    Attributes:
        results (list[FaithfulnessRunResult]): List of results with score and reasoning.
    """

    results: list[FaithfulnessRunResult]


class FaithfulnessRunSingleInput(BaseModel):
    """
    Single-run input model for faithfulness evaluation.

    Attributes:
        question (str): The question.
        answer (str): The corresponding answer.
        context (str): The context for the evaluation.
        verbose (bool): Flag to enable verbose logging.
    """

    question: str
    answer: str
    context: str
    verbose: bool = False


class FaithfulnessEvaluator(BaseEvaluator):
    """
    Evaluator class for faithfulness metric.

    Pipeline:
      1) Statement Simplification: The answer is broken down into unambiguous statements
         with no pronouns.
      2) NLI Evaluation: Each statement is compared against the context. A verdict of 1 means
         the statement is faithful; 0 means it is not.
      3) Score Computation: The score is the ratio of faithful statements to total statements.
      4) Detailed Reasoning: A user-friendly explanation is output with every step.

    Attributes:
        llm (BaseLLM): The language model used for evaluation.
    """
    name: str = "Faithfulness"
    llm: BaseLLM

    _statement_simplifier: LLMEvaluator = PrivateAttr()
    _nli_evaluator: LLMEvaluator = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self._initialize_evaluators()

    def _initialize_evaluators(self):
        simplify_instructions = (
            "Given a 'Question' and an 'Answer', break down each sentence in the "
            "Answer into one or more fully understandable statements.\n"
            "- Ensure no pronouns are used in each statement.\n"
            "- Output as a JSON object with key 'statements', where the value is a "
            "list of statements.\n"
            "- Ensure your response is valid JSON, using double quotes for all strings."
        )
        self._statement_simplifier = LLMEvaluator(
            instructions=simplify_instructions.strip(),
            inputs=[{"name": "question", "type": list[str]}, {"name": "answer", "type": list[str]}],
            outputs=[{"name": "statements", "type": list[str]}],
            examples=[
                {
                    "inputs": {
                        "question": ["Who was Albert Einstein and what is he best known for?"],
                        "answer": [
                            "He was a German-born theoretical physicist, widely "
                            "acknowledged to be one of the greatest and most influential "
                            "physicists of all time. He was best known for developing "
                            "the theory of relativity, he also made important contributions "
                            "to the development of quantum mechanics."
                        ],
                    },
                    "outputs": {
                        "statements": [
                            "Albert Einstein was a German-born theoretical physicist.",
                            "Albert Einstein is recognized as one of the greatest and most influential "
                            "physicists of all time.",
                            "Albert Einstein was best known for developing the theory of relativity.",
                            "Albert Einstein also made important contributions to the development of "
                            "quantum mechanics.",
                        ]
                    },
                },
            ],
            llm=self.llm,
        )
        nli_instructions = (
            "Your task is to judge the faithfulness of a series of statements based "
            "on a given Context.\n"
            "- For each statement, return 'verdict': 1 if it can be directly inferred "
            "from the Context, or 0 if not.\n"
            "- Provide a brief 'reason' for the verdict.\n"
            "- Output as a JSON object with key 'results', where the value is a list of "
            "dictionaries with keys 'statement', 'verdict', and 'reason'.\n"
            "- Ensure your response is valid JSON, using double quotes for all strings."
        )
        self._nli_evaluator = LLMEvaluator(
            instructions=nli_instructions.strip(),
            inputs=[{"name": "context", "type": list[str]}, {"name": "statements", "type": list[list[str]]}],
            outputs=[{"name": "results", "type": list[dict[str, Any]]}],
            examples=[
                {
                    "inputs": {
                        "context": [
                            "John is a student at XYZ University. He is pursuing a "
                            "degree in Computer Science. He is enrolled in several courses "
                            "this semester, including Data Structures, Algorithms, and "
                            "Database Management. John is a diligent student and spends a "
                            "significant amount of time studying and completing assignments. "
                            "He often stays late in the library to work on his projects."
                        ],
                        "statements": [
                            [
                                "John is majoring in Biology.",
                                "John is taking a course on Artificial Intelligence.",
                                "John is a dedicated student.",
                                "John has a part-time job.",
                            ]
                        ],
                    },
                    "outputs": {
                        "results": [
                            {
                                "statement": "John is majoring in Biology.",
                                "reason": "The context states that John is pursuing a degree in "
                                "Computer Science, not Biology.",
                                "verdict": 0,
                            },
                            {
                                "statement": "John is taking a course on Artificial Intelligence.",
                                "reason": "The context lists his courses, and Artificial Intelligence "
                                "is not mentioned.",
                                "verdict": 0,
                            },
                            {
                                "statement": "John is a dedicated student.",
                                "reason": "The context mentions he spends significant time studying and "
                                "stays late to work on projects.",
                                "verdict": 1,
                            },
                            {
                                "statement": "John has a part-time job.",
                                "reason": "There is no information in the context about John having a "
                                "part-time job.",
                                "verdict": 0,
                            },
                        ]
                    },
                },
            ],
            llm=self.llm,
        )

    def simplify_statements(self, questions: list[str], answers: list[str]) -> list[list[str]]:
        """
        Simplify the answers into clear, unambiguous statements.

        Args:
            questions (list[str]): List of questions.
            answers (list[str]): List of corresponding answers.

        Returns:
            list[list[str]]: Simplified statements.
        """
        input_data = SimplifyStatementsInput(questions=questions, answers=answers)
        results = self._statement_simplifier.run(question=input_data.questions, answer=input_data.answers)
        statements_list = []
        for result in results["results"]:
            statements = result.get("statements")
            if isinstance(statements, list):
                statements_list.append(statements)
            else:
                statements_list.append([statements])
        output_data = SimplifyStatementsOutput(statements_list=statements_list)
        return output_data.statements_list

    def check_faithfulness(self, contexts: list[str], statements_list: list[list[str]]) -> list[list[NLIResultItem]]:
        """
        Check the faithfulness of statements against contexts.

        Args:
            contexts (list[str]): List of contexts.
            statements_list (list[list[str]]): Simplified statements.

        Returns:
            list[list[NLIResultItem]]: NLI results.
        """
        input_data = NLIInput(contexts=contexts, statements_list=statements_list)
        results = self._nli_evaluator.run(context=input_data.contexts, statements=input_data.statements_list)
        results_list = []
        for result in results["results"]:
            items = []
            for item in result["results"]:
                nli_item = NLIResultItem(
                    statement=item["statement"], verdict=int(item["verdict"]), reason=item["reason"]
                )
                items.append(nli_item)
            results_list.append(items)
        output_data = NLIOutput(results_list=results_list)
        return output_data.results_list

    def _build_reasoning(
        self,
        statements: list[str],
        nli_results: list[NLIResultItem],
        num_statements: int,
        num_faithful: int,
        score: float,
    ) -> str:
        """
        Build detailed reasoning for the faithfulness evaluation.

        This explanation covers:
        • How the answer was simplified into candidate statements.
        • The NLI verdict for each statement along with brief reasons.
        • The calculation of the final faithfulness score.

        Args:
            statements (list[str]): Simplified candidate statements.
            nli_results (list[NLIResultItem]): NLI results.
            num_statements (int): Total number of statements.
            num_faithful (int): Number of statements deemed faithful.
            score (float): The computed faithfulness score.

        Returns:
            str: Detailed reasoning.
        """
        lines = []
        lines.extend(
            [
                "Reasoning:",
                "",
                "Overview:",
                "  The answer is first simplified into clear statements (without pronouns).",
                "  Each statement is then evaluated for faithfulness against the context via NLI.",
                "  A '✅' indicates the statement is faithful; '❌' indicates it is not.",
                "",
                "1. Simplified Statements:",
            ]
        )

        # Add each simplified statement
        lines.extend([f"   - {stmt}" for stmt in statements])

        lines.extend(["", "2. NLI Evaluation Results:"])

        # Add each NLI result with its verdict and explanation
        for res in nli_results:
            mark = "✅" if res.verdict == 1 else "❌"
            lines.extend([f" {mark} - {res.statement}", f"     Explanation: {res.reason}", ""])

        lines.extend(
            [
                f" -> Faithful Statements = {num_faithful} out of {num_statements}",
                f" -> Faithfulness Score = {score:.2f} (faithful/total)",
                "",
                f"Final Score = {score:.2f}",
            ]
        )

        return "\n".join(lines)

    def run_single(self, question: str, answer: str, context: str, verbose: bool = False) -> FaithfulnessRunResult:
        """
        Evaluate the faithfulness for a single sample.

        Args:
            question (str): The question.
            answer (str): The corresponding answer.
            context (str): The evaluation context.
            verbose (bool): Flag to enable verbose logging.

        Returns:
            FaithfulnessRunResult: The result with faithfulness score and detailed reasoning.
        """
        # Validate the single input using a pydantic model
        single_input = FaithfulnessRunSingleInput(question=question, answer=answer, context=context, verbose=verbose)

        # Simplify the answer (using question and answer)
        statements_list = self.simplify_statements([single_input.question], [single_input.answer])
        if not statements_list or statements_list[0] is None:
            if single_input.verbose:
                logger.debug(f"No simplified statements for answer: {single_input.answer}. Using empty list.")
            statements = []
        else:
            statements = statements_list[0]

        # Evaluate faithfulness via NLI
        nli_results_list = self.check_faithfulness([single_input.context], [statements])
        if not nli_results_list or nli_results_list[0] is None:
            if single_input.verbose:
                logger.debug("No NLI results for context or statements. Using empty list for NLI evaluation.")
            nli_results = []
        else:
            nli_results = nli_results_list[0]

        num_statements = len(nli_results)
        num_faithful = sum(item.verdict for item in nli_results)
        score = num_faithful / num_statements if num_statements else 0.0
        score = round(float(score), 2)

        reasoning = self._build_reasoning(
            statements=statements,
            nli_results=nli_results,
            num_statements=num_statements,
            num_faithful=num_faithful,
            score=score,
        )
        if single_input.verbose:
            logger.debug(f"Question: {single_input.question}")
            logger.debug(f"Answer: {single_input.answer}")
            logger.debug(f"Context: {single_input.context}")
            logger.debug("Simplified Statements:")
            logger.debug(statements)
            logger.debug("NLI Results:")
            logger.debug([item.model_dump() for item in nli_results])
            logger.debug(reasoning)
            logger.debug("-" * 50)
        result_item = FaithfulnessRunResult(score=score, reasoning=reasoning)
        return result_item

    def run(
        self,
        questions: list[str],
        answers: list[str],
        contexts: list[str] | list[list[str]],
        verbose: bool = False,
    ) -> RunOutput:
        """
        Evaluate the faithfulness of answers given contexts.

        Pipeline:
        1) Simplify the answer into clear candidate statements.
        2) Evaluate each statement via NLI against the context.
        3) Compute the faithfulness score as the ratio of faithful statements.
        4) Generate detailed reasoning explaining the process and final score.

        Args:
            questions (list[str]): List of questions.
            answers (list[str]): List of corresponding answers.
            contexts (list[str] | list[list[str]]): List of context texts.
            verbose (bool): Flag to enable verbose logging.

        Returns:
            RunOutput: Contains a list of FaithfulnessRunResult.
        """
        input_data = RunInput(questions=questions, answers=answers, contexts=contexts, verbose=verbose)
        results_out = []
        for idx in range(len(input_data.questions)):
            question = input_data.questions[idx]
            answer = input_data.answers[idx]
            context = input_data.contexts[idx]
            result_item = self.run_single(question=question, answer=answer, context=context, verbose=input_data.verbose)
            results_out.append(result_item)
        output_data = RunOutput(results=results_out)
        return output_data
