from pydantic import BaseModel, Field, PrivateAttr, model_validator
from dynamiq.evaluations import BaseEvaluator
from dynamiq.evaluations.llm_evaluator import LLMEvaluator
from dynamiq.nodes.llms import BaseLLM
from dynamiq.utils.logger import logger


class ExtractStatementsInput(BaseModel):
    """
    Input model for extracting candidate statements.
    """
    question: str = Field(description="The question to answer")
    answer: str = Field(description="The answer to the question")


class ExtractStatementsOutput(BaseModel):
    """
    Output model for extracted candidate statements.
    """
    statements: list[str] = Field(description="The generated statements")


class ClassifyStatementInput(BaseModel):
    """
    Input model for classifying a candidate pair.
    """
    question: str = Field(description="The question for context")
    answer_statement: str = Field(description="A candidate statement from the answer")
    ground_truth_statement: str = Field(
        description=("A string of candidate statements extracted from the ground truth answer")
    )


class ClassifyStatementOutput(BaseModel):
    """
    Output model for classifying a candidate pair.
    """
    match: bool = Field(
        description=("Verdict: true if the core fact of the statement is supported by the ground truth")
    )
    reasoning: str = Field(description="Explanation for why the statement is or is not supported")


class RunInput(BaseModel):
    """
    Input model for running the evaluator.
    """
    questions: list[str]
    answers: list[str]
    ground_truth_answers: list[str]
    verbose: bool = False

    @model_validator(mode="after")
    def check_equal_length(self):
        if len(self.questions) != len(self.answers) or len(self.questions) != len(self.ground_truth_answers):
            raise ValueError("Questions, answers, and ground truth answers must have the same length.")
        return self


class AnswerCorrectnessRunSingleInput(BaseModel):
    """
    Single-run input model for answer correctness evaluation.
    """

    question: str = Field(description="The question to answer")
    answer: str = Field(description="The answer to the question")
    ground_truth_answer: str = Field(description="The ground truth answer")
    verbose: bool = False


class F1Result(BaseModel):
    """
    Model for F1 score calculation.
    """
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


class RunResult(BaseModel):
    """
    Result containing final score and detailed, user-friendly reasoning.
    """
    score: float
    reasoning: str


class RunOutput(BaseModel):
    """
    Output model for final scores and detailed reasoning.
    """
    results: list[RunResult]


class AnswerCorrectnessEvaluator(BaseEvaluator):
    """
    Evaluator for computing answer correctness using candidate statements with
    explanation of match decisions and weighted scoring.

    Overview:
    • The evaluator splits both the answer and the ground truth answer into a set of
        core fact “candidate statements.”
    • It then compares each statement from the answer with the statements from the ground
        truth answer to decide if the core fact is present. A "✅" indicates that the statement
        is supported by the ground truth, whereas a "❌" indicates it is not.
    • Similarly, ground truth statements are checked against the answer to see if any are missing.
    • Based on these comparisons, the metrics are computed:
        - TP (True Positive): Number of answer statements that are correctly supported.
        - FP (False Positive): Number of answer statements that are not supported.
        - FN (False Negative): Number of ground truth statements missing from the answer.
        - Precision = TP / (TP + FP)
        - Recall    = TP / (TP + FN)
        - F1 Score  = 2 * (Precision * Recall) / (Precision + Recall)

    The evaluator outputs both the final score and detailed reasoning explaining each step.
    """
    name: str = "AnswerCorrectness"
    llm: BaseLLM

    _statement_extractor: LLMEvaluator = PrivateAttr()
    _statement_classifier: LLMEvaluator = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self._initialize_evaluators()

    def _initialize_evaluators(self):
        """
        Initialize the LLMEvaluators.
        """
        extract_instr = (
            "Given a question and an answer, analyze each sentence of the answer and "
            "break it down into one or more fully understandable unique statements. "
            "Replace pronouns with explicit references. "
            "Output the candidate statements as a JSON array of strings using double quotes."
        )
        self._statement_extractor = LLMEvaluator(
            instructions=extract_instr.strip(),
            inputs=[{"name": "questions", "type": list[str]}, {"name": "texts", "type": list[str]}],
            outputs=[{"name": "statements", "type": list[str]}],
            examples=[
                {
                    "inputs": {
                        "questions": ["What is the capital of France?"],
                        "texts": [
                            (
                                "The capital of France is Paris. It is known for its rich history, art, "
                                "culture, and landmarks such as the Eiffel Tower."
                            )
                        ],
                    },
                    "outputs": {
                        "statements": [
                            "The capital of France is Paris.",
                            "Paris is known for its rich history, art, culture, "
                            "and landmarks such as the Eiffel Tower.",
                        ]
                    },
                },
                {
                    "inputs": {
                        "questions": ["Who developed the theory of relativity?"],
                        "texts": [
                            (
                                "The theory of relativity was developed by Albert Einstein in the early "
                                "20th century, revolutionizing our understanding of space and time."
                            )
                        ],
                    },
                    "outputs": {
                        "statements": [
                            "The theory of relativity was developed by Albert Einstein in the early 20th century.",
                            "The theory of relativity revolutionized our understanding of space and time.",
                        ]
                    },
                },
            ],
            llm=self.llm,
        )

        classify_instructions = (
            "Given a question, an answer statement, and a reference text, determine if the answer statement "
            "is supported by the reference text. Explain briefly why the statement is or is not supported. "
            "Return a JSON object with keys 'reasoning' (a short explanation) and 'match' (true/false)"
        )
        self._statement_classifier = LLMEvaluator(
            instructions=classify_instructions.strip(),
            inputs=[
                {"name": "question", "type": str},
                {"name": "answer_statement", "type": str},
                {"name": "reference_text", "type": str},
            ],
            outputs=[{"name": "match", "type": bool}, {"name": "reasoning", "type": str}],
            examples=[
                {
                    "inputs": {
                        "question": "What is the capital of France?",
                        "answer_statement": "The capital of France is Paris.",
                        "reference_text": "Paris is the capital of France.",
                    },
                    "outputs": {
                        "reasoning": "The statement exactly matches the core fact in the reference.",
                        "match": True,
                    },
                },
                {
                    "inputs": {
                        "question": "What is the capital of France?",
                        "answer_statement": "Paris is known for its rich history.",
                        "reference_text": "The capital of France is Paris.",
                    },
                    "outputs": {
                        "reasoning": "The statement includes extra details about history "
                        "that are not present in reference.",
                        "match": False,
                    },
                },
                {
                    "inputs": {
                        "question": "Who developed the theory of relativity?",
                        "answer_statement": "The theory was developed by Albert Einstein.",
                        "reference_text": "The theory of relativity was developed by Albert Einstein.",
                    },
                    "outputs": {
                        "reasoning": "The statement conveys the same core fact as the reference "
                        "despite wording differences.",
                        "match": True,
                    },
                },
            ],
            llm=self.llm,
        )

    def _get_unique_candidates(self, candidates: list[str]) -> list[str]:
        """
        Return unique candidate statements preserving order.
        Comparison is done on lowercased, stripped strings.
        """
        seen = set()
        unique = []
        for stmt in candidates:
            norm = stmt.strip().lower()
            if norm not in seen:
                seen.add(norm)
                unique.append(stmt)
        return unique

    def extract_statements(self, questions: list[str], texts: list[str]) -> list[list[str]]:
        """
        Run the extraction evaluator to get candidate statements.
        """
        results = self._statement_extractor.run(questions=questions, texts=texts)
        all_stmts = []
        for res in results["results"]:
            stmts = res.get("statements", [])
            if not isinstance(stmts, list):
                stmts = [stmts]
            all_stmts.append(self._get_unique_candidates(stmts))
        return all_stmts

    def classify_statement(self, question: str, answer_stmt: str, ref_text: str) -> tuple[bool, str]:
        """
        Run the classification evaluator.
        The ref_text is the string of candidate statements from the ground truth answer.
        Returns a tuple (match, explanation).
        """
        data = {"question": [question], "answer_statement": [answer_stmt], "reference_text": [ref_text]}
        result = self._statement_classifier.run(**data)
        m = bool(result["results"][0].get("match", False))
        expl = result["results"][0].get("reasoning", "")
        return m, expl

    def _join_candidates(self, candidates: list[str]) -> str:
        """
        Join candidate statements into a single string. Append punctuation if needed.
        """
        joined = ". ".join(candidates)
        if joined and joined[-1] not in ".!?":
            joined += "."
        return joined

    def _evaluate_candidates(self, question: str, candidates: list[str], ref_text: str) -> list[tuple[str, bool, str]]:
        """
        Classify each candidate statement against the ground truth answer.
        Returns a list of tuples (statement, match, explanation).
        """
        outs = []
        for stmt in candidates:
            m, expl = self.classify_statement(question, stmt, ref_text)
            outs.append((stmt, m, expl))
        return outs

    def _build_reasoning(
        self,
        ans_class: list[tuple[str, bool, str]],
        gt_class: list[tuple[str, bool, str]],
        tp: int,
        fp: int,
        fn: int,
        precision: float,
        recall: float,
        f1: float,
    ) -> str:
        """
        Build a detailed reasoning string.
        This section explains:
        • How the answer was split into statements and compared to the ground truth answer.
        • What each symbol (✅/❌) means.
        • How TP, FP, and FN are computed.
        • How Precision, Recall, and F1 Score are calculated.
        """
        lines = []
        lines.extend(
            [
                "Reasoning:",
                "",
                "Overview:",
                "  The evaluator splits the answer and the ground truth answer into core fact statements.",
                "  Each statement from the answer is compared to the ground truth answer to determine if",
                "  the core fact is supported. Similarly, ground truth statements are checked for their",
                "  presence in the answer. '✅' indicates support/presence, while '❌' indicates lack thereof.",
                "",
                "1. Answer Statements Analysis:",
                "   The answer is split into statements and compared to the ground truth answer.",
                "   '✅' means the statement's core fact is supported; '❌' means it is not.",
                "",
                "Answer Statements Classification:",
            ]
        )

        for stmt, m, expl in ans_class:
            mark = "✅" if m else "❌"
            lines.extend([f" {mark} - {stmt}", f"     Explanation: {expl}", ""])

        lines.extend(
            [
                f" -> TP (supported) = {tp}  (correctly supported statements)",
                f" -> FP (not supported) = {fp}  (unsupported statements)",
            ]
        )

        if (tp + fp) > 0:
            lines.append(f" -> Precision = TP/(TP+FP) = {precision:.2f}")
        else:
            lines.append(" -> Precision = 0.00")

        lines.extend(
            [
                "",
                "2. Ground Truth Statements Analysis:",
                "   The ground truth answer is split into statements and compared to the answer.",
                "   '✅' means the statement is present in the answer; '❌' means it is missing.",
                "",
                "Ground Truth Statements Classification:",
            ]
        )

        for stmt, m, expl in gt_class:
            mark = "✅" if m else "❌"
            lines.extend([f" {mark} - {stmt}", f"     Explanation: {expl}", ""])

        lines.extend(
            [
                f" -> TP (present) = {tp}  (ground truth statements found in answer)",
                f" -> FN (missing) = {fn}  (ground truth statements not found)",
            ]
        )

        if (tp + fn) > 0:
            lines.append(f" -> Recall = TP/(TP+FN) = {recall:.2f}")
        else:
            lines.append(" -> Recall = 0.00")

        lines.extend(
            [
                "",
                "3. Final Metrics:",
                "   F1 Score is the harmonic mean of Precision and Recall:",
                "       F1 Score = 2*(Precision*Recall)/(Precision+Recall)",
                f"       F1 Score = {f1:.2f}",
                "",
                f"Final Score = F1 Score = {round(f1, 2)}",
            ]
        )

        return "\n".join(lines)

    def _evaluate_question(self, question: str, answer_stmts: list[str], gt_stmts: list[str]) -> RunResult:
        """
        Evaluate one question by comparing candidate statements.
        """
        unique_ans = self._get_unique_candidates(answer_stmts)
        unique_gt = self._get_unique_candidates(gt_stmts)
        gt_text = self._join_candidates(unique_gt)
        ans_class = self._evaluate_candidates(question, unique_ans, gt_text)
        ans_text = self._join_candidates(unique_ans)
        gt_class = self._evaluate_candidates(question, unique_gt, ans_text)
        tp = sum(1 for _, m, _ in ans_class if m)
        fp = len(ans_class) - tp
        fn = sum(1 for _, m, _ in gt_class if not m)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        reasoning = self._build_reasoning(ans_class, gt_class, tp, fp, fn, precision, recall, f1)
        return RunResult(score=round(f1, 2), reasoning=reasoning)

    def run_single(self, question: str, answer: str, ground_truth_answer: str, verbose: bool = False) -> RunResult:
        """
        Evaluate answer correctness for a single sample.

        Steps:
          1) Extract candidate statements from both the answer and the ground truth answer.
          2) Compare the candidate statements.
          3) Compute Precision, Recall, and F1 Score.
          4) Generate detailed reasoning.

        Args:
          question (str): The question.
          answer (str): The answer.
          ground_truth_answer (str): The ground truth answer.
          verbose (bool): Flag to output verbose logs.

        Returns:
          RunResult: The evaluation result with score and reasoning.
        """
        # Extract candidate statements for answer and ground truth
        ans_candidates = self.extract_statements([question], [answer])[0]
        gt_candidates = self.extract_statements([question], [ground_truth_answer])[0]
        result = self._evaluate_question(question, ans_candidates, gt_candidates)
        if verbose:
            logger.debug(f"Question: {question}")
            logger.debug(f"Answer: {self._join_candidates(ans_candidates)}")
            logger.debug(f"Ground Truth Answer: {self._join_candidates(gt_candidates)}")
            logger.debug(result.reasoning)
        return result

    def run(
        self, questions: list[str], answers: list[str], ground_truth_answers: list[str], verbose: bool = False
    ) -> RunOutput:
        """
        Evaluate answer correctness:
          1) Extract candidate statements from both the answer and ground truth answer.
          2) For each question, compare the answer statements to the ground truth answer
             and vice versa.
          3) Compute Precision, Recall, and F1 Score.
          4) Generate detailed and easy-to-understand reasoning that explains the metrics.

        Args:
          questions (list[str]): List of questions.
          answers (list[str]): List of answers.
          ground_truth_answers (list[str]): List of ground truth answers.
          verbose (bool): Flag for verbose logging.

        Returns:
          RunOutput: The overall evaluation results.
        """
        run_input = RunInput(
            questions=questions, answers=answers, ground_truth_answers=ground_truth_answers, verbose=verbose
        )
        out_results = []
        for i in range(len(run_input.questions)):
            result = self.run_single(
                question=run_input.questions[i],
                answer=run_input.answers[i],
                ground_truth_answer=run_input.ground_truth_answers[i],
                verbose=run_input.verbose,
            )
            out_results.append(result)
        return RunOutput(results=out_results)
