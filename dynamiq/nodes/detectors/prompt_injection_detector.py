from concurrent.futures import ThreadPoolExecutor
from typing import Any, ClassVar, Literal
from urllib.parse import urljoin

import requests
from pydantic import BaseModel, ConfigDict, Field

from dynamiq.connections import HuggingFace, Lakera
from dynamiq.nodes.node import ConnectionNode, ErrorHandling, ensure_config
from dynamiq.nodes.types import NodeGroup
from dynamiq.runnables import RunnableConfig

HUGGINGFACE_URL = "https://router.huggingface.co/hf-inference/models/"

# Labels of the prompt-injection classifiers on the Hugging Face Hub: protectai (SAFE, INJECTION), deepset (LEGIT,
# INJECTION), Prompt Guard v1 (BENIGN, INJECTION, JAILBREAK), Prompt Guard 2 (BENIGN, MALICIOUS) and configs that
# keep the generic names (LABEL_0, LABEL_1).
BENIGN_LABELS = frozenset({"SAFE", "LEGIT", "BENIGN", "LABEL_0"})
ATTACK_LABELS = frozenset({"INJECTION", "JAILBREAK", "MALICIOUS", "LABEL_1"})

# These classifiers read at most 512 tokens and their servers drop the rest silently, so longer text is scored in
# overlapping windows. Their DeBERTa-v3 tokenizers make at most one token per character, so a window of 500 is read
# whole in any script, and the overlap holds an injection that the edge of a window cuts.
WINDOW_CHARS = 500
WINDOW_OVERLAP_CHARS = 100
MAX_PARALLEL_REQUESTS = 4
ATTACK_THRESHOLD = 0.5

# Lakera detectors whose flags are no attack: personal data is the job of the pii hook, which masks it, and a link
# alone is not an injection. Both are in Lakera's default policy.
LAKERA_NON_ATTACK_DETECTORS = ("pii/", "unknown_links")


class PromptInjDetectorInputSchema(BaseModel):
    message: str = Field(..., description="Parameter to provide value to validate.")


class PromptInjectionDetector(ConnectionNode):
    group: Literal[NodeGroup.DETECTORS] = NodeGroup.DETECTORS
    name: str = "prompt-injection-detector"
    description: str = "Node that detects prompt injection."
    connection: HuggingFace | Lakera | None = None
    model: str = "protectai/deberta-v3-base-prompt-injection-v2"
    timeout: float = 30
    error_handling: ErrorHandling = Field(default_factory=lambda: ErrorHandling(max_retries=1))

    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_schema: ClassVar[type[PromptInjDetectorInputSchema]] = PromptInjDetectorInputSchema

    def __init__(self, **kwargs):
        """Initialize the PII Detector.

        If neither client nor connection is provided in kwargs, a new HuggingFace connection is created.

        Args:
            **kwargs: Keyword arguments to initialize the node.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = HuggingFace()
        super().__init__(**kwargs)

    def execute(
        self, input_data: PromptInjDetectorInputSchema, config: RunnableConfig = None, **kwargs
    ) -> dict[str, Any]:
        """
        Detects prompt injection in input data

        Args:
            input_data (PromptInjDetectorInputSchema): input data for the detector, contains message to validate.
            config (RunnableConfig, optional): Configuration for the runnable, including callbacks.
            **kwargs: Additional arguments passed to the execution context.

        Returns:
            dict[str, Any]: A dictionary containing detection results.
        """
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        message = input_data.message
        try:
            if isinstance(self.connection, HuggingFace):
                result = self.huggingface_detect(message=message)
            elif isinstance(self.connection, Lakera):
                result = self.lakera_detect(message=message)
            else:
                raise ValueError(f"Unsupported provider: {type(self.connection)}")
            return result
        except Exception as e:
            msg = f"Encountered an error while performing validation. \nError details: {e}"
            raise ValueError(msg)

    def huggingface_detect(self, message: str) -> dict[str, Any]:
        """
        Detects prompt injection attempts in a given message using a Hugging Face text-classification model,
        served by the Hugging Face router or by the connection's own endpoint.

        Args:
            message (str): The input message to analyze for prompt injection.

        Returns:
            dict[str, Any]: ``prompt_detected`` and ``score``, the highest attack probability of any window.
        """
        api_url = self.connection.url or urljoin(HUGGINGFACE_URL, self.model)
        headers = {"Authorization": f"Bearer {self.connection.api_key}"}
        windows = split_windows(message, WINDOW_CHARS, WINDOW_OVERLAP_CHARS)

        def score(window: str) -> float:
            response = requests.post(api_url, headers=headers, json={"inputs": window}, timeout=self.timeout)
            if response.status_code in [400, 401]:
                raise ValueError("Request failed. Possible causes: bad request or invalid API key.")
            if response.status_code != 200:
                raise ValueError(f"Request failed with status code {response.status_code}")
            return attack_probability(response.json())

        with ThreadPoolExecutor(max_workers=min(len(windows), MAX_PARALLEL_REQUESTS)) as pool:
            highest = max(pool.map(score, windows))
        return {"prompt_detected": highest > ATTACK_THRESHOLD, "score": highest}

    def lakera_detect(self, message: str) -> dict[str, Any]:
        """
        Detects prompt injection attempts in a given message using a Lakera endpoint.

        Lakera flags a request when any detector of the project's policy fires. A flag that only the detectors of
        ``LAKERA_NON_ATTACK_DETECTORS`` raised does not count; any other, such as content moderation or a custom
        detector, does, so a policy still blocks what it was set up to block. A project in Lakera's Detect mode
        never flags, so it never blocks here either.

        Args:
            message (str): The input message to analyze for prompt injection.
        """
        payload = {"messages": [{"content": message, "role": "user"}], "breakdown": True}
        if self.connection.project_id:
            payload["project_id"] = self.connection.project_id
        response = self.client.post(
            self.connection.guard_url, headers=self.connection.conn_params, json=payload, timeout=self.timeout
        )
        if response.status_code == 401:
            raise ValueError("Invalid API key")
        if response.status_code != 200:
            raise ValueError(f"Request failed with status code {response.status_code}")

        result = response.json()
        if result.get("flagged") is None:
            raise ValueError("Lakera returned no `flagged` verdict")
        breakdown = result.get("breakdown")
        attack = breakdown is None or any(
            item.get("detected") and not str(item.get("detector_type", "")).startswith(LAKERA_NON_ATTACK_DETECTORS)
            for item in breakdown
        )
        return {"prompt_detected": bool(result["flagged"]) and attack}


def split_windows(text: str, size: int, overlap: int) -> list[str]:
    """``text`` in windows of at most ``size`` characters, each starting ``overlap`` characters before the previous one
    ends."""
    if len(text) <= size:
        return [text]
    return [text[start : start + size] for start in range(0, len(text) - overlap, size - overlap)]


def attack_probability(predictions: list) -> float:
    """The probability that a classified text is an attack, from a text-classification response.

    The response lists ``{label, score}`` for every class or only for the top one, nested in a list per input by the
    legacy Inference API.

    Raises:
        ValueError: The response is no list of labels, holds no label, or holds a label that is not one of
            ``BENIGN_LABELS`` or ``ATTACK_LABELS``, so the model is not a prompt-injection classifier this node can
            read.
    """
    if not isinstance(predictions, list):
        raise ValueError(f"The classifier returned no list of labels: {predictions}")
    if predictions and isinstance(predictions[0], list):
        predictions = predictions[0]
    scores = {str(item["label"]).upper(): float(item["score"]) for item in predictions}
    if not scores:
        raise ValueError("The classifier returned no label.")
    if unknown := scores.keys() - BENIGN_LABELS - ATTACK_LABELS:
        raise ValueError(f"Unrecognized classifier labels {sorted(unknown)}; not a prompt-injection classifier.")
    attack = [score for label, score in scores.items() if label in ATTACK_LABELS]
    if attack:
        return sum(attack)
    return 1 - sum(scores.values())
