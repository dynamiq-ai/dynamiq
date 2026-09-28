import logging
import os

from dynamiq import Workflow
from dynamiq.runnables import RunnableConfig
from dynamiq.types.mocking import MockConfig

logger = logging.getLogger(__name__)

TICKETS = [
    "My card was charged twice for the September invoice and I need this fixed today.",
    "The webhook integration stopped delivering events after your release last night.",
    "Do you offer a discount for annual plans?",
]

# What the judgement returns when there is no TypeSafe key: one fixed verdict, so the routing still runs.
MOCK_VERDICT = {
    "content": "Judgement by mock: is_urgent yes, team billing, anger frustrated",
    "answers": {
        "is_urgent": {"type": "noul", "probability": 0.92, "decision": True, "confidence": 0.84},
        "team": {
            "type": "choice",
            "choice": "billing",
            "probabilities": {"billing": 0.84, "technical": 0.15, "sales": 0.01},
            "confidence": 0.76,
        },
        "anger": {
            "type": "score",
            "level": "frustrated",
            "index": 1,
            "score": 1.05,
            "probabilities": {"calm": 0.1, "frustrated": 0.75, "furious": 0.15},
            "confidence": 0.63,
        },
    },
    "decisions": {"is_urgent": True, "team": "billing", "anger": "frustrated"},
    "confidence": 0.63,
    "needs_review": False,
    "low_confidence": [],
    "model": "mock",
    "backend": "system_one",
    "confidence_source": "model",
    "usage": None,
    "rationale": None,
    "evidence": None,
}


def run():
    has_key = bool(os.getenv("TYPESAFE_API_KEY"))
    if not has_key:
        # The connection reads its key from the environment; the node is mocked below and never calls the API.
        os.environ["TYPESAFE_API_KEY"] = "mock"
    yaml_path = os.path.join(os.path.dirname(__file__), "judgement_workflow.yaml")
    workflow = Workflow.from_yaml_file(file_path=yaml_path, init_components=True)
    if not has_key:
        logger.warning("TYPESAFE_API_KEY is not set: the judgement is mocked and every ticket gets the same verdict")
        triage = next(node for node in workflow.flow.nodes if node.name == "triage")
        triage.mock = MockConfig(enabled=True, output=MOCK_VERDICT)

    for ticket in TICKETS:
        result = workflow.run(input_data={"ticket": ticket}, config=RunnableConfig(callbacks=[]))
        output = result.output["end"]["output"]
        decisions = {
            name: answer.get("decision", answer.get("choice", answer.get("level")))
            for name, answer in output["answers"].items()
        }
        logger.info(f"{ticket[:50]!r}: queue={output['queue']} priority={output['priority']} {decisions}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
