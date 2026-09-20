import logging
import os

from dynamiq import Workflow
from dynamiq.runnables import RunnableConfig

logger = logging.getLogger(__name__)

APPLICATIONS = [
    {"fico": 760, "ltv": 75, "program": "Conventional", "base_rate": 6.5},
    {"fico": 700, "ltv": 85, "program": "FHA", "base_rate": 6.5},
    {"fico": 600, "ltv": 90, "program": "FHA", "base_rate": 6.5},
    {"fico": 550, "ltv": 80, "program": "VA", "base_rate": 6.5},
]


def run():
    yaml_path = os.path.join(os.path.dirname(__file__), "decision_workflow.yaml")
    workflow = Workflow.from_yaml_file(file_path=yaml_path, init_components=True)
    result = workflow.run(input_data={"applications": APPLICATIONS}, config=RunnableConfig(callbacks=[]))
    for application, priced in zip(APPLICATIONS, result.output["batch_end"]["output"]["priced"]):
        rules = ", ".join(rule["name"] for rule in priced["matched_rules"]) or "none"
        logger.info(
            f"fico={application['fico']} ltv={application['ltv']} {application['program']}: "
            f"{priced['decision']} at {priced['rate']}% (tier {priced['tier']}, adjustments: {rules})"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
