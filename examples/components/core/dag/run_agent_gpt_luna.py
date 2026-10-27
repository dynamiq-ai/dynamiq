import logging
import os
import time

from dynamiq import Workflow
from dynamiq.connections.managers import get_connection_manager
from dynamiq.serializers.loaders.yaml import WorkflowYAMLLoader

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("luna")
logger.setLevel(logging.INFO)

QUERIES = {
    "simple": (
        "Three people A, B, C each tell the truth or lie. A says 'B lies'. B says 'C lies'. "
        "C says 'A and B both lie'. Who tells the truth? Be brief."
    ),
    "complex": (
        "Five houses in a row each have a different color, owner nationality, drink, pet and cigar brand. "
        "The Brit lives in the red house. The Swede keeps dogs. The Dane drinks tea. The green house is "
        "immediately left of the white house. The green house owner drinks coffee. The Pall Mall smoker keeps "
        "birds. The yellow house owner smokes Dunhill. The man in the center house drinks milk. The Norwegian "
        "lives in the first house. The Blends smoker lives next to the cat owner. The horse owner lives next to "
        "the Dunhill smoker. The Blue Master smoker drinks beer. The German smokes Prince. The Norwegian lives "
        "next to the blue house. The Blends smoker has a neighbor who drinks water. Who owns the fish? "
        "Give the full arrangement briefly."
    ),
}
RUNS = 3

if __name__ == "__main__":
    yaml_file_path = os.path.join(os.path.dirname(__file__), "agent_gpt_luna.yaml")

    with get_connection_manager() as cm:
        wf_data = WorkflowYAMLLoader.load(file_path=yaml_file_path, connection_manager=cm, init_components=True)
        wf = Workflow.from_yaml_file_data(file_data=wf_data, wf_id="luna-reasoning-workflow")

        # Agents run one at a time (not as a parallel flow) so latencies are comparable.
        for agent in wf.flow.nodes:
            llm = agent.llm
            sent = llm.update_completion_params({"max_tokens": llm.max_tokens}).get("reasoning_effort", "<omitted>")
            for label, query in QUERIES.items():
                latencies = []
                for _ in range(RUNS):
                    start = time.perf_counter()
                    result = agent.run(input_data={"input": query})
                    latencies.append(time.perf_counter() - start)
                logger.info(
                    "%-14s %-7s configured=%-6s sent=%-6s status=%s avg=%.2fs runs=%s answer=%s",
                    agent.name,
                    label,
                    getattr(llm.reasoning_effort, "value", llm.reasoning_effort),
                    getattr(sent, "value", sent),
                    result.status.value,
                    sum(latencies) / len(latencies),
                    [round(t, 2) for t in latencies],
                    str(result.output.get("answer"))[:60].replace("\n", " "),
                )
