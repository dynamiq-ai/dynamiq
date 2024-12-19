import logging

import typer
from dotenv import find_dotenv, load_dotenv

from dynamiq import Workflow, runnables
from dynamiq.connections.managers import get_connection_manager

# Import Dynamiq evaluators
from dynamiq.evaluations.metrics import ContextRecallEvaluator, FaithfulnessEvaluator
from dynamiq.nodes.llms import OpenAI
from dynamiq.serializers.loaders.yaml import WorkflowYAMLLoader

logger = logging.getLogger(__name__)

load_dotenv(find_dotenv())

app = typer.Typer()


@app.command()
def main(
    question: str = "How to build an advanced RAG pipeline?",
    dag_yaml_file_path: str = "examples/rag/dag_pinecone.yaml",
):
    with get_connection_manager() as cm:
        wf_data = WorkflowYAMLLoader.load(
            file_path=dag_yaml_file_path,
            connection_manager=cm,
            init_components=True,
        )

        retrieval_wf = Workflow.from_yaml_file_data(
            file_data=wf_data, wf_id="retrieval-workflow"
        )
        wf_result = retrieval_wf.run(
            input_data={"query": question},
            config=runnables.RunnableConfig(callbacks=[]),
        )

        # Extracting the answer and documents
        answer = wf_result.output.get("openai-1").get("output").get("answer")
        documents = (
            wf_result.output.get("document-retriever-node-1")
            .get("output")
            .get("documents")
        )
        context_list = [doc["content"] for doc in documents]
        context = " ".join(context_list)

        # Initialize the LLM (replace 'gpt-4o-mini' with your available model)
        llm = OpenAI(model="gpt-4o-mini")

        # Initialize Dynamiq Evaluators
        context_recall_evaluator = ContextRecallEvaluator(llm=llm)
        faithfulness_evaluator = FaithfulnessEvaluator(llm=llm)

        # Evaluate Context Recall
        recall_scores = context_recall_evaluator.run(
            question=[question],
            answer=[answer],
            context=[context],
        )

        # Evaluate Faithfulness
        faithfulness_scores = faithfulness_evaluator.run(
            question=[question],
            answer=[answer],
            context=[context],
        )

        # Aggregate Evaluation Metrics
        eval_metrics = {
            "context_recall": recall_scores[0],
            "faithfulness": faithfulness_scores[0],
        }

        # Log the results
        logger.info(f"Retrieval output (Answer):\n {answer}")
        logger.info(f"Dynamiq Evaluation Result:\n {eval_metrics}")


if __name__ == "__main__":
    app()
