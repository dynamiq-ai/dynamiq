import logging
import os

import typer
from datasets import Dataset
from dotenv import find_dotenv, load_dotenv
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

from dynamiq import ROOT_PATH, Workflow, runnables
from dynamiq.connections.managers import get_connection_manager
from dynamiq.serializers.loaders.yaml import WorkflowYAMLLoader

logger = logging.getLogger(__name__)

load_dotenv(find_dotenv())

app = typer.Typer()


@app.command()
def main(
    question: str = "How to build an advanced RAG pipeline?", ground_truth: str = ""
):
    with get_connection_manager() as cm:
        dag_yaml_file_path = os.path.join(os.path.dirname(ROOT_PATH), "examples/rag/dag_pinecone.yaml")
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

        answer = wf_result.output.get("openai-1").get("output").get("answer")
        documents = (
            wf_result.output.get("document-retriever-node-1")
            .get("output")
            .get("documents")
        )
        context = [doc["content"] for doc in documents]

        eval_dataset = Dataset.from_dict(
            {
                "question": [question],
                "ground_truth": [ground_truth],
                "answer": [answer],
                "contexts": [context],
            }
        )

        eval_metrics = evaluate(
            dataset=eval_dataset,
            metrics=[
                context_precision,
                faithfulness,
                answer_relevancy,
                context_recall,
            ],
        )

        logger.info(f"Retrival output (Answer):\n {answer}")
        logger.info(f"RAG Evaluation Result:\n {eval_metrics}")


if __name__ == "__main__":
    app()
