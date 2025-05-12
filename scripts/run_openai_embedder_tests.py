#!/usr/bin/env python

import argparse
import sys
import uuid

from dynamiq import Workflow, connections
from dynamiq.flows import Flow
from dynamiq.nodes.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from dynamiq.runnables import RunnableStatus
from dynamiq.types import Document


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run OpenAI embedder tests with real API")
    parser.add_argument(
        "--api-key",
        type=str,
        required=True,
        help="OpenAI API key",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="text-embedding-3-small",
        help="OpenAI embedding model to use (default: text-embedding-3-small)",
    )
    return parser


class TestRunner:
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        self.connection = connections.OpenAI(
            id=str(uuid.uuid4()),
            api_key=api_key,
        )
        self.model = model
        self.success_count = 0
        self.failure_count = 0
        self.tests_run = 0

    def print_result(self, test_name: str, passed: bool, error: str | None = None) -> None:
        self.tests_run += 1
        status = "\033[92mPASSED\033[0m" if passed else "\033[91mFAILED\033[0m"
        print(f"Test: {test_name} - {status}")

        if passed:
            self.success_count += 1
        else:
            self.failure_count += 1
            if error:
                print(f"  Error: {error}")
        print()

    def assert_result(self, test_name: str, condition: bool, error_message: str = "Test condition failed") -> None:
        self.print_result(test_name, condition, None if condition else error_message)

    def run_text_embedder_test(self, text: str) -> dict:
        embedder = OpenAITextEmbedder(
            name="OpenAITextEmbedder",
            connection=self.connection,
            model=self.model,
        )
        workflow = Workflow(
            id=str(uuid.uuid4()),
            flow=Flow(nodes=[embedder]),
        )
        input_data = {"query": text}
        return workflow.run(input_data=input_data)

    def run_document_embedder_test(self, documents: list[Document]) -> dict:
        embedder = OpenAIDocumentEmbedder(
            name="OpenAIDocumentEmbedder",
            connection=self.connection,
            model=self.model,
        )
        workflow = Workflow(
            id=str(uuid.uuid4()),
            flow=Flow(nodes=[embedder]),
        )
        input_data = {"documents": documents}
        return workflow.run(input_data=input_data)

    def test_text_embedder_success(self) -> None:
        response = self.run_text_embedder_test("This is a test for embedding.")
        node_id = response.flow.nodes[0].id
        result = response.output[node_id]

        has_embedding = "embedding" in result["output"]
        embedding_is_list = isinstance(result["output"].get("embedding", None), list)
        valid_status = result["status"] == RunnableStatus.SUCCESS.value

        self.assert_result(
            "Text Embedder - Success",
            has_embedding and embedding_is_list and valid_status,
            "Failed to get valid embedding or success status",
        )

    def test_text_embedder_empty_input(self) -> None:
        response = self.run_text_embedder_test("")
        node_id = response.flow.nodes[0].id
        result = response.output[node_id]

        is_failure = result["status"] == RunnableStatus.FAILURE.value

        self.assert_result("Text Embedder - Empty Input", is_failure, "Expected failure with empty input")

    def test_document_embedder_success(self) -> None:
        documents = [Document(content="This is a document for embedding.")]
        response = self.run_document_embedder_test(documents)
        node_id = response.flow.nodes[0].id
        result = response.output[node_id]

        has_documents = "documents" in result["output"]
        has_embedding = False

        if has_documents and result["output"]["documents"]:
            has_embedding = "embedding" in result["output"]["documents"][0]

        valid_status = result["status"] == RunnableStatus.SUCCESS.value

        self.assert_result(
            "Document Embedder - Success",
            has_documents and has_embedding and valid_status,
            "Failed to get valid document embedding or success status",
        )

    def test_document_embedder_empty_document_list(self) -> None:
        response = self.run_document_embedder_test([])
        node_id = response.flow.nodes[0].id
        result = response.output[node_id]

        is_success = result["status"] == RunnableStatus.SUCCESS.value

        self.assert_result(
            "Document Embedder - Empty Document List", is_success, "Expected success for empty document list"
        )

    def test_document_embedder_empty_content(self) -> None:
        documents = [Document(content="")]
        response = self.run_document_embedder_test(documents)
        node_id = response.flow.nodes[0].id
        result = response.output[node_id]

        is_failure = result["status"] == RunnableStatus.FAILURE.value

        self.assert_result(
            "Document Embedder - Empty Content", is_failure, "Expected failure for document with empty content"
        )

    def test_invalid_model(self) -> None:
        original_model = self.model
        self.model = "non-existent-model"

        response = self.run_text_embedder_test("Test with invalid model")
        node_id = response.flow.nodes[0].id
        result = response.output[node_id]

        self.model = original_model

        is_failure = result["status"] == RunnableStatus.FAILURE.value

        self.assert_result("Text Embedder - Invalid Model", is_failure, "Expected failure with invalid model name")

    def run_all_tests(self) -> None:
        print(f"Running OpenAI embedder tests with model: {self.model}\n")

        self.test_text_embedder_success()
        self.test_document_embedder_success()

        self.test_text_embedder_empty_input()
        self.test_document_embedder_empty_document_list()
        self.test_document_embedder_empty_content()
        self.test_invalid_model()

        print(f"\nTests completed: {self.tests_run}")
        print(f"Passed: {self.success_count}")
        print(f"Failed: {self.failure_count}")

        if self.failure_count > 0:
            sys.exit(1)


def main() -> None:
    parser = setup_parser()
    args = parser.parse_args()

    runner = TestRunner(api_key=args.api_key, model=args.model)
    runner.run_all_tests()


if __name__ == "__main__":
    main()
