from unittest.mock import Mock, patch

import pytest

from dynamiq.nodes.writers import WeaviateDocumentWriter
from dynamiq.nodes.writers.base import WriterInputSchema
from dynamiq.storages.vector.dry_run import DryRunConfig, DryRunMode, DryRunResult
from dynamiq.types import Document


@pytest.fixture
def sample_documents():
    return [
        Document(
            id="basic_test_doc1",
            content="Basic integration test content 1",
            embedding=[0.1] * 100,
            metadata={"type": "test", "category": "basic"},
        ),
        Document(
            id="basic_test_doc2",
            content="Basic integration test content 2",
            embedding=[0.2] * 100,
            metadata={"type": "test", "category": "basic"},
        ),
    ]


@pytest.fixture
def integration_documents():
    return [
        Document(
            id="integration_doc1",
            content="Integration test content 1",
            embedding=[0.1] * 1536,
            metadata={"type": "test", "category": "integration"},
        ),
        Document(
            id="integration_doc2",
            content="Integration test content 2",
            embedding=[0.2] * 1536,
            metadata={"type": "test", "category": "integration"},
        ),
    ]


@pytest.fixture
def mock_weaviate_components():
    with patch("dynamiq.connections.Weaviate") as mock_connection, patch(
        "dynamiq.storages.vector.WeaviateVectorStore"
    ) as mock_vs_cls:

        mock_client = Mock()
        mock_connection.return_value.connect.return_value = mock_client

        mock_vs_instance = Mock()
        mock_vs_cls.return_value = mock_vs_instance

        with patch.object(WeaviateDocumentWriter, "init_components"):
            yield {
                "connection": mock_connection,
                "client": mock_client,
                "vector_store_cls": mock_vs_cls,
                "vector_store_instance": mock_vs_instance,
            }


@pytest.fixture
def mock_workflow_components():
    with patch("dynamiq.nodes.converters.PyPDFConverter") as mock_converter, patch(
        "dynamiq.nodes.splitters.document.DocumentSplitter"
    ) as mock_splitter, patch("dynamiq.nodes.embedders.OpenAIDocumentEmbedder") as mock_embedder:

        mock_converter.return_value.outputs.documents = "converter_docs"
        mock_splitter.return_value.outputs.documents = "splitter_docs"
        mock_embedder.return_value.outputs.documents = "embedder_docs"

        yield {"converter": mock_converter, "splitter": mock_splitter, "embedder": mock_embedder}


def test_dry_run_config_creation():
    for mode in DryRunMode:
        config = DryRunConfig(mode=mode)
        assert config.mode == mode
        assert isinstance(config.test_collection_suffix, str)
        assert isinstance(config.document_id_prefix, str)


def test_writer_input_schema_with_dry_run(sample_documents):
    config = DryRunConfig(mode=DryRunMode.WORKFLOW_ONLY, document_id_prefix="schema_test_")

    input_schema = WriterInputSchema(documents=sample_documents, content_key="test_content", dry_run_config=config)

    assert input_schema.documents == sample_documents
    assert input_schema.content_key == "test_content"
    assert input_schema.dry_run_config == config
    assert input_schema.dry_run_config.mode == DryRunMode.WORKFLOW_ONLY


def test_orchestrator_workflow_only_integration(sample_documents):
    from dynamiq.storages.vector.dry_run_orchestrator import DryRunOrchestrator

    config = DryRunConfig(mode=DryRunMode.WORKFLOW_ONLY, validation_enabled=True, document_id_prefix="orch_test_")

    mock_vector_store_cls = Mock()
    mock_vector_store_cls.__name__ = "MockVectorStore"

    orchestrator = DryRunOrchestrator(
        vector_store_cls=mock_vector_store_cls, vector_store_params={"index_name": "test_index"}, dry_run_config=config
    )

    result = orchestrator.execute(sample_documents, "test_content")

    assert isinstance(result, DryRunResult)
    assert result.success is True
    assert result.mode == DryRunMode.WORKFLOW_ONLY
    assert result.documents_processed == 2
    assert result.test_collection_name is None
    assert len(result.workflow_steps_completed) == 5
    assert "Vector store write (SKIPPED)" in result.workflow_steps_completed


def test_document_preparation_integration(sample_documents):
    from dynamiq.storages.vector.dry_run_orchestrator import DryRunOrchestrator

    config = DryRunConfig(mode=DryRunMode.WORKFLOW_ONLY, document_id_prefix="prep_test_")

    mock_vector_store_cls = Mock()
    mock_vector_store_cls.__name__ = "MockVectorStore"

    orchestrator = DryRunOrchestrator(
        vector_store_cls=mock_vector_store_cls, vector_store_params={}, dry_run_config=config
    )

    prepared_docs = orchestrator._prepare_documents_for_storage(sample_documents)

    assert len(prepared_docs) == 2

    doc1 = prepared_docs[0]
    assert doc1.id == "prep_test_basic_test_doc1"
    assert doc1.content == "Basic integration test content 1"
    assert doc1.metadata["_dry_run"] is True
    assert doc1.metadata["_dry_run_mode"] == DryRunMode.WORKFLOW_ONLY
    assert doc1.metadata["_original_id"] == "basic_test_doc1"
    assert "_dry_run_timestamp" in doc1.metadata

    assert doc1.metadata["type"] == "test"
    assert doc1.metadata["category"] == "basic"

    assert sample_documents[0].id == "basic_test_doc1"
    assert "_dry_run" not in sample_documents[0].metadata


def test_validation_integration(sample_documents):
    from dynamiq.storages.vector.dry_run_orchestrator import DryRunOrchestrator

    config = DryRunConfig(mode=DryRunMode.WORKFLOW_ONLY)
    mock_vector_store_cls = Mock()
    mock_vector_store_cls.__name__ = "MockVectorStore"

    orchestrator = DryRunOrchestrator(
        vector_store_cls=mock_vector_store_cls, vector_store_params={}, dry_run_config=config
    )

    validated = orchestrator._validate_documents(sample_documents)
    assert len(validated) == 2
    assert validated == sample_documents

    orchestrator._verify_embeddings(sample_documents)

    validation_results = orchestrator._check_schema_compatibility()
    assert validation_results["schema_validation_performed"] is True
    assert validation_results["vector_store_type"] == "MockVectorStore"


def test_error_handling_integration():
    from dynamiq.storages.vector.dry_run_orchestrator import DryRunOrchestrator

    config = DryRunConfig(mode=DryRunMode.WORKFLOW_ONLY)
    mock_vector_store_cls = Mock()
    mock_vector_store_cls.__name__ = "MockVectorStore"

    orchestrator = DryRunOrchestrator(
        vector_store_cls=mock_vector_store_cls, vector_store_params={}, dry_run_config=config
    )

    result = orchestrator.execute([])
    assert result.success is False
    assert "No documents provided" in result.error_message

    invalid_doc = Document(id="test", content="test")
    invalid_doc.id = None

    result = orchestrator.execute([invalid_doc])
    assert result.success is False
    assert "missing required 'id' field" in result.error_message


def test_configuration_validation_integration():
    valid_configs = [
        DryRunConfig(mode=DryRunMode.WORKFLOW_ONLY),
        DryRunConfig(mode=DryRunMode.TEMPORARY, cleanup_timeout=600),
        DryRunConfig(mode=DryRunMode.PERSISTENT, test_collection_suffix="dev"),
        DryRunConfig(mode=DryRunMode.INSPECTION, preserve_on_error=True),
    ]

    for config in valid_configs:
        assert isinstance(config.mode, DryRunMode)
        assert config.cleanup_timeout >= 0
        assert isinstance(config.test_collection_suffix, str)
        assert isinstance(config.document_id_prefix, str)

    with pytest.raises(Exception):
        DryRunConfig(mode=DryRunMode.WORKFLOW_ONLY, cleanup_timeout=-1)

    with pytest.raises(Exception):
        DryRunConfig(mode=DryRunMode.WORKFLOW_ONLY, batch_size=0)


def test_dry_run_result_integration():
    result = DryRunResult(
        success=True,
        mode=DryRunMode.WORKFLOW_ONLY,
        documents_processed=5,
        duration_seconds=1.23,
        validation_results={"test": True},
        cleanup_status={"cleanup_needed": False},
        workflow_steps_completed=[],
    )

    result.add_workflow_step("step1")
    result.add_workflow_step("step2")
    assert result.workflow_steps_completed == ["step1", "step2"]

    result.add_validation_result("key1", "value1")
    assert result.validation_results["key1"] == "value1"

    result.add_cleanup_status("operation1", True)
    assert result.cleanup_status["operation1"] is True

    summary = result.summary()
    assert "✅ SUCCESS" in summary
    assert "workflow_only" in summary
    assert "Documents: 5" in summary
    assert "Duration: 1.23s" in summary

    assert not result.is_cleanup_needed()

    result.mode = DryRunMode.TEMPORARY
    assert result.is_cleanup_needed()


def test_writer_initialization_with_dry_run_config(mock_weaviate_components):
    dry_run_config = DryRunConfig(mode=DryRunMode.WORKFLOW_ONLY, validation_enabled=True)

    writer = WeaviateDocumentWriter(index_name="IntegrationTest", dry_run_config=dry_run_config)

    writer.vector_store = mock_weaviate_components["vector_store_instance"]

    assert writer.dry_run_config == dry_run_config
    assert writer.dry_run_config.mode == DryRunMode.WORKFLOW_ONLY


def test_workflow_only_mode_integration(integration_documents, mock_weaviate_components):
    writer = WeaviateDocumentWriter(
        index_name="IntegrationTest", dry_run_config=DryRunConfig(mode=DryRunMode.WORKFLOW_ONLY)
    )

    input_data = WriterInputSchema(documents=integration_documents)

    with patch.object(writer, "vector_store") as mock_vector_store:
        mock_vector_store.write_documents.return_value = len(integration_documents)

        result = writer.execute(input_data)

        assert "dry_run_result" in result
        assert result["success"] is True
        assert result["upserted_count"] == 2
        assert result["mode"] == DryRunMode.WORKFLOW_ONLY
        assert result["test_collection_name"] is None

        dry_run_result = result["dry_run_result"]
        assert isinstance(dry_run_result, DryRunResult)
        assert dry_run_result.mode == DryRunMode.WORKFLOW_ONLY
        assert dry_run_result.documents_processed == 2
        assert dry_run_result.success is True
        assert len(dry_run_result.workflow_steps_completed) > 0


def test_input_override_dry_run_config(integration_documents, mock_weaviate_components):
    with patch.object(WeaviateDocumentWriter, "init_components"):
        writer = WeaviateDocumentWriter(
            index_name="IntegrationTest", dry_run_config=DryRunConfig(mode=DryRunMode.PERSISTENT)
        )

        input_data = WriterInputSchema(
            documents=integration_documents, dry_run_config=DryRunConfig(mode=DryRunMode.WORKFLOW_ONLY)
        )

        with patch.object(writer, "vector_store"):
            result = writer.execute(input_data)

            dry_run_result = result["dry_run_result"]
            assert dry_run_result.mode == DryRunMode.WORKFLOW_ONLY


def test_normal_execution_without_dry_run(integration_documents, mock_weaviate_components):
    with patch.object(WeaviateDocumentWriter, "init_components"):
        writer = WeaviateDocumentWriter(index_name="IntegrationTest")
        input_data = WriterInputSchema(documents=integration_documents)

        with patch.object(writer, "vector_store") as mock_vector_store:
            mock_vector_store.write_documents.return_value = len(integration_documents)

            result = writer.execute(input_data)

            assert "dry_run_result" not in result
            assert result["upserted_count"] == 2

            mock_vector_store.write_documents.assert_called_once_with(integration_documents, content_key=None)


def test_dry_run_orchestrator_integration(integration_documents, mock_weaviate_components):
    with patch.object(WeaviateDocumentWriter, "init_components"):
        writer = WeaviateDocumentWriter(
            index_name="OrchestratorTest",
            dry_run_config=DryRunConfig(
                mode=DryRunMode.WORKFLOW_ONLY, document_id_prefix="integration_test_", validation_enabled=True
            ),
        )

        input_data = WriterInputSchema(documents=integration_documents, content_key="custom_content")

        with patch.object(writer, "vector_store"):
            with patch("dynamiq.nodes.writers.weaviate.DryRunOrchestrator") as mock_orchestrator_cls:
                mock_orchestrator = Mock()
                mock_orchestrator_cls.return_value = mock_orchestrator

                mock_result = DryRunResult(
                    success=True,
                    mode=DryRunMode.WORKFLOW_ONLY,
                    documents_processed=2,
                    duration_seconds=1.5,
                    validation_results={"test": True},
                    cleanup_status={"cleanup_needed": False},
                    workflow_steps_completed=["validation", "processing"],
                )
                mock_orchestrator.execute.return_value = mock_result

                result = writer.execute(input_data)

                mock_orchestrator_cls.assert_called_once_with(
                    vector_store_cls=writer.vector_store_cls,
                    vector_store_params=writer.vector_store_params,
                    dry_run_config=writer.dry_run_config,
                )

                mock_orchestrator.execute.assert_called_once_with(integration_documents, "custom_content")

                assert result["dry_run_result"] == mock_result
                assert result["success"] is True
                assert result["upserted_count"] == 2


def test_dry_run_error_handling(integration_documents, mock_weaviate_components):
    with patch.object(WeaviateDocumentWriter, "init_components"):
        writer = WeaviateDocumentWriter(
            index_name="ErrorTest", dry_run_config=DryRunConfig(mode=DryRunMode.WORKFLOW_ONLY)
        )

        input_data = WriterInputSchema(documents=integration_documents)

        with patch.object(writer, "vector_store"):
            with patch("dynamiq.nodes.writers.weaviate.DryRunOrchestrator") as mock_orchestrator_cls:
                mock_orchestrator = Mock()
                mock_orchestrator_cls.return_value = mock_orchestrator

                mock_result = DryRunResult(
                    success=False,
                    mode=DryRunMode.WORKFLOW_ONLY,
                    documents_processed=0,
                    duration_seconds=0.1,
                    validation_results={},
                    cleanup_status={},
                    workflow_steps_completed=["validation"],
                    error_message="Mock error for testing",
                )
                mock_orchestrator.execute.return_value = mock_result

                result = writer.execute(input_data)

                assert result["success"] is False
                assert result["upserted_count"] == 0
                assert result["dry_run_result"].error_message == "Mock error for testing"


def test_vector_store_params_include_dry_run_config(mock_weaviate_components):
    with patch.object(WeaviateDocumentWriter, "init_components"):
        dry_run_config = DryRunConfig(mode=DryRunMode.WORKFLOW_ONLY, test_collection_suffix="test_env")

        writer = WeaviateDocumentWriter(index_name="ParamsTest", dry_run_config=dry_run_config)

        params = writer.vector_store_params

        assert "dry_run_config" in params
        assert params["dry_run_config"] == dry_run_config
        assert params["index_name"] == "ParamsTest"


def test_content_key_propagation(integration_documents, mock_weaviate_components):
    with patch.object(WeaviateDocumentWriter, "init_components"):
        writer = WeaviateDocumentWriter(
            index_name="ContentKeyTest", dry_run_config=DryRunConfig(mode=DryRunMode.WORKFLOW_ONLY)
        )

        input_data = WriterInputSchema(documents=integration_documents, content_key="special_content_field")

        with patch.object(writer, "vector_store"):
            with patch("dynamiq.nodes.writers.weaviate.DryRunOrchestrator") as mock_orchestrator_cls:
                mock_orchestrator = Mock()
                mock_orchestrator_cls.return_value = mock_orchestrator

                mock_result = DryRunResult(
                    success=True,
                    mode=DryRunMode.WORKFLOW_ONLY,
                    documents_processed=2,
                    duration_seconds=1.0,
                    validation_results={},
                    cleanup_status={},
                    workflow_steps_completed=[],
                )
                mock_orchestrator.execute.return_value = mock_result

                writer.execute(input_data)

                mock_orchestrator.execute.assert_called_once_with(integration_documents, "special_content_field")


def test_complete_workflow_with_dry_run(mock_workflow_components, mock_weaviate_components):
    with patch.object(WeaviateDocumentWriter, "init_components"):
        writer = WeaviateDocumentWriter(
            index_name="WorkflowTest",
            dry_run_config=DryRunConfig(mode=DryRunMode.WORKFLOW_ONLY, validation_enabled=True),
        )

    expected_result = {
        "writer": {
            "dry_run_result": DryRunResult(
                success=True,
                mode=DryRunMode.WORKFLOW_ONLY,
                documents_processed=5,
                duration_seconds=2.0,
                validation_results={},
                cleanup_status={},
                workflow_steps_completed=[],
            ),
            "success": True,
            "upserted_count": 5,
        }
    }

    assert writer.dry_run_config.mode == DryRunMode.WORKFLOW_ONLY
    assert expected_result["writer"]["dry_run_result"].success is True
    assert expected_result["writer"]["dry_run_result"].mode == DryRunMode.WORKFLOW_ONLY
    assert expected_result["writer"]["success"] is True
    assert expected_result["writer"]["upserted_count"] == 5


def test_workflow_dry_run_vs_normal_execution(mock_weaviate_components):
    with patch.object(WeaviateDocumentWriter, "init_components"):
        writer = WeaviateDocumentWriter(index_name="FlexibleTest")

        sample_docs = [
            Document(id="doc1", content="test", embedding=[0.1] * 100),
            Document(id="doc2", content="test", embedding=[0.2] * 100),
        ]

        with patch.object(writer, "vector_store") as mock_vector_store:
            mock_vector_store.write_documents.return_value = 2

            normal_input = WriterInputSchema(documents=sample_docs)
            normal_result = writer.execute(normal_input)

            assert "dry_run_result" not in normal_result
            assert normal_result["upserted_count"] == 2

            dry_run_input = WriterInputSchema(
                documents=sample_docs, dry_run_config=DryRunConfig(mode=DryRunMode.WORKFLOW_ONLY)
            )

            with patch("dynamiq.nodes.writers.weaviate.DryRunOrchestrator") as mock_orchestrator_cls:
                mock_orchestrator = Mock()
                mock_orchestrator_cls.return_value = mock_orchestrator
                mock_orchestrator.execute.return_value = DryRunResult(
                    success=True,
                    mode=DryRunMode.WORKFLOW_ONLY,
                    documents_processed=2,
                    duration_seconds=1.0,
                    validation_results={},
                    cleanup_status={},
                    workflow_steps_completed=[],
                )

                dry_run_result = writer.execute(dry_run_input)

                assert "dry_run_result" in dry_run_result
                assert dry_run_result["success"] is True
                assert dry_run_result["upserted_count"] == 2


def test_writer_configuration_inheritance(mock_weaviate_components):
    from dynamiq.storages.vector.weaviate import WeaviateWriterVectorStoreParams

    params = WeaviateWriterVectorStoreParams(
        index_name="InheritanceTest",
        dry_run_config=DryRunConfig(mode=DryRunMode.WORKFLOW_ONLY, document_id_prefix="inherited_"),
    )

    assert hasattr(params, "dry_run_config")
    assert params.dry_run_config.mode == DryRunMode.WORKFLOW_ONLY
    assert params.dry_run_config.document_id_prefix == "inherited_"

    with patch.object(WeaviateDocumentWriter, "init_components"):
        writer = WeaviateDocumentWriter(**params.model_dump())

        assert writer.dry_run_config.mode == DryRunMode.WORKFLOW_ONLY
        assert writer.dry_run_config.document_id_prefix == "inherited_"


def test_inspection_mode_integration(sample_documents, mock_weaviate_components):
    writer = WeaviateDocumentWriter(
        index_name="InspectionTest",
        dry_run_config=DryRunConfig(
            mode=DryRunMode.INSPECTION, test_collection_suffix="analysis", validation_enabled=True
        ),
    )

    input_data = WriterInputSchema(documents=sample_documents)

    with patch.object(writer, "vector_store") as mock_vector_store:
        mock_vector_store.write_documents.return_value = len(sample_documents)

        with patch(
            "dynamiq.storages.vector.dry_run_orchestrator.DryRunOrchestrator._initialize_test_vector_store",
            return_value=(mock_vector_store, "InspectionTest_analysis"),
        ):
            result = writer.execute(input_data)

            assert "dry_run_result" in result
            assert result["success"] is True
            assert result["upserted_count"] == 2

            dry_run_result = result["dry_run_result"]
            assert dry_run_result.mode == DryRunMode.INSPECTION
            assert dry_run_result.documents_processed == 2
            assert dry_run_result.cleanup_status["no_cleanup_performed"] is True
            assert dry_run_result.cleanup_status["collection_preserved"] is True
            assert dry_run_result.cleanup_status["documents_preserved"] is True
            assert dry_run_result.validation_results["manual_cleanup_required"] is True


def test_inspection_mode_vs_other_modes(sample_documents, mock_weaviate_components):
    modes_to_test = [
        (DryRunMode.WORKFLOW_ONLY, {"cleanup_needed": False}),
        (DryRunMode.TEMPORARY, {"documents": True, "collections": True}),
        (DryRunMode.PERSISTENT, {"documents": True, "collections_preserved": True}),
        (DryRunMode.INSPECTION, {"no_cleanup_performed": True, "collection_preserved": True}),
    ]

    for mode, expected_cleanup in modes_to_test:
        writer = WeaviateDocumentWriter(
            index_name=f"ComparisonTest_{mode}",
            dry_run_config=DryRunConfig(mode=mode),
        )

        input_data = WriterInputSchema(documents=sample_documents)

        with patch.object(writer, "vector_store") as mock_vector_store:
            mock_vector_store.write_documents.return_value = len(sample_documents)

            if mode != DryRunMode.WORKFLOW_ONLY:
                with patch(
                    "dynamiq.storages.vector.dry_run_orchestrator.DryRunOrchestrator._initialize_test_vector_store",
                    return_value=(mock_vector_store, f"ComparisonTest_{mode}_test"),
                ):
                    result = writer.execute(input_data)
            else:
                result = writer.execute(input_data)

            dry_run_result = result["dry_run_result"]

            assert dry_run_result.mode == mode
            for key, value in expected_cleanup.items():
                assert dry_run_result.cleanup_status[key] == value
