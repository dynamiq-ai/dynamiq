from unittest.mock import Mock, patch

import pytest

from dynamiq.storages.vector.dry_run import DryRunConfig, DryRunMode, DryRunResult, DryRunValidationError
from dynamiq.storages.vector.dry_run_orchestrator import DryRunOrchestrator
from dynamiq.storages.vector.resource_tracker import DryRunResourceTracker
from dynamiq.storages.vector.weaviate.weaviate import WeaviateVectorStore
from dynamiq.types import Document


@pytest.fixture
def sample_documents():
    return [
        Document(
            id="test_doc_1",
            content="Sample document content about machine learning",
            embedding=[0.1] * 10,
            metadata={"category": "tech", "type": "sample"},
        ),
        Document(
            id="test_doc_2",
            content="Another document about artificial intelligence",
            embedding=[0.2] * 10,
            metadata={"category": "tech", "type": "sample"},
        ),
        Document(
            id="test_doc_3",
            content="Document about natural language processing",
            embedding=[0.3] * 10,
            metadata={"category": "nlp", "type": "sample"},
        ),
    ]


@pytest.fixture
def workflow_only_config():
    return DryRunConfig(mode=DryRunMode.WORKFLOW_ONLY)


@pytest.fixture
def mock_vector_store_cls():
    mock_cls = Mock()
    mock_cls.__name__ = "MockVectorStore"
    mock_cls.return_value = Mock()
    return mock_cls


@pytest.fixture
def tenant_config():
    return {"tenant_name": "test_tenant"}


def test_config_defaults():
    config = DryRunConfig(mode=DryRunMode.WORKFLOW_ONLY)
    assert config.mode == DryRunMode.WORKFLOW_ONLY
    assert config.test_collection_suffix == "test"
    assert config.document_id_prefix == "test_"
    assert config.cleanup_timeout == 300
    assert config.preserve_on_error is False
    assert config.batch_size is None
    assert config.validation_enabled is True


def test_config_custom_values():
    config = DryRunConfig(
        mode=DryRunMode.WORKFLOW_ONLY,
        test_collection_suffix="custom",
        document_id_prefix="dry_",
        cleanup_timeout=600,
        preserve_on_error=True,
        batch_size=50,
        validation_enabled=False,
    )
    assert config.mode == DryRunMode.WORKFLOW_ONLY
    assert config.test_collection_suffix == "custom"
    assert config.document_id_prefix == "dry_"
    assert config.cleanup_timeout == 600
    assert config.preserve_on_error is True
    assert config.batch_size == 50
    assert config.validation_enabled is False


def test_mode_validation():
    assert DryRunMode.WORKFLOW_ONLY == "workflow_only"
    assert DryRunMode.TEMPORARY == "temporary"
    assert DryRunMode.PERSISTENT == "persistent"
    assert DryRunMode.INSPECTION == "inspection"


def test_invalid_mode_raises_error():
    with pytest.raises(ValueError):
        DryRunConfig(mode="invalid_mode")


def test_cleanup_timeout_validation():
    with pytest.raises(Exception):
        DryRunConfig(mode=DryRunMode.WORKFLOW_ONLY, cleanup_timeout=-1)

    config = DryRunConfig(mode=DryRunMode.WORKFLOW_ONLY, cleanup_timeout=0)
    assert config.cleanup_timeout == 0


def test_batch_size_validation():
    with pytest.raises(Exception):
        DryRunConfig(mode=DryRunMode.WORKFLOW_ONLY, batch_size=-1)

    with pytest.raises(Exception):
        DryRunConfig(mode=DryRunMode.WORKFLOW_ONLY, batch_size=0)


def test_suffix_validation():
    with pytest.raises(Exception):
        DryRunConfig(mode=DryRunMode.WORKFLOW_ONLY, test_collection_suffix="")


def test_prefix_validation():
    with pytest.raises(Exception):
        DryRunConfig(mode=DryRunMode.WORKFLOW_ONLY, document_id_prefix="")


def test_result_creation():
    result = DryRunResult(mode=DryRunMode.WORKFLOW_ONLY, success=True, documents_processed=5, duration_seconds=1.5)

    assert result.mode == DryRunMode.WORKFLOW_ONLY
    assert result.success is True
    assert result.documents_processed == 5
    assert result.duration_seconds == 1.5
    assert result.error_message is None


def test_error_result():
    result = DryRunResult(
        mode=DryRunMode.WORKFLOW_ONLY,
        success=False,
        documents_processed=0,
        duration_seconds=0.5,
        error_message="Test error",
    )

    assert result.success is False
    assert result.error_message == "Test error"


def test_helper_methods():
    result = DryRunResult(mode=DryRunMode.WORKFLOW_ONLY, success=True, documents_processed=3, duration_seconds=2.0)

    result.add_workflow_step("Document validation")
    result.add_workflow_step("Embedding verification")
    assert len(result.workflow_steps_completed) == 2

    result.add_validation_result("schema_valid", True)
    assert result.validation_results["schema_valid"] is True

    result.add_cleanup_status("documents_cleaned", True)
    assert result.cleanup_status["documents_cleaned"] is True


def test_is_cleanup_needed():
    result = DryRunResult(mode=DryRunMode.WORKFLOW_ONLY, success=True, documents_processed=5, duration_seconds=1.0)
    assert result.is_cleanup_needed() is False

    result_temp = DryRunResult(mode=DryRunMode.TEMPORARY, success=True, documents_processed=5, duration_seconds=1.0)
    assert result_temp.is_cleanup_needed() is True


def test_summary():
    result = DryRunResult(mode=DryRunMode.WORKFLOW_ONLY, success=True, documents_processed=5, duration_seconds=1.5)

    summary = result.summary()
    assert "workflow_only" in summary
    assert "5" in summary
    assert "1.50" in summary
    assert "SUCCESS" in summary


def test_orchestrator_initialization(mock_vector_store_cls, workflow_only_config):
    orchestrator = DryRunOrchestrator(
        vector_store_cls=mock_vector_store_cls,
        vector_store_params={"param1": "value1"},
        dry_run_config=workflow_only_config,
    )

    assert orchestrator.vector_store_cls == mock_vector_store_cls
    assert orchestrator.vector_store_params == {"param1": "value1"}
    assert orchestrator.dry_run_config == workflow_only_config


def test_workflow_only_execution_success(mock_vector_store_cls, workflow_only_config, sample_documents):
    orchestrator = DryRunOrchestrator(
        vector_store_cls=mock_vector_store_cls, vector_store_params={}, dry_run_config=workflow_only_config
    )

    result = orchestrator.execute(sample_documents)

    assert isinstance(result, DryRunResult)
    assert result.mode == DryRunMode.WORKFLOW_ONLY
    assert result.success is True
    assert result.documents_processed == len(sample_documents)
    assert result.duration_seconds > 0
    assert len(result.workflow_steps_completed) > 0


def test_workflow_only_execution_validation_disabled(mock_vector_store_cls, sample_documents):
    config = DryRunConfig(mode=DryRunMode.WORKFLOW_ONLY, validation_enabled=False)
    orchestrator = DryRunOrchestrator(
        vector_store_cls=mock_vector_store_cls, vector_store_params={}, dry_run_config=config
    )

    result = orchestrator.execute(sample_documents)

    assert result.success is True


def test_inspection_mode_execution_success(mock_vector_store_cls, sample_documents):
    config = DryRunConfig(mode=DryRunMode.INSPECTION, test_collection_suffix="analysis")
    orchestrator = DryRunOrchestrator(
        vector_store_cls=mock_vector_store_cls, vector_store_params={}, dry_run_config=config
    )

    mock_vector_store = Mock()
    mock_vector_store.write_documents.return_value = len(sample_documents)
    mock_vector_store_cls.return_value = mock_vector_store

    with patch.object(orchestrator, "_initialize_test_vector_store", return_value=(mock_vector_store, "test_analysis")):
        result = orchestrator.execute(sample_documents)

    assert result.success is True
    assert result.mode == DryRunMode.INSPECTION
    assert result.test_collection_name == "test_analysis"
    assert result.documents_processed == len(sample_documents)
    assert result.cleanup_status["no_cleanup_performed"] is True
    assert result.cleanup_status["collection_preserved"] is True
    assert result.cleanup_status["documents_preserved"] is True
    assert result.validation_results["manual_cleanup_required"] is True
    assert "Inspection setup (NO CLEANUP)" in result.workflow_steps_completed


def test_inspection_mode_error_handling(mock_vector_store_cls, sample_documents):
    config = DryRunConfig(mode=DryRunMode.INSPECTION, preserve_on_error=True)
    orchestrator = DryRunOrchestrator(
        vector_store_cls=mock_vector_store_cls, vector_store_params={}, dry_run_config=config
    )

    with patch.object(orchestrator, "_validate_documents", side_effect=ValueError("Test error")):
        result = orchestrator.execute(sample_documents)

    assert result.success is False
    assert "Test error" in result.error_message
    assert "error_resources_preserved" not in result.cleanup_status


def test_inspection_mode_error_with_vector_store(mock_vector_store_cls, sample_documents):
    config = DryRunConfig(mode=DryRunMode.INSPECTION, preserve_on_error=True)
    orchestrator = DryRunOrchestrator(
        vector_store_cls=mock_vector_store_cls, vector_store_params={}, dry_run_config=config
    )

    mock_vector_store = Mock()
    mock_vector_store.write_documents.side_effect = ValueError("Upload error")

    with patch.object(
        orchestrator, "_initialize_test_vector_store", return_value=(mock_vector_store, "test_collection")
    ):
        result = orchestrator.execute(sample_documents)

    assert result.success is False
    assert "Upload error" in result.error_message
    assert result.cleanup_status["error_resources_preserved"] is True


def test_execution_error_handling(mock_vector_store_cls, workflow_only_config):
    orchestrator = DryRunOrchestrator(
        vector_store_cls=mock_vector_store_cls, vector_store_params={}, dry_run_config=workflow_only_config
    )

    result = orchestrator.execute([])
    assert result.success is False
    assert result.error_message is not None


def test_validate_documents_success(mock_vector_store_cls, workflow_only_config, sample_documents):
    orchestrator = DryRunOrchestrator(
        vector_store_cls=mock_vector_store_cls, vector_store_params={}, dry_run_config=workflow_only_config
    )

    orchestrator._validate_documents(sample_documents)


def test_validate_documents_empty_list(mock_vector_store_cls, workflow_only_config):
    orchestrator = DryRunOrchestrator(
        vector_store_cls=mock_vector_store_cls, vector_store_params={}, dry_run_config=workflow_only_config
    )

    with pytest.raises(DryRunValidationError, match="No documents provided"):
        orchestrator._validate_documents([])


def test_validate_documents_invalid_document(mock_vector_store_cls, workflow_only_config):
    orchestrator = DryRunOrchestrator(
        vector_store_cls=mock_vector_store_cls, vector_store_params={}, dry_run_config=workflow_only_config
    )

    with pytest.raises(DryRunValidationError):
        orchestrator._validate_documents(["not_a_document"])


def test_validate_documents_missing_id(mock_vector_store_cls, workflow_only_config):
    orchestrator = DryRunOrchestrator(
        vector_store_cls=mock_vector_store_cls, vector_store_params={}, dry_run_config=workflow_only_config
    )

    doc_without_id = Document(id=None, content="test", embedding=[0.1, 0.2])
    with pytest.raises(DryRunValidationError):
        orchestrator._validate_documents([doc_without_id])


def test_verify_embeddings_success(mock_vector_store_cls, workflow_only_config, sample_documents):
    orchestrator = DryRunOrchestrator(
        vector_store_cls=mock_vector_store_cls, vector_store_params={}, dry_run_config=workflow_only_config
    )

    orchestrator._verify_embeddings(sample_documents)


def test_verify_embeddings_dimension_mismatch(mock_vector_store_cls, workflow_only_config):
    orchestrator = DryRunOrchestrator(
        vector_store_cls=mock_vector_store_cls, vector_store_params={}, dry_run_config=workflow_only_config
    )

    docs_with_different_dims = [
        Document(id="1", content="test", embedding=[0.1, 0.2]),
        Document(id="2", content="test", embedding=[0.1, 0.2, 0.3]),
    ]

    with pytest.raises(DryRunValidationError, match="embedding dimension mismatch"):
        orchestrator._verify_embeddings(docs_with_different_dims)


def test_verify_embeddings_invalid_type(mock_vector_store_cls, workflow_only_config):
    orchestrator = DryRunOrchestrator(
        vector_store_cls=mock_vector_store_cls, vector_store_params={}, dry_run_config=workflow_only_config
    )

    doc_with_invalid_embedding = Document(id="1", content="test", embedding=[0.1, 0.2])
    object.__setattr__(doc_with_invalid_embedding, "embedding", "invalid")

    with pytest.raises(DryRunValidationError, match="has invalid embedding type"):
        orchestrator._verify_embeddings([doc_with_invalid_embedding])


def test_verify_embeddings_non_numeric(mock_vector_store_cls, workflow_only_config):
    orchestrator = DryRunOrchestrator(
        vector_store_cls=mock_vector_store_cls, vector_store_params={}, dry_run_config=workflow_only_config
    )

    doc_with_non_numeric = Document(id="1", content="test", embedding=[0.1, "invalid", 0.3])
    with pytest.raises(DryRunValidationError, match="has non-numeric embedding values"):
        orchestrator._verify_embeddings([doc_with_non_numeric])


def test_prepare_documents_for_storage(mock_vector_store_cls, workflow_only_config, sample_documents):
    orchestrator = DryRunOrchestrator(
        vector_store_cls=mock_vector_store_cls, vector_store_params={}, dry_run_config=workflow_only_config
    )

    prepared_docs = orchestrator._prepare_documents_for_storage(sample_documents)

    assert len(prepared_docs) == len(sample_documents)
    for i, doc in enumerate(prepared_docs):
        assert doc.id.startswith("test_")
        assert doc.metadata["_dry_run"] is True
        assert doc.metadata["_dry_run_mode"] == DryRunMode.WORKFLOW_ONLY
        assert doc.metadata["_original_id"] == sample_documents[i].id


def test_check_schema_compatibility(mock_vector_store_cls, workflow_only_config):
    orchestrator = DryRunOrchestrator(
        vector_store_cls=mock_vector_store_cls, vector_store_params={}, dry_run_config=workflow_only_config
    )

    result = orchestrator._check_schema_compatibility()
    assert isinstance(result, dict)
    assert "schema_validation_performed" in result
    assert result["schema_validation_performed"] is True


def test_vector_store_params_include_tenant_name(mock_vector_store_cls, workflow_only_config):
    params_with_tenant = {"tenant_name": "test_tenant", "other_param": "value"}

    orchestrator = DryRunOrchestrator(
        vector_store_cls=mock_vector_store_cls,
        vector_store_params=params_with_tenant,
        dry_run_config=workflow_only_config,
    )

    assert orchestrator.vector_store_params["tenant_name"] == "test_tenant"
    assert orchestrator.vector_store_params["other_param"] == "value"


def test_orchestrator_receives_tenant_params(mock_vector_store_cls, workflow_only_config, tenant_config):
    orchestrator = DryRunOrchestrator(
        vector_store_cls=mock_vector_store_cls, vector_store_params=tenant_config, dry_run_config=workflow_only_config
    )

    assert "tenant_name" in orchestrator.vector_store_params
    assert orchestrator.vector_store_params["tenant_name"] == "test_tenant"


def test_workflow_only_mode_with_tenant(mock_vector_store_cls, workflow_only_config, sample_documents, tenant_config):
    orchestrator = DryRunOrchestrator(
        vector_store_cls=mock_vector_store_cls, vector_store_params=tenant_config, dry_run_config=workflow_only_config
    )

    result = orchestrator.execute(sample_documents)

    assert result.success is True
    assert result.mode == DryRunMode.WORKFLOW_ONLY
    assert result.documents_processed == len(sample_documents)


def test_dry_run_document_preparation_with_tenant(
    mock_vector_store_cls, workflow_only_config, sample_documents, tenant_config
):
    orchestrator = DryRunOrchestrator(
        vector_store_cls=mock_vector_store_cls, vector_store_params=tenant_config, dry_run_config=workflow_only_config
    )

    prepared_docs = orchestrator._prepare_documents_for_storage(sample_documents)

    for doc in prepared_docs:
        assert doc.metadata["_dry_run"] is True
        assert doc.metadata["_dry_run_mode"] == DryRunMode.WORKFLOW_ONLY


def test_test_collection_naming_with_tenant(
    mock_vector_store_cls, workflow_only_config, sample_documents, tenant_config
):
    orchestrator = DryRunOrchestrator(
        vector_store_cls=mock_vector_store_cls, vector_store_params=tenant_config, dry_run_config=workflow_only_config
    )

    result = orchestrator.execute(sample_documents)

    assert result.success is True


def test_input_level_tenant_override(mock_vector_store_cls, workflow_only_config, sample_documents):
    orchestrator = DryRunOrchestrator(
        vector_store_cls=mock_vector_store_cls,
        vector_store_params={"base_param": "value"},
        dry_run_config=workflow_only_config,
    )

    with patch.object(orchestrator, "_prepare_documents_for_storage") as mock_prep:
        mock_prep.return_value = sample_documents

        orchestrator.vector_store_params.update({"tenant_name": "runtime_tenant"})
        result = orchestrator.execute(sample_documents)

        assert result.success is True


def test_error_handling_with_invalid_tenant(mock_vector_store_cls, workflow_only_config, sample_documents):
    invalid_tenant_config = {"tenant_name": ""}

    orchestrator = DryRunOrchestrator(
        vector_store_cls=mock_vector_store_cls,
        vector_store_params=invalid_tenant_config,
        dry_run_config=workflow_only_config,
    )

    result = orchestrator.execute(sample_documents)
    assert result.success is True


def test_end_to_end_tenant_dry_run(mock_vector_store_cls, workflow_only_config, sample_documents, tenant_config):
    orchestrator = DryRunOrchestrator(
        vector_store_cls=mock_vector_store_cls, vector_store_params=tenant_config, dry_run_config=workflow_only_config
    )

    result = orchestrator.execute(sample_documents)

    assert result.success is True
    assert result.documents_processed == len(sample_documents)
    assert result.duration_seconds > 0


def test_tenant_isolation_in_dry_run(mock_vector_store_cls, workflow_only_config, sample_documents):
    tenant1_config = {"tenant_name": "tenant1"}
    tenant2_config = {"tenant_name": "tenant2"}

    orchestrator1 = DryRunOrchestrator(
        vector_store_cls=mock_vector_store_cls, vector_store_params=tenant1_config, dry_run_config=workflow_only_config
    )
    result1 = orchestrator1.execute(sample_documents)

    orchestrator2 = DryRunOrchestrator(
        vector_store_cls=mock_vector_store_cls, vector_store_params=tenant2_config, dry_run_config=workflow_only_config
    )
    result2 = orchestrator2.execute(sample_documents)

    assert result1.success is True
    assert result2.success is True

    assert orchestrator1.vector_store_params["tenant_name"] == "tenant1"
    assert orchestrator2.vector_store_params["tenant_name"] == "tenant2"


def test_temporary_mode_execution(mock_vector_store_cls, sample_documents):
    config = DryRunConfig(mode=DryRunMode.TEMPORARY)
    orchestrator = DryRunOrchestrator(
        vector_store_cls=mock_vector_store_cls,
        vector_store_params={"index_name": "test_collection"},
        dry_run_config=config,
    )

    result = orchestrator.execute(sample_documents)

    assert result.success is True
    assert result.mode == DryRunMode.TEMPORARY
    assert result.test_collection_name is not None
    assert len(result.workflow_steps_completed) > 5
    assert "Document validation" in result.workflow_steps_completed
    assert "Vector store initialization" in result.workflow_steps_completed
    assert "Document upload" in result.workflow_steps_completed
    assert "Cleanup" in result.workflow_steps_completed
    assert result.cleanup_status.get("documents") is True
    assert result.cleanup_status.get("collections") is True


def test_persistent_mode_execution(mock_vector_store_cls, sample_documents):
    config = DryRunConfig(mode=DryRunMode.PERSISTENT)
    orchestrator = DryRunOrchestrator(
        vector_store_cls=mock_vector_store_cls,
        vector_store_params={"index_name": "test_collection"},
        dry_run_config=config,
    )

    result = orchestrator.execute(sample_documents)

    assert result.success is True
    assert result.mode == DryRunMode.PERSISTENT
    assert result.test_collection_name is not None
    assert len(result.workflow_steps_completed) > 5
    assert "Document validation" in result.workflow_steps_completed
    assert "Vector store initialization" in result.workflow_steps_completed
    assert "Document upload" in result.workflow_steps_completed
    assert "Cleanup" in result.workflow_steps_completed
    assert result.cleanup_status.get("documents") is True
    assert result.cleanup_status.get("collections_preserved") is True


def test_resource_tracker_functionality():
    tracker = DryRunResourceTracker()
    mock_vector_store = Mock()

    tracker.register_collection("test_collection", mock_vector_store, DryRunMode.TEMPORARY)
    tracker.register_documents(["doc1", "doc2"], mock_vector_store)

    assert len(tracker.collections) == 1
    assert len(tracker.documents) == 1
    assert tracker.get_document_count() == 2

    collection_info = tracker.get_collection_info("test_collection")
    assert collection_info is not None
    assert collection_info["name"] == "test_collection"
    assert collection_info["mode"] == DryRunMode.TEMPORARY

    tracker.clear()
    assert len(tracker.collections) == 0
    assert len(tracker.documents) == 0


def test_resource_tracker_inspection_mode():
    tracker = DryRunResourceTracker()
    mock_vector_store = Mock()

    tracker.register_collection("inspection_collection", mock_vector_store, DryRunMode.INSPECTION)
    tracker.register_documents(["doc1", "doc2", "doc3"], mock_vector_store)

    cleanup_result = tracker.cleanup(DryRunMode.INSPECTION)

    assert cleanup_result["no_cleanup_required"] is True
    assert len(tracker.collections) == 1
    assert len(tracker.documents) == 1
    assert tracker.get_document_count() == 3


def test_weaviate_dry_run_collection_methods():
    with patch("dynamiq.connections.Weaviate") as mock_connection:
        mock_client = Mock()
        mock_client.collections.exists.return_value = False
        mock_client.collections.create.return_value = None
        mock_client.collections.delete.return_value = None
        mock_connection.return_value.connect.return_value = mock_client

        mock_collection = Mock()
        mock_collection.name = "TestCollection"
        mock_client.collections.get.return_value = mock_collection

        store = WeaviateVectorStore(
            connection=mock_connection.return_value, index_name="TestCollection", create_if_not_exist=True
        )

        store.client = mock_client

        exists = store.collection_exists()
        assert exists is False
        mock_client.collections.exists.assert_called_with("TestCollection")

        created = store.create_collection()
        assert created is True

        mock_client.collections.exists.return_value = True
        deleted = store.delete_collection()
        assert deleted is True
        mock_client.collections.delete.assert_called_with("TestCollection")

        health = store.health_check()
        assert "healthy" in health
        assert "collection_exists" in health
        assert "client_ready" in health
