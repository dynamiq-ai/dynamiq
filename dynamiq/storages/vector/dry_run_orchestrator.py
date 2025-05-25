import time
from typing import Any

from dynamiq.types import Document
from dynamiq.utils.logger import logger

from .dry_run import DryRunConfig, DryRunExecutionError, DryRunMode, DryRunResult, DryRunValidationError
from .resource_tracker import DryRunResourceTracker


class DryRunOrchestrator:
    """Central orchestrator for dry run operations across all vector store providers.

    This class manages the execution of dry run operations, handling validation,
    document preparation, and coordination with vector store implementations.
    """

    def __init__(self, vector_store_cls: type, vector_store_params: dict[str, Any], dry_run_config: DryRunConfig):
        """Initialize the orchestrator.

        Args:
            vector_store_cls: The vector store class to use
            vector_store_params: Parameters for vector store initialization
            dry_run_config: Configuration for the dry run operation
        """
        self.vector_store_cls = vector_store_cls
        self.vector_store_params = vector_store_params
        self.dry_run_config = dry_run_config
        self.resource_tracker = DryRunResourceTracker()

    def execute(self, documents: list[Document], content_key: str = None) -> DryRunResult:
        """Execute a dry run operation.

        Args:
            documents: List of documents to process
            content_key: Optional content key override

        Returns:
            DryRunResult: Comprehensive result of the dry run operation
        """
        start_time = time.time()
        result = DryRunResult(success=False, mode=self.dry_run_config.mode, documents_processed=0, duration_seconds=0.0)

        try:
            logger.info(f"Starting dry run operation in {self.dry_run_config.mode} mode")

            if self.dry_run_config.mode == DryRunMode.WORKFLOW_ONLY:
                return self._execute_workflow_only(documents, start_time, result)
            elif self.dry_run_config.mode in [DryRunMode.TEMPORARY, DryRunMode.PERSISTENT]:
                return self._execute_with_vector_store(documents, content_key, start_time, result)
            else:
                # INSPECTION mode - for Stage 3
                raise NotImplementedError(f"Mode {self.dry_run_config.mode} will be implemented in Stage 3")

        except Exception as e:
            logger.error(f"Dry run operation failed: {str(e)}")
            result.error_message = str(e)
            result.duration_seconds = time.time() - start_time
            return result

    def _execute_workflow_only(
        self, documents: list[Document], start_time: float, result: DryRunResult
    ) -> DryRunResult:
        """Execute workflow-only dry run mode.

        This mode processes documents through all workflow steps but skips
        the actual vector store write operation.

        Args:
            documents: Documents to process
            start_time: Start time for duration calculation
            result: Result object to populate

        Returns:
            DryRunResult: Updated result object
        """
        try:
            result.add_workflow_step("Document validation")
            validated_docs = self._validate_documents(documents)
            logger.debug(f"Validated {len(validated_docs)} documents")

            result.add_workflow_step("Embedding verification")
            self._verify_embeddings(validated_docs)
            logger.debug("Embedding verification completed")

            if self.dry_run_config.validation_enabled:
                result.add_workflow_step("Schema compatibility check")
                validation_results = self._check_schema_compatibility()
                for key, value in validation_results.items():
                    result.add_validation_result(key, value)
                logger.debug("Schema compatibility check completed")

            result.add_workflow_step("Document preparation")
            prepared_docs = self._prepare_documents_for_storage(validated_docs)
            logger.debug(f"Prepared {len(prepared_docs)} documents for storage")

            result.add_workflow_step("Vector store write (SKIPPED)")
            logger.info("Vector store write operation skipped in workflow-only mode")

            result.success = True
            result.documents_processed = len(prepared_docs)
            result.duration_seconds = time.time() - start_time
            result.add_cleanup_status("cleanup_needed", False)

            logger.info(f"Workflow-only dry run completed successfully: {result.summary()}")
            return result

        except Exception as e:
            logger.error(f"Workflow-only dry run failed: {str(e)}")
            raise DryRunExecutionError(f"Workflow-only execution failed: {str(e)}") from e

    def _execute_with_vector_store(
        self, documents: list[Document], content_key: str, start_time: float, result: DryRunResult
    ) -> DryRunResult:
        """Execute dry run modes that interact with vector stores.

        This method implements TEMPORARY and PERSISTENT modes that actually
        interact with vector stores but with proper isolation and cleanup.

        Args:
            documents: Documents to process
            content_key: Content key for storage
            start_time: Start time for duration calculation
            result: Result object to populate

        Returns:
            DryRunResult: Updated result object
        """
        vector_store = None

        try:
            result.add_workflow_step("Document validation")
            validated_docs = self._validate_documents(documents)
            logger.debug(f"Validated {len(validated_docs)} documents")

            result.add_workflow_step("Embedding verification")
            self._verify_embeddings(validated_docs)
            logger.debug("Embedding verification completed")

            result.add_workflow_step("Document preparation")
            prepared_docs = self._prepare_documents_for_storage(validated_docs)
            logger.debug(f"Prepared {len(prepared_docs)} documents for storage")

            result.add_workflow_step("Vector store initialization")
            vector_store, test_collection_name = self._initialize_test_vector_store()
            result.test_collection_name = test_collection_name
            logger.debug(f"Initialized test vector store with collection: {test_collection_name}")

            if self.dry_run_config.validation_enabled:
                result.add_workflow_step("Schema compatibility check")
                validation_results = self._check_schema_compatibility_with_store(vector_store)
                for key, value in validation_results.items():
                    result.add_validation_result(key, value)
                logger.debug("Schema compatibility check completed")

            result.add_workflow_step("Document upload")
            doc_count = self._upload_documents_to_test_store(vector_store, prepared_docs, content_key)
            result.documents_processed = doc_count
            logger.debug(f"Uploaded {doc_count} documents to test collection")

            result.add_workflow_step("Cleanup")
            cleanup_status = self._perform_cleanup(vector_store)
            for operation, status in cleanup_status.items():
                result.add_cleanup_status(operation, status)
            logger.debug(f"Cleanup completed: {cleanup_status}")

            result.success = True
            result.duration_seconds = time.time() - start_time

            logger.info(f"Vector store dry run completed successfully: {result.summary()}")
            return result

        except Exception as e:
            logger.error(f"Vector store dry run failed: {str(e)}")

            if vector_store and not self.dry_run_config.preserve_on_error:
                try:
                    emergency_cleanup = self._perform_cleanup(vector_store)
                    result.add_cleanup_status("emergency_cleanup", emergency_cleanup.get("documents", False))
                    logger.info("Emergency cleanup performed after error")
                except Exception as cleanup_error:
                    logger.error(f"Emergency cleanup failed: {str(cleanup_error)}")
                    result.add_cleanup_status("emergency_cleanup", False)

            raise DryRunExecutionError(f"Vector store execution failed: {str(e)}") from e

    def _validate_documents(self, documents: list[Document]) -> list[Document]:
        """Validate documents for dry run processing.

        Args:
            documents: Documents to validate

        Returns:
            List[Document]: Validated documents

        Raises:
            DryRunValidationError: If validation fails
        """
        if not documents:
            raise DryRunValidationError("No documents provided for processing")

        validated_docs = []
        for i, doc in enumerate(documents):
            try:
                if not isinstance(doc, Document):
                    raise DryRunValidationError(f"Document {i} is not a valid Document instance")

                if not doc.id or doc.id == "":
                    raise DryRunValidationError(f"Document {i} missing required 'id' field")

                if doc.content is None:
                    logger.warning(f"Document {doc.id} has no content")

                validated_docs.append(doc)

            except Exception as e:
                raise DryRunValidationError(f"Document {i} validation failed: {str(e)}") from e

        logger.debug(f"Successfully validated {len(validated_docs)} documents")
        return validated_docs

    def _verify_embeddings(self, documents: list[Document]) -> None:
        """Verify document embeddings are valid.

        Args:
            documents: Documents to verify

        Raises:
            DryRunValidationError: If embedding verification fails
        """
        embedding_dimensions = None

        for doc in documents:
            if doc.embedding is not None:
                if not isinstance(doc.embedding, (list, tuple)):
                    raise DryRunValidationError(f"Document {doc.id} has invalid embedding type: {type(doc.embedding)}")

                if embedding_dimensions is None:
                    embedding_dimensions = len(doc.embedding)
                elif len(doc.embedding) != embedding_dimensions:
                    raise DryRunValidationError(
                        f"Document {doc.id} embedding dimension mismatch: "
                        f"expected {embedding_dimensions}, got {len(doc.embedding)}"
                    )

                try:
                    float_embedding = [float(x) for x in doc.embedding]
                    if any(not (-1e10 <= x <= 1e10) for x in float_embedding):
                        logger.warning(f"Document {doc.id} has extreme embedding values")
                except (ValueError, TypeError) as e:
                    raise DryRunValidationError(f"Document {doc.id} has non-numeric embedding values: {str(e)}") from e

        if embedding_dimensions:
            logger.debug(f"Verified embeddings with {embedding_dimensions} dimensions")

    def _check_schema_compatibility(self) -> dict[str, Any]:
        """Check schema compatibility with production vector store.

        This is a placeholder for more sophisticated schema validation
        that will be implemented per vector store type.

        Returns:
            Dict[str, Any]: Validation results
        """
        validation_results = {
            "schema_validation_performed": True,
            "validation_timestamp": time.time(),
            "vector_store_type": self.vector_store_cls.__name__,
        }

        # In Stage 2, this will include:
        # - Collection/index existence checks
        # - Dimension compatibility verification
        # - Property schema validation
        # - Distance metric compatibility

        logger.debug("Basic schema compatibility check completed")
        return validation_results

    def _prepare_documents_for_storage(self, documents: list[Document]) -> list[Document]:
        """Prepare documents for dry run storage.

        This method adds dry run prefixes and metadata to documents
        to distinguish them from production data.

        Args:
            documents: Documents to prepare

        Returns:
            List[Document]: Prepared documents
        """
        prepared_docs = []

        for doc in documents:
            prepared_doc = Document(
                id=f"{self.dry_run_config.document_id_prefix}{doc.id}",
                content=doc.content,
                metadata=doc.metadata.copy() if doc.metadata else {},
                embedding=doc.embedding,
                score=doc.score,
            )

            if prepared_doc.metadata is None:
                prepared_doc.metadata = {}

            prepared_doc.metadata.update(
                {
                    "_dry_run": True,
                    "_dry_run_mode": self.dry_run_config.mode,
                    "_original_id": doc.id,
                    "_dry_run_timestamp": time.time(),
                }
            )

            prepared_docs.append(prepared_doc)

        logger.debug(f"Prepared {len(prepared_docs)} documents with dry run metadata")
        return prepared_docs

    def _initialize_test_vector_store(self):
        """Initialize vector store with test collection.

        Returns:
            Tuple of (vector_store_instance, test_collection_name)
        """
        original_index_name = self.vector_store_params.get("index_name", "default")
        test_collection_name = f"{original_index_name}_{self.dry_run_config.test_collection_suffix}"

        test_params = self.vector_store_params.copy()
        test_params["index_name"] = test_collection_name

        vector_store = self.vector_store_cls(**test_params)

        self.resource_tracker.register_collection(test_collection_name, vector_store, self.dry_run_config.mode)

        logger.debug(f"Created test collection: {test_collection_name}")
        return vector_store, test_collection_name

    def _check_schema_compatibility_with_store(self, vector_store) -> dict[str, Any]:
        """Check schema compatibility with actual vector store.

        Args:
            vector_store: The vector store instance

        Returns:
            Dict[str, Any]: Detailed validation results
        """
        validation_results = {
            "schema_validation_performed": True,
            "validation_timestamp": time.time(),
            "vector_store_type": self.vector_store_cls.__name__,
            "test_collection_name": vector_store.index_name if hasattr(vector_store, "index_name") else "unknown",
        }

        try:
            if hasattr(vector_store, "collection_exists"):
                collection_exists = vector_store.collection_exists()
                validation_results["collection_exists"] = collection_exists

                if not collection_exists and hasattr(vector_store, "create_collection"):
                    vector_store.create_collection()
                    validation_results["collection_created"] = True
                    logger.debug("Test collection created successfully")

            if hasattr(vector_store, "health_check"):
                health_status = vector_store.health_check()
                validation_results["health_check"] = health_status

            validation_results["schema_compatible"] = True

        except Exception as e:
            logger.warning(f"Schema compatibility check failed: {str(e)}")
            validation_results["schema_compatible"] = False
            validation_results["error"] = str(e)

        return validation_results

    def _upload_documents_to_test_store(self, vector_store, documents: list[Document], content_key: str) -> int:
        """Upload documents to the test vector store.

        Args:
            vector_store: The vector store instance
            documents: Documents to upload
            content_key: Content key for storage

        Returns:
            int: Number of documents successfully uploaded
        """
        try:
            doc_ids = [doc.id for doc in documents]

            self.resource_tracker.register_documents(doc_ids, vector_store)

            if hasattr(vector_store, "write_documents"):
                uploaded_count = vector_store.write_documents(documents, content_key=content_key)
                logger.debug(f"Successfully uploaded {uploaded_count} documents")
                return uploaded_count
            else:
                logger.warning("Vector store doesn't have write_documents method, simulating upload")
                return len(documents)

        except Exception as e:
            logger.error(f"Document upload failed: {str(e)}")
            raise DryRunExecutionError(f"Failed to upload documents: {str(e)}") from e

    def _perform_cleanup(self, vector_store) -> dict[str, bool]:
        """Perform cleanup based on dry run mode.

        Args:
            vector_store: The vector store instance

        Returns:
            Dict[str, bool]: Cleanup status for each operation
        """
        try:
            cleanup_status = self.resource_tracker.cleanup(self.dry_run_config.mode)

            if self.dry_run_config.mode == DryRunMode.TEMPORARY:
                logger.info("TEMPORARY mode: Cleaning up documents and collections")
            elif self.dry_run_config.mode == DryRunMode.PERSISTENT:
                logger.info("PERSISTENT mode: Cleaning up documents only, preserving collections")

            return cleanup_status

        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")
            return {"cleanup_failed": True, "error": str(e)}
