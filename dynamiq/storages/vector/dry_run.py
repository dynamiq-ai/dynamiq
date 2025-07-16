from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DryRunMode(str, Enum):
    """Enumeration of available dry run modes.

    Attributes:
        WORKFLOW_ONLY: Test workflow without vector store writes
        TEMPORARY: Create temporary collection, upload documents, cleanup everything
        PERSISTENT: Create/reuse collection, upload documents, cleanup only documents
        INSPECTION: Create/reuse collection, upload documents, no cleanup (manual)
    """

    WORKFLOW_ONLY = "workflow_only"
    TEMPORARY = "temporary"
    PERSISTENT = "persistent"
    INSPECTION = "inspection"


class DryRunConfig(BaseModel):
    """Configuration for dry run operations.

    This class defines all parameters needed to configure dry run behavior
    across different vector store implementations.

    Attributes:
        mode: The dry run mode to use
        test_collection_suffix: Suffix to append to collection name for test collections
        document_id_prefix: Prefix to add to all document IDs in dry run mode
        cleanup_timeout: Maximum time in seconds to wait for cleanup operations
        preserve_on_error: Whether to preserve resources when errors occur
        batch_size: Optional batch size for document processing (uses store default if None)
        validation_enabled: Whether to enable schema and compatibility validation
    """

    mode: DryRunMode
    test_collection_suffix: str = Field(default="test", description="Suffix for test collection names")
    document_id_prefix: str = Field(default="test_", description="Prefix for document IDs in dry run mode")
    cleanup_timeout: int = Field(default=300, ge=0, description="Cleanup timeout in seconds")
    preserve_on_error: bool = Field(default=False, description="Preserve resources when errors occur")
    batch_size: int | None = Field(default=None, ge=1, description="Batch size for document processing")
    validation_enabled: bool = Field(default=True, description="Enable schema and compatibility validation")

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization validation."""
        if not self.test_collection_suffix.replace("_", "").replace("-", "").isalnum():
            raise ValueError(
                "test_collection_suffix must contain only alphanumeric characters, hyphens, and underscores"
            )

        if not self.document_id_prefix.replace("_", "").replace("-", "").isalnum():
            raise ValueError("document_id_prefix must contain only alphanumeric characters, hyphens, and underscores")


class DryRunResult(BaseModel):
    """Result of a dry run operation.

    This class contains comprehensive information about the execution
    of a dry run operation, including success status, timing, and cleanup details.

    Attributes:
        success: Whether the dry run operation completed successfully
        mode: The dry run mode that was used
        test_collection_name: Name of the test collection (if created)
        documents_processed: Number of documents that were processed
        duration_seconds: Total execution time in seconds
        validation_results: Results of schema and compatibility validation
        cleanup_status: Status of cleanup operations performed
        workflow_steps_completed: List of workflow steps that were completed
        error_message: Error message if the operation failed
        created_at: Timestamp when the result was created
    """

    success: bool
    mode: DryRunMode
    test_collection_name: str | None = None
    documents_processed: int
    duration_seconds: float
    validation_results: dict[str, Any] = Field(default_factory=dict)
    cleanup_status: dict[str, bool] = Field(default_factory=dict)
    workflow_steps_completed: list[str] = Field(default_factory=list)
    error_message: str | None = None
    created_at: datetime = Field(default_factory=datetime.now)

    def add_workflow_step(self, step: str) -> None:
        self.workflow_steps_completed.append(step)

    def add_validation_result(self, key: str, value: Any) -> None:
        self.validation_results[key] = value

    def add_cleanup_status(self, operation: str, success: bool) -> None:
        self.cleanup_status[operation] = success

    def is_cleanup_needed(self) -> bool:
        return self.mode in [DryRunMode.TEMPORARY, DryRunMode.PERSISTENT]

    def summary(self) -> str:
        status = "✅ SUCCESS" if self.success else "❌ FAILED"
        summary_lines = [
            f"{status} - Dry Run ({self.mode.value})",
            f"Documents: {self.documents_processed}",
            f"Duration: {self.duration_seconds:.2f}s",
        ]

        if self.test_collection_name:
            summary_lines.append(f"Collection: {self.test_collection_name}")

        if self.error_message:
            summary_lines.append(f"Error: {self.error_message}")

        return " | ".join(summary_lines)


class DryRunValidationError(Exception):
    """Exception raised when dry run validation fails."""

    pass


class DryRunExecutionError(Exception):
    """Exception raised when dry run execution fails."""

    pass


class DryRunCleanupError(Exception):
    """Exception raised when dry run cleanup fails."""

    pass
