from io import BytesIO
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dynamiq.components.converters.textract import TextractAdapter, TextractFeatureType
from dynamiq.components.converters.textract import TextractFileConverter as TextractFileConverterComponent
from dynamiq.components.converters.textract import (
    TextractNotificationChannel,
    TextractProcessingMode,
    TextractQuery,
    TextractS3Object,
)
from dynamiq.connections import AWSTextract
from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes.node import ConnectionNode, ErrorHandling, NodeGroup, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.types import DocumentCreationMode
from dynamiq.utils.logger import logger


class TextractFileConverterInputSchema(BaseModel):
    """Inputs are nullable so a YAML selector pointing at an absent key resolves to None."""

    file_paths: list[str] | None = Field(default=None, description="Parameter to provide path to files.")
    files: list[BytesIO | bytes] | None = Field(default=None, description="Parameter to provide files.")
    s3_objects: list[TextractS3Object] | None = Field(
        default=None,
        description="Parameter to provide S3-resident documents as {'bucket': ..., 'name': ..., 'version': ...}.",
    )
    metadata: dict | list | None = Field(default=None, description="Parameter to provide metadata.")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def validate_file_source(self):
        """Validate that at least one of `file_paths`, `files` or `s3_objects` is specified"""
        if not self.file_paths and not self.files and not self.s3_objects:
            raise ValueError("Either `file_paths`, `files` or `s3_objects` must be provided.")
        return self


class TextractFileConverter(ConnectionNode):
    """
    A converter that extracts text, tables, forms and query answers with Amazon Textract OCR.

    Textract is the right converter for scanned or image-only documents (PDF, PNG, JPEG, TIFF)
    where no text layer exists for PyPDF to read. It returns typed blocks - lines, tables, cells,
    key-value pairs, query answers, signatures and layout regions - which this node renders into
    Markdown Documents with the structured data preserved in metadata.

    Args:
        connection (AWSTextract, optional): The AWS connection used to build the Textract client.
            Defaults to a new AWSTextract connection resolved from environment/profile credentials.
        document_creation_mode (Literal["one-doc-per-file", "one-doc-per-page"], optional): How to
            create Documents from the analyzed pages. Defaults to "one-doc-per-file".
        feature_types (list[TextractFeatureType], optional): Analysis features to request. Each
            feature adds cost and latency, so the default is an empty list, which uses the cheaper
            `DetectDocumentText` API and returns plain text lines. `LAYOUT` gives the best Markdown
            structure; `TABLES` renders Markdown tables; `FORMS` extracts key-value pairs;
            `QUERIES` answers `queries`; `SIGNATURES` detects signatures.
        queries (list[TextractQuery], optional): Questions answered against the document. Required
            with, and only valid with, the `QUERIES` feature.
        adapters (list[TextractAdapter], optional): Trained Textract adapters to apply.
        processing_mode (TextractProcessingMode, optional): Whether to use the synchronous API, the
            asynchronous (S3-based) API, or pick automatically. Defaults to "async".
        min_confidence (float, optional): Drop text blocks below this Textract confidence (0-100).
        skip_headers_and_footers (bool, optional): Drop layout headers, footers and page numbers.
            Requires the `LAYOUT` feature. Defaults to False.
        Local documents sent asynchronously are staged in an internal per-account, per-region
        bucket managed by Dynamiq. The bucket name and retention settings are intentionally not
        configurable. Staged objects are deleted after processing, with a one-day lifecycle policy
        as a backstop for interrupted runs.

    Example:
        ```python
        from dynamiq.components.converters.textract import TextractFeatureType
        from dynamiq.nodes.converters import TextractFileConverter

        converter = TextractFileConverter(
            feature_types=[TextractFeatureType.LAYOUT, TextractFeatureType.TABLES],
        )
        result = converter.run(input_data={"file_paths": ["scanned-invoice.pdf"]})
        ```
    """

    group: Literal[NodeGroup.CONVERTERS] = NodeGroup.CONVERTERS
    name: str = "textract-file-converter"
    description: str = (
        "Converter that extracts text, tables, forms and query answers from scanned documents "
        "and images using Amazon Textract OCR."
    )
    error_handling: ErrorHandling = Field(
        default_factory=lambda: ErrorHandling(timeout_seconds=900.0),
        description="Default execution timeout. Asynchronous Textract jobs on long PDFs can run for "
        "minutes, so this is higher than the other converters. Set timeout_seconds to None to disable.",
    )
    connection: AWSTextract | None = None
    document_creation_mode: Literal[DocumentCreationMode.ONE_DOC_PER_FILE, DocumentCreationMode.ONE_DOC_PER_PAGE] = (
        DocumentCreationMode.ONE_DOC_PER_FILE
    )
    feature_types: list[TextractFeatureType] = Field(default_factory=list)
    queries: list[TextractQuery] = Field(default_factory=list)
    adapters: list[TextractAdapter] = Field(default_factory=list)
    processing_mode: TextractProcessingMode = TextractProcessingMode.ASYNC
    separator: str = "\n\n"

    render_tables_as_markdown: bool = True
    include_form_section: bool = True
    include_query_section: bool = True
    skip_headers_and_footers: bool = False
    min_confidence: float | None = Field(default=None, ge=0, le=100)

    notification_channel: TextractNotificationChannel | None = None
    kms_key_id: str | None = None
    job_tag: str | None = None
    poll_interval_seconds: float = Field(default=2.0, gt=0)
    max_poll_seconds: float = Field(default=900.0, gt=0)

    file_converter: TextractFileConverterComponent | None = None
    input_schema: ClassVar[type[TextractFileConverterInputSchema]] = TextractFileConverterInputSchema

    def __init__(self, **kwargs):
        """
        Initialize the TextractFileConverter.

        If neither a connection nor a client is provided, a new AWSTextract connection is created
        from the ambient AWS credentials.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = AWSTextract()
        super().__init__(**kwargs)

    @property
    def to_dict_exclude_params(self):
        return super().to_dict_exclude_params | {"file_converter": True}

    def init_components(self, connection_manager: ConnectionManager | None = None):
        """
        Initialize the components of the TextractFileConverter.

        Args:
            connection_manager (ConnectionManager, optional): The connection manager to use.
                Defaults to a new ConnectionManager instance.
        """
        connection_manager = connection_manager or ConnectionManager()
        super().init_components(connection_manager)
        if self.file_converter is None:
            self.file_converter = TextractFileConverterComponent(
                connection=self.connection,
                client=self.client,
                document_creation_mode=self.document_creation_mode,
                feature_types=self.feature_types,
                queries=self.queries,
                adapters=self.adapters,
                processing_mode=self.processing_mode,
                separator=self.separator,
                render_tables_as_markdown=self.render_tables_as_markdown,
                include_form_section=self.include_form_section,
                include_query_section=self.include_query_section,
                skip_headers_and_footers=self.skip_headers_and_footers,
                min_confidence=self.min_confidence,
                notification_channel=self.notification_channel,
                kms_key_id=self.kms_key_id,
                job_tag=self.job_tag,
                poll_interval_seconds=self.poll_interval_seconds,
                max_poll_seconds=self.max_poll_seconds,
            )

    def execute(
        self, input_data: TextractFileConverterInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, list[Any]]:
        """
        Execute the TextractFileConverter to convert documents to Documents.

        Args:
            input_data (TextractFileConverterInputSchema): An instance containing 'file_paths',
                'files', 's3_objects' and/or 'metadata'.
            config (RunnableConfig): Optional configuration for the execution.
            **kwargs: Additional keyword arguments.

        Returns:
            dict[str, list[Any]]: Dictionary with 'documents' key containing a list of converted Documents.

        Example:
            input_data = {
                "file_paths": ["/path/to/scan.pdf"],
                "s3_objects": [{"bucket": "my-bucket", "name": "contracts/lease.pdf"}],
                "metadata": {"source": "user_upload"},
            }
        """
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        # ensure_client may have swapped in a fresh client since init_components.
        self.file_converter.client = self.client

        file_paths = input_data.file_paths
        files = input_data.files
        s3_objects = input_data.s3_objects
        metadata = input_data.metadata

        output = self.file_converter.run(
            file_paths=file_paths,
            files=files,
            metadata=metadata,
            s3_objects=s3_objects,
        )
        documents = output["documents"]

        count_file_paths = len(file_paths) if file_paths else 0
        count_files = len(files) if files else 0
        count_s3_objects = len(s3_objects) if s3_objects else 0

        logger.debug(
            f"Converted {count_file_paths} file paths, {count_files} file objects and "
            f"{count_s3_objects} S3 objects to {len(documents)} Documents."
        )

        return {"documents": documents}
