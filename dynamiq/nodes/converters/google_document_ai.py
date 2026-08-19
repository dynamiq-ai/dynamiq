from io import BytesIO
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dynamiq.components.converters.google_document_ai import DocumentAICreationMode, DocumentCreationMode
from dynamiq.components.converters.google_document_ai import (
    GoogleDocumentAIFileConverter as GoogleDocumentAIFileConverterComponent,
)
from dynamiq.components.converters.google_document_ai import ProcessorId
from dynamiq.connections import GoogleDocumentAI
from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes.node import ConnectionNode, ErrorHandling, NodeGroup, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger


class GoogleDocumentAIFileConverterInputSchema(BaseModel):
    file_paths: list[str] = Field(default=None, description="Parameter to provide path to files.")
    files: list[BytesIO | bytes] = Field(default=None, description="Parameter to provide files.")
    metadata: dict | list = Field(default=None, description="Parameter to provide metadata.")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def validate_file_source(self):
        """Validate that either `file_paths` or `files` is specified"""
        if not self.file_paths and not self.files:
            raise ValueError("Either `file_paths` or `files` must be provided.")
        return self


class GoogleDocumentAIFileConverter(ConnectionNode):
    """
    A node that extracts plain text from documents using Google Document AI.

    Only the text layer of the processed document is kept - tables, form fields, entities and
    layout blocks are deliberately ignored. Use a splitter downstream to chunk the result.

    PDFs longer than Document AI's online page limit are split locally and processed in
    consecutive batches, whose texts are then concatenated. Images are sent as-is; any other
    file type is rejected before the request is made.

    Args:
        connection (GoogleDocumentAI, optional): The connection to use for the Document AI API.
            Defaults to a new GoogleDocumentAI connection, which reads its settings from the
            environment.
        processor_id (str): The id of the Document AI processor to run, e.g. '1e0795e517c827d1'.
        processor_version (str | None, optional): An optional pinned processor version, e.g.
            'pretrained-ocr-v2.0-2023-06-02'. Defaults to None, which runs the processor's
            default version.
        document_creation_mode (DocumentCreationMode, optional): Determines how to create Documents
            from the processed result. Options are:
            - ONE_DOC_PER_FILE: Creates one Document per file, with all page texts concatenated.
            - ONE_DOC_PER_PAGE: Creates one Document per page.
            Defaults to ONE_DOC_PER_FILE.
        enable_native_pdf_parsing (bool, optional): Whether to read embedded text from digital PDFs
            instead of OCR-ing their rendered pages. Defaults to True.
        imageless_mode (bool, optional): Whether to omit page images from the response. Halves the
            payload and doubles the online page limit to 30. Defaults to True.
    """

    group: Literal[NodeGroup.CONVERTERS] = NodeGroup.CONVERTERS
    name: str = "google-document-ai-file-converter"
    description: str = "Extract text from documents using Google Document AI."
    error_handling: ErrorHandling = Field(
        default_factory=lambda: ErrorHandling(timeout_seconds=300.0),
        description="Default execution timeout. Set timeout_seconds to None to disable.",
    )
    connection: GoogleDocumentAI = Field(
        default_factory=GoogleDocumentAI,
        description="Connection to the Document AI API. Supplies the project id and location.",
    )
    processor_id: ProcessorId = Field(description="The id of the Document AI processor to run.")
    processor_version: str | None = Field(
        default=None,
        description="Optional pinned processor version. Defaults to the processor's default version.",
    )
    document_creation_mode: DocumentAICreationMode = Field(
        default=DocumentCreationMode.ONE_DOC_PER_FILE,
        description="Create one document per file, with page texts joined, or one per page.",
    )
    enable_native_pdf_parsing: bool = Field(
        default=True,
        description="Read embedded text from born-digital PDFs instead of OCR-ing rendered pages.",
    )
    imageless_mode: bool = Field(
        default=True,
        description="Omit page images from the response, which also doubles the online page limit.",
    )
    file_converter: GoogleDocumentAIFileConverterComponent | None = None
    input_schema: ClassVar[type[GoogleDocumentAIFileConverterInputSchema]] = GoogleDocumentAIFileConverterInputSchema

    @property
    def to_dict_exclude_params(self):
        return super().to_dict_exclude_params | {"file_converter": True}

    def init_components(self, connection_manager: ConnectionManager | None = None):
        """
        Initialize the components of the GoogleDocumentAIFileConverter.

        Args:
            connection_manager (ConnectionManager, optional): The connection manager to use.
                Defaults to a new ConnectionManager instance.
        """
        connection_manager = connection_manager or ConnectionManager()
        super().init_components(connection_manager)
        if self.file_converter is None:
            self.file_converter = GoogleDocumentAIFileConverterComponent(
                connection=self.connection,
                client=self.client,
                processor_id=self.processor_id,
                processor_version=self.processor_version,
                document_creation_mode=self.document_creation_mode,
                enable_native_pdf_parsing=self.enable_native_pdf_parsing,
                imageless_mode=self.imageless_mode,
            )

    def execute(
        self, input_data: GoogleDocumentAIFileConverterInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, list[Any]]:
        """
        Execute the GoogleDocumentAIFileConverter to convert files to Documents.

        Args:
            input_data (GoogleDocumentAIFileConverterInputSchema): An instance containing
                'file_paths', 'files', and/or 'metadata'.
            config (RunnableConfig): Optional configuration for the execution.
            **kwargs: Additional keyword arguments.

        Returns:
            dict[str, list[Any]]: Dictionary with 'documents' key containing a list of converted Documents.

        Example:
            input_data = {
                "file_paths": ["/path/to/file1.pdf"],
                "files": [BytesIO(b"file content")],
                "metadata": {"source": "user_upload"}
            }
        """
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        file_paths = input_data.file_paths
        files = input_data.files
        metadata = input_data.metadata

        output = self.file_converter.run(file_paths=file_paths, files=files, metadata=metadata)
        documents = output["documents"]

        count_file_paths = len(file_paths) if file_paths else 0
        count_files = len(files) if files else 0

        logger.debug(
            f"Converted {count_file_paths} file paths and {count_files} file objects " f"to {len(documents)} Documents."
        )

        return {"documents": documents}
