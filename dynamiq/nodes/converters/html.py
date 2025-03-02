from io import BytesIO
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dynamiq.components.converters.html import HTMLConverter as HTMLConverterComponent
from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes.node import Node, NodeGroup, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.types import DocumentCreationMode
from dynamiq.utils.logger import logger


class HTMLConverterInputSchema(BaseModel):
    file_paths: list[str] = Field(default=None, description="Parameter to provide path to files.")
    files: list[BytesIO] = Field(default=None, description="Parameter to provide files.")
    metadata: dict | list = Field(default=None, description="Parameter to provide metadata.")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def validate_file_source(self):
        """Validate that either `file_paths` or `files` is specified"""
        if not self.file_paths and not self.files:
            raise ValueError("Either `file_paths` or `files` must be provided.")
        return self


class HTMLConverter(Node):
    """
    A component for converting HTML files to Documents using the HTML converter.

    Args:
        document_creation_mode (Literal["one-doc-per-file"], optional):
            Determines how to create Documents from the HTML content. Currently only supports:
            - "one-doc-per-file": Creates one Document per file.
                All content is converted to markdown format.
            Defaults to "one-doc-per-file".
    """

    group: Literal[NodeGroup.CONVERTERS] = NodeGroup.CONVERTERS
    name: str = "HTML File Converter"
    document_creation_mode: DocumentCreationMode = DocumentCreationMode.ONE_DOC_PER_FILE
    file_converter: HTMLConverterComponent | None = None
    input_schema: ClassVar[type[HTMLConverterInputSchema]] = HTMLConverterInputSchema

    @property
    def to_dict_exclude_params(self):
        return super().to_dict_exclude_params | {"file_converter": True}

    def init_components(self, connection_manager: ConnectionManager | None = None):
        """
        Initialize the components of the HTMLConverter.

        Args:
            connection_manager (ConnectionManager, optional): The connection manager to use.
                Defaults to a new ConnectionManager instance.
        """
        connection_manager = connection_manager or ConnectionManager()
        super().init_components(connection_manager)
        if self.file_converter is None:
            self.file_converter = HTMLConverterComponent(
                document_creation_mode=self.document_creation_mode,
            )

    def execute(
        self, input_data: HTMLConverterInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, list[Any]]:
        """
        Execute the HTMLConverter to convert files to Documents.

        Args:
            input_data (HTMLConverterInputSchema): An instance containing 'file_paths', 'files', and/or 'metadata'.
            config (RunnableConfig): Optional configuration for the execution.
            **kwargs: Additional keyword arguments.

        Returns:
            Dict with 'documents' key containing a list of converted Documents.

        Raises:
            KeyError: If required keys are missing in input_data.

        Example:
            input_data = {
                "file_paths": ["/path/to/file1.html"],
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
