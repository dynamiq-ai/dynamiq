from typing import Any, Literal

from dynamiq.components.converters.pptx import PPTXConverter as PPTXConverterComponent
from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes.node import Node, NodeGroup, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.types import DocumentCreationMode
from dynamiq.utils.logger import logger


class PPTXFileConverter(Node):
    """
    A component for converting files to Documents using the pptx converter.

    Args:
        document_creation_mode (Literal["one-doc-per-file", "one-doc-per-page", "one-doc-per-element"],
            optional): Determines how to create Documents from the elements returned by PdfReader.
            Options are:
            - "one-doc-per-file": Creates one Document per file.
                All elements are concatenated into one text field.
            - "one-doc-per-page": Creates one Document per page.
                All elements on a page are concatenated into one text field.
            Defaults to "one-doc-per-file".
    """

    group: Literal[NodeGroup.CONVERTERS] = NodeGroup.CONVERTERS
    name: str = "PPTX File Converter"
    document_creation_mode: DocumentCreationMode = DocumentCreationMode.ONE_DOC_PER_FILE
    file_converter: PPTXConverterComponent = None

    @property
    def to_dict_exclude_params(self):
        return super().to_dict_exclude_params | {"file_converter": True}

    def init_components(self, connection_manager: ConnectionManager = ConnectionManager()):
        """
        Initialize the components of the PPTXConverter.

        Args:
            connection_manager (ConnectionManager, optional): The connection manager to use.
                Defaults to a new ConnectionManager instance.
        """
        super().init_components(connection_manager)
        if self.file_converter is None:
            self.file_converter = PPTXConverterComponent(
                document_creation_mode=self.document_creation_mode,
            )

    def execute(
        self, input_data: dict[str, Any], config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, list[Any]]:
        """
        Execute the PPTXConverter to convert files to Documents.

        Args:
            input_data: Dict containing 'file_paths', 'files', and/or 'metadata' keys.
            config: Optional configuration for the execution.
            **kwargs: Additional keyword arguments.

        Returns:
            Dict with 'documents' key containing a list of converted Documents.

        Raises:
            KeyError: If required keys are missing in input_data.

        Example:
            input_data = {
                "file_paths": ["/path/to/file1.pptx"],
                "files": [BytesIO(b"file content")],
                "metadata": {"source": "user_upload"}
            }
        """
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        file_paths = input_data.get("file_paths")
        files = input_data.get("files")
        metadata = input_data.get("metadata")

        output = self.file_converter.run(file_paths=file_paths, files=files, metadata=metadata)
        documents = output["documents"]

        count_file_paths = len(file_paths) if file_paths else 0
        count_files = len(files) if files else 0

        logger.debug(
            f"Converted {count_file_paths} file paths and {count_files} file objects " f"to {len(documents)} Documents."
        )

        return {"documents": documents}
