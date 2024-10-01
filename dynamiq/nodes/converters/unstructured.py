from typing import Any, Literal

from dynamiq.components.converters.unstructured import (
    ConvertStrategy,
    DocumentCreationMode,
)
from dynamiq.components.converters.unstructured import (
    UnstructuredFileConverter as UnstructuredFileConverterComponent,
)
from dynamiq.connections import Unstructured
from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes.node import ConnectionNode, NodeGroup, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger


class UnstructuredFileConverter(ConnectionNode):
    """
    A component for converting files to Documents using the Unstructured API (hosted or running locally).

    Args:
        connection (UnstructuredConnection, optional): The connection to use for the Unstructured API.
            Defaults to None, which will initialize a new UnstructuredConnection.
        document_creation_mode (Literal["one-doc-per-file", "one-doc-per-page", "one-doc-per-element"],
            optional): Determines how to create Documents from the elements returned by Unstructured.
            Options are:
            - "one-doc-per-file": Creates one Document per file.
                All elements are concatenated into one text field.
            - "one-doc-per-page": Creates one Document per page.
                All elements on a page are concatenated into one text field.
            - "one-doc-per-element": Creates one Document per element.
                Each element is converted to a separate Document.
            Defaults to "one-doc-per-file".
        strategy (Literal["auto", "fast", "hi_res", "ocr_only"], optional): The strategy to use for
            document processing. Defaults to "auto".
        unstructured_kwargs (Optional[dict[str, Any]], optional): Additional parameters to pass to the
            Unstructured API. See Unstructured API docs for available parameters. Defaults to None.
    """

    group: Literal[NodeGroup.CONVERTERS] = NodeGroup.CONVERTERS
    name: str = "Unstructured File Converter"
    connection: Unstructured = None
    document_creation_mode: DocumentCreationMode = DocumentCreationMode.ONE_DOC_PER_FILE
    strategy: ConvertStrategy = ConvertStrategy.AUTO
    unstructured_kwargs: dict[str, Any] | None = None
    file_converter: UnstructuredFileConverterComponent = None

    def __init__(self, **kwargs):
        """
        Initialize the UnstructuredFileConverter.

        If no connection is provided, a new Unstructured connection will be created.

        Args:
            **kwargs: Keyword arguments to initialize the UnstructuredFileConverter.
        """
        if kwargs.get("connection") is None:
            kwargs["connection"] = Unstructured()
        super().__init__(**kwargs)

    @property
    def to_dict_exclude_params(self):
        return super().to_dict_exclude_params | {"file_converter": True}

    def init_components(
        self, connection_manager: ConnectionManager = ConnectionManager()
    ):
        """
        Initialize the components of the UnstructuredFileConverter.

        Args:
            connection_manager (ConnectionManager, optional): The connection manager to use.
                Defaults to a new ConnectionManager instance.
        """
        super().init_components(connection_manager)
        if self.file_converter is None:
            self.file_converter = UnstructuredFileConverterComponent(
                connection=self.connection,
                document_creation_mode=self.document_creation_mode,
                strategy=self.strategy,
                unstructured_kwargs=self.unstructured_kwargs,
            )

    def execute(
        self, input_data: dict[str, Any], config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, list[Any]]:
        """
        Execute the UnstructuredFileConverter to convert files to Documents.

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
                "file_paths": ["/path/to/file1.pdf", "/path/to/file2.docx"],
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
            f"Converted {count_file_paths} file paths and {count_files} file objects "
            f"to {len(documents)} Documents."
        )

        return {"documents": documents}
