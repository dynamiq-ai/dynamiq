import copy
import logging
from io import BytesIO
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes.converters.docx import DOCXFileConverter
from dynamiq.nodes.converters.html import HTMLConverter
from dynamiq.nodes.converters.pptx import PPTXFileConverter
from dynamiq.nodes.converters.pypdf import PyPDFConverter
from dynamiq.nodes.converters.text import TextFileConverter
from dynamiq.nodes.extractors.extractors import EXTENSION_MAP, FileType, FileTypeExtractor, FileTypeExtractorInputSchema
from dynamiq.nodes.node import Node, NodeDependency, ensure_config
from dynamiq.nodes.types import NodeGroup
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.types import Document
from dynamiq.types.cancellation import check_cancellation

logger = logging.getLogger(__name__)


DEFAULT_FILE_TYPE_TO_CONVERTER_CLASS_MAP = {
    FileType.PDF: PyPDFConverter,
    FileType.DOCUMENT: DOCXFileConverter,
    FileType.PRESENTATION: PPTXFileConverter,
    FileType.HTML: HTMLConverter,
    FileType.TEXT: TextFileConverter,
    FileType.MARKDOWN: TextFileConverter,
}


# TODO: Add parallel processing for multiple files.
class MultiFileTypeConverterInputSchema(BaseModel):
    """Schema for MultiFileConverter input data."""

    file_paths: list[str] = Field(default=None, description="Parameter to provide path to files.")
    files: list[BytesIO] = Field(default=None, description="Parameter to provide files.")
    metadata: dict | list[dict] = Field(default=None, description="Parameter to provide metadata.")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def validate_file_source(self):
        """Validate that either `file_paths` or `files` is specified"""
        if not self.file_paths and not self.files:
            raise ValueError("Either `file_paths` or `files` must be provided.")
        return self


class MultiFileTypeConverter(Node):
    """
    Multi-file-type document converter that routes documents to appropriate converters based on file type.

    This component uses the FileTypeExtractor to determine the document type and then
    routes it to the appropriate available converter:
        - PyPDFConverter
        - DOCXFileConverter
        - PPTXFileConverter
        - HTMLConverter
        - UnstructuredFileConverter (fallback)
    """

    group: Literal[NodeGroup.CONVERTERS] = NodeGroup.CONVERTERS
    name: str = "multi-file-type-converter"
    description: str = "Meta converter that routes documents to appropriate converters based on file type."
    fallback_converter: Node | None = Field(
        default=None,
        description="Fallback converter to use for unsupported file types. Defaults to UnstructuredFileConverter.",
    )
    file_type_extractor: FileTypeExtractor | None = None
    converters: list[Node] | None = None
    converter_mapping: dict[FileType, Node] | None = None
    model_config = ConfigDict(arbitrary_types_allowed=True)
    input_schema: ClassVar[type[MultiFileTypeConverterInputSchema]] = MultiFileTypeConverterInputSchema

    def __init__(self, **kwargs):
        """
        Initializes the MultiFileTypeConverter with the given parameters and creates a default fallback converter.

        Args:
            **kwargs: Additional keyword arguments to be passed to the parent class constructor.
        """
        super().__init__(**kwargs)
        self._run_depends = []

    def reset_run_state(self):
        """
        Reset the intermediate steps (run_depends) of the node.
        """
        self._run_depends = []

    @property
    def to_dict_exclude_params(self):
        """
        Property to define which parameters should be excluded when converting the class instance to a dictionary.

        Returns:
            dict: A dictionary defining the parameters to exclude.
        """
        return super().to_dict_exclude_params | {
            "fallback_converter": True,
            "file_type_extractor": True,
            "converters": True,
            "converter_mapping": True,
        }

    def to_dict(self, **kwargs) -> dict:
        """Converts the instance to a dictionary.

        Returns:
            dict: A dictionary representation of the instance.
        """
        data = super().to_dict(**kwargs)
        if self.fallback_converter:
            data["fallback_converter"] = self.fallback_converter.to_dict(**kwargs)
        if self.file_type_extractor:
            data["file_type_extractor"] = self.file_type_extractor.to_dict(**kwargs)
        if self.converters:
            data["converters"] = [converter.to_dict(**kwargs) for converter in self.converters]
        return data

    def init_components(self, connection_manager: ConnectionManager | None = None):
        """
        Initialize the components of the MultiFileTypeConverter.

        Args:
            connection_manager (ConnectionManager, optional): The connection manager to use.
                Defaults to a new ConnectionManager instance.
        """
        connection_manager = connection_manager or ConnectionManager()
        super().init_components(connection_manager)
        if self.file_type_extractor is None:
            self.file_type_extractor = FileTypeExtractor()

        if self.file_type_extractor.is_postponed_component_init:
            self.file_type_extractor.init_components(connection_manager)

        self._setup_converters()

        # Initialize components for converters in the mapping
        initialized_converters = set()
        for converter in self.converter_mapping.values():
            if id(converter) not in initialized_converters:
                if converter.is_postponed_component_init:
                    converter.init_components(connection_manager)
                initialized_converters.add(id(converter))
                logger.info(f"Initialized converter: {converter.name}")

    def _setup_converters(self):
        """Setup internal converter components."""

        if not self.converters:
            # Create default converter instances
            self.converter_mapping = {
                file_type: converter_class()
                for file_type, converter_class in DEFAULT_FILE_TYPE_TO_CONVERTER_CLASS_MAP.items()
            }
        else:
            self._add_file_type_mapping(self.converters)

    def _add_file_type_mapping(self, converter_instances: list[Node]):
        """Create mapping from file types to converter instances from a list of converters."""

        self.converter_mapping = {}

        # Map each converter instance to its supported file types
        for converter_instance in converter_instances:
            converter_class = converter_instance.__class__

            # Find file types that the converter class supports
            supported_file_types = [
                file_type
                for file_type, default_converter_class in DEFAULT_FILE_TYPE_TO_CONVERTER_CLASS_MAP.items()
                if converter_class == default_converter_class
            ]

            for file_type in supported_file_types:
                if file_type not in self.converter_mapping:
                    self.converter_mapping[file_type] = converter_instance

        # Add default converters for missing file types
        self._add_missing_default_converters()

        # Add fallback converter for remaining unmapped file types
        self._add_fallback_mapping()

    def _add_missing_default_converters(self):
        """Instantiate default converters for standard file types that don't have converters yet."""

        # Check which file types are missing converters
        for file_type, converter_class in DEFAULT_FILE_TYPE_TO_CONVERTER_CLASS_MAP.items():
            if file_type not in self.converter_mapping:
                # Create the default converter for the file type
                default_converter = converter_class()
                self.converter_mapping[file_type] = default_converter

    def _add_fallback_mapping(self):
        """Setup fallback converter for file types that don't have specific converters from instances list."""

        # Find file types that don't have converters
        file_types_without_converters = [
            file_type for file_type in EXTENSION_MAP.keys() if file_type not in self.converter_mapping
        ]

        if file_types_without_converters:
            if self.fallback_converter:
                for file_type in file_types_without_converters:
                    self.converter_mapping[file_type] = self.fallback_converter

    def execute(
        self, input_data: MultiFileTypeConverterInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        """
        Convert documents by routing to appropriate converter based on file type.

        Args:
            input_data (MultiFileTypeConverterInputSchema): Input data containing files to convert.
            config (RunnableConfig, optional): Configuration for the runnable, including callbacks.
            **kwargs: Additional arguments passed to the execution context.

        Returns:
            dict[str, Any]: A dictionary containing the converted documents.
        """
        config = ensure_config(config)
        self.reset_run_state()
        self.run_on_node_execute_run(config.callbacks, **kwargs)
        kwargs = kwargs | {"parent_run_id": kwargs.get("run_id")}

        documents = self.process_files(
            file_paths=input_data.file_paths,
            files=input_data.files,
            metadata=input_data.metadata,
            config=config,
            **kwargs,
        )

        return {"documents": documents}

    def process_files(
        self,
        file_paths: list[str] | None = None,
        files: list[BytesIO] | None = None,
        metadata: dict[str, Any] | list[dict[str, Any]] | None = None,
        config: RunnableConfig = None,
        **kwargs,
    ) -> list[Document]:
        """
        Process the files.
        """

        all_documents = []

        if file_paths is not None:
            paths_obj = [Path(path) for path in file_paths]
            filepaths = [path for path in paths_obj if path.is_file()]
            filepaths_in_directories = [
                filepath for path in paths_obj if path.is_dir() for filepath in path.glob("*.*") if filepath.is_file()
            ]
            if filepaths_in_directories and isinstance(metadata, list):
                raise ValueError(
                    "If providing directories in the `file_paths` parameter, "
                    "`metadata` can only be a dictionary (metadata applied to every file), "
                    "and not a list. To specify different metadata for each file, "
                    "provide an explicit list of direct paths instead."
                )

            all_filepaths = list(set(filepaths + filepaths_in_directories))

            if not all_filepaths:
                raise FileNotFoundError(f"No files found in the provided paths: {file_paths}")

            meta_list = self._normalize_metadata(metadata, len(all_filepaths))

            for filepath, meta in zip(all_filepaths, meta_list):
                check_cancellation(config)
                try:
                    with open(filepath, "rb") as f:
                        file_content = BytesIO(f.read())

                    filename = filepath.name
                    documents = self._process_single_file(file_content, filename, meta, config, **kwargs)
                    all_documents.extend(documents)
                except Exception as e:
                    logger.error(f"Failed to process file {filepath}: {str(e)}")
                    raise

        if files is not None:
            meta_list = self._normalize_metadata(metadata, len(files))
            for i, (file, meta) in enumerate(zip(files, meta_list)):
                check_cancellation(config)
                try:
                    filename = meta.get("filename", f"file_{i}")
                    documents = self._process_single_file(file, filename, meta, config, **kwargs)
                    all_documents.extend(documents)
                except Exception as e:
                    logger.error(f"Failed to process file {i}: {str(e)}")
                    raise

        if len(all_documents) == 0:
            raise ValueError(
                "No documents were created from the provided inputs. Please check your files and try again."
            )

        count_file_paths = len(file_paths) if file_paths else 0
        count_files = len(files) if files else 0

        logger.debug(
            f"Converted {count_file_paths} file paths and {count_files} file objects "
            f"to {len(all_documents)} Documents."
        )

        return all_documents

    def call_file_type_extractor(
        self, input_data: FileTypeExtractorInputSchema, config: RunnableConfig, **run_kwargs
    ) -> dict:
        """
        Call the file type extractor.
        """
        file_type_extractor_result = self.file_type_extractor.run(
            input_data=input_data, config=config, run_depends=self._run_depends, **run_kwargs
        )
        self._run_depends = [NodeDependency(node=self.file_type_extractor).to_dict(for_tracing=True)]

        if file_type_extractor_result.status != RunnableStatus.SUCCESS:
            logger.error(
                f"Node {self.name} - {self.id}: FileTypeExtractor execution failed: "
                f"{file_type_extractor_result.error.to_dict()}"
            )
            raise ValueError("FileTypeExtractor execution failed")
        return file_type_extractor_result.output

    def call_converter(self, input_data: dict, detected_type: FileType, config: RunnableConfig, **run_kwargs) -> dict:
        """
        Call the converter.
        """
        converter_result = self.converter_mapping[detected_type].run(
            input_data=input_data, config=config, run_depends=self._run_depends, **run_kwargs
        )
        self._run_depends = [NodeDependency(node=self.converter_mapping[detected_type]).to_dict(for_tracing=True)]

        converter_name = self.converter_mapping[detected_type].name

        if converter_result.status != RunnableStatus.SUCCESS:
            logger.error(
                f"Node {self.name} - {self.id}: {converter_name} execution failed: "
                f"{converter_result.error.to_dict()}"
            )
            raise ValueError(f"{converter_name} execution failed")
        return converter_result.output

    def call_fallback_converter(self, input_data: dict, config: RunnableConfig, **run_kwargs) -> dict:
        """
        Call the fallback converter.
        """
        fallback_converter_result = self.fallback_converter.run(
            input_data=input_data, config=config, run_depends=self._run_depends, **run_kwargs
        )
        self._run_depends = [NodeDependency(node=self.fallback_converter).to_dict(for_tracing=True)]

        fallback_converter_name = self.fallback_converter.name

        if fallback_converter_result.status != RunnableStatus.SUCCESS:
            logger.error(
                f"Node {self.name} - {self.id}: {fallback_converter_name} execution failed: "
                f"{fallback_converter_result.error.to_dict()}"
            )
            raise ValueError(f"{fallback_converter_name} execution failed")
        return fallback_converter_result.output

    def _process_single_file(
        self, file: BytesIO, filename: str, metadata: dict, config: RunnableConfig, **kwargs
    ) -> list:
        """
        Process a single file by routing to appropriate converter based on file type.

        Args:
            file: The file to convert
            filename: The filename for file type detection
            metadata: Metadata for the file
            config: Runtime configuration
            **kwargs: Additional arguments

        Returns:
            list: List of documents from the conversion
        """
        try:
            file_type_extractor_result = self.call_file_type_extractor(
                input_data={"file": file, "filename": filename}, config=config, **kwargs
            )

            detected_type = file_type_extractor_result.get("type")
            logger.info(f"Detected file type: {detected_type} for file: {filename}")

            if detected_type and detected_type in self.converter_mapping:
                converter_name = self.converter_mapping[detected_type].name

                try:
                    if not hasattr(file, "name"):
                        file.name = filename

                    converter_input = {"files": [file]}
                    if metadata:
                        converter_input["metadata"] = [metadata]

                    result = self.call_converter(
                        input_data=converter_input, detected_type=detected_type, config=config, **kwargs
                    )
                    logger.info(f"Successfully converted using {converter_name}")
                    return result.get("documents", [])

                except Exception as e:
                    logger.warning(f"Failed to convert with {converter_name}: {str(e)}")
                    if self.fallback_converter:
                        return self._convert_with_fallback_converter(file, filename, metadata, config, **kwargs)
                    else:
                        raise

            elif self.fallback_converter:
                return self._convert_with_fallback_converter(file, filename, metadata, config, **kwargs)

            else:
                raise ValueError(f"Unsupported file type: {detected_type}")

        except Exception as e:
            error_msg = f"Failed to convert document {filename}: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _convert_with_fallback_converter(
        self, file: BytesIO, filename: str, metadata: dict, config: RunnableConfig, **kwargs
    ) -> list:
        """
        Try to convert using the fallback converter.

        Args:
            file: The file to convert
            filename: The filename
            metadata: Metadata for the file
            config: Runtime configuration
            **kwargs: Additional arguments

        Returns:
            list: List of documents from the conversion
        """
        try:
            logger.info(f"Attempting conversion with fallback converter for {filename}")

            converter_input = {"files": [file]}
            if metadata:
                converter_input["metadata"] = [metadata]

            result = self.call_fallback_converter(input_data=converter_input, config=config, **kwargs)

            logger.info(
                f"Successfully converted using fallback converter: {self.fallback_converter.__class__.__name__}"
            )
            return result.get("documents", [])

        except Exception as e:
            logger.error(f"Fallback converter also failed: {str(e)}")
            raise ValueError(f"All conversion attempts failed. Last error: {str(e)}")

    @staticmethod
    def _normalize_metadata(
        metadata: dict[str, Any] | list[dict[str, Any]] | None, sources_count: int
    ) -> list[dict[str, Any]]:
        """Normalizes metadata input for a converter.

        Given all possible values of the metadata input for a converter (None, dictionary, or list of
        dicts), ensures to return a list of dictionaries of the correct length for the converter to use.

        Args:
            metadata: The meta input of the converter, as-is. Can be None, a dictionary, or a list of
                dictionaries.
            sources_count: The number of sources the converter received.

        Returns:
            A list of dictionaries of the same length as the sources list.

        Raises:
            ValueError: If metadata is not None, a dictionary, or a list of dictionaries, or if the length
                of the metadata list doesn't match the number of sources.
        """
        if metadata is None:
            return [{} for _ in range(sources_count)]
        if isinstance(metadata, dict):
            return [copy.deepcopy(metadata) for _ in range(sources_count)]
        if isinstance(metadata, list):
            metadata_count = len(metadata)
            if sources_count != metadata_count:
                raise ValueError(
                    f"The length of the metadata list [{metadata_count}] "
                    f"must match the number of sources [{sources_count}]."
                )
            return metadata
        raise ValueError("metadata must be either None, a dictionary or a list of dictionaries.")
