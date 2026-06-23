import copy
import logging
from concurrent.futures import FIRST_EXCEPTION, wait
from io import BytesIO
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dynamiq.connections.managers import ConnectionManager
from dynamiq.executors.context import ContextAwareThreadPoolExecutor
from dynamiq.nodes.converters.docx import DOCXFileConverter
from dynamiq.nodes.converters.html import HTMLConverter
from dynamiq.nodes.converters.llm_text_extractor import LLMImageConverter, LLMPDFConverter
from dynamiq.nodes.converters.pptx import PPTXFileConverter
from dynamiq.nodes.converters.pypdf import PyPDFConverter
from dynamiq.nodes.converters.text import TextFileConverter
from dynamiq.nodes.extractors.extractors import EXTENSION_MAP, FileType, FileTypeExtractor, FileTypeExtractorInputSchema
from dynamiq.nodes.node import ErrorHandling, Node, NodeDependency, ensure_config
from dynamiq.nodes.types import NodeGroup
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.types import Document
from dynamiq.types.cancellation import CanceledException, check_cancellation

logger = logging.getLogger(__name__)


DEFAULT_FILE_TYPE_TO_CONVERTER_CLASS_MAP = {
    FileType.PDF: PyPDFConverter,
    FileType.DOCUMENT: DOCXFileConverter,
    FileType.PRESENTATION: PPTXFileConverter,
    FileType.HTML: HTMLConverter,
    FileType.TEXT: TextFileConverter,
    FileType.MARKDOWN: TextFileConverter,
}

FILE_TYPE_TO_SUPPORTED_CONVERTER_CLASS_MAP = {
    FileType.PDF: (PyPDFConverter, LLMPDFConverter),
    FileType.IMAGE: (LLMImageConverter,),
    FileType.DOCUMENT: (DOCXFileConverter,),
    FileType.PRESENTATION: (PPTXFileConverter,),
    FileType.HTML: (HTMLConverter,),
    FileType.TEXT: (TextFileConverter,),
    FileType.MARKDOWN: (TextFileConverter,),
}

DEFAULT_TIMEOUT_SECONDS = 600.0

# How often the main thread polls for cancellation while parallel conversions run.
_CANCELLATION_POLL_INTERVAL = 0.05


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
        - LLMImageConverter
        - PyPDFConverter
        - LLMPDFConverter
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
    error_handling: ErrorHandling = Field(
        default_factory=lambda: ErrorHandling(timeout_seconds=DEFAULT_TIMEOUT_SECONDS),
        description="Overall execution timeout for the whole batch. Sub-converters carry their own "
        "per-file timeouts. Set timeout_seconds to None to disable.",
    )
    max_workers: int | None = Field(
        default=None,
        description="Maximum number of files to convert concurrently. Defaults to None, which sizes the "
        "thread pool to the number of input files. When the surrounding RunnableConfig sets "
        "max_node_workers, that value is used as an upper bound.",
    )
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
            supported_file_types = self._get_supported_file_types(converter_instance)

            for file_type in supported_file_types:
                if file_type not in self.converter_mapping:
                    self.converter_mapping[file_type] = converter_instance

        # Add default converters for missing file types
        self._add_missing_default_converters()

        # Add fallback converter for remaining unmapped file types
        self._add_fallback_mapping()

    @staticmethod
    def _get_supported_file_types(converter_instance: Node) -> list[FileType]:
        """Return supported file types for a converter, resolving subclasses before their parents."""

        if isinstance(converter_instance, LLMPDFConverter):
            return [FileType.PDF]
        if isinstance(converter_instance, LLMImageConverter):
            return [FileType.IMAGE]

        return [
            file_type
            for file_type, supported_converter_classes in FILE_TYPE_TO_SUPPORTED_CONVERTER_CLASS_MAP.items()
            if isinstance(converter_instance, supported_converter_classes)
        ]

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
        Process the files, converting each one concurrently while preserving input order.

        Files originating from ``file_paths`` are processed first (in their resolved order),
        followed by files supplied directly via ``files``. Conversions run on a
        ``ContextAwareThreadPoolExecutor`` so contextvars (tracing, request ids) propagate to
        worker threads. Output ordering and error semantics match sequential processing: the
        documents are concatenated in input order, and the first failure is re-raised.
        """

        work_items = self._build_work_items(file_paths=file_paths, files=files, metadata=metadata, config=config)

        # Snapshot the inbound dependency chain so every per-file conversion starts from the
        # same base (the FileTypeExtractor / upstream node). Workers never mutate the shared
        # self._run_depends; they each return the dependency chain they produced.
        base_run_depends = list(self._run_depends)

        results = self._convert_work_items_parallel(work_items, base_run_depends, config, **kwargs)

        all_documents: list[Document] = []
        collected_run_depends: list[dict] = []
        for documents, run_depends in results:
            all_documents.extend(documents)
            collected_run_depends.extend(run_depends)

        # Surface the union of every per-file converter as this node's dependency chain so
        # downstream tracing reflects all sub-converters that ran, not just the last one.
        if collected_run_depends:
            self._run_depends = collected_run_depends

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

    def _build_work_items(
        self,
        file_paths: list[str] | None,
        files: list[BytesIO] | None,
        metadata: dict[str, Any] | list[dict[str, Any]] | None,
        config: RunnableConfig = None,
    ) -> list[tuple[BytesIO, str, dict]]:
        """Resolve all inputs into an ordered list of ``(file, filename, metadata)`` work items.

        File reading and path/directory resolution happen here (single-threaded) so that the
        parallel stage only performs the actual, independent per-file conversion work. Reading
        every input can be slow, so ``check_cancellation`` is polled per item to honor a cancel
        signal promptly during this prep phase.
        """
        work_items: list[tuple[BytesIO, str, dict]] = []

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

            # Order-preserving de-duplication: keep the first occurrence of each path so that
            # documents map back to the input order and stay aligned with list-valued metadata.
            # (A plain set() would scramble both the output order and the metadata pairing.)
            all_filepaths = list(dict.fromkeys(filepaths + filepaths_in_directories))

            if not all_filepaths:
                raise FileNotFoundError(f"No files found in the provided paths: {file_paths}")

            meta_list = self._normalize_metadata(metadata, len(all_filepaths))

            for filepath, meta in zip(all_filepaths, meta_list):
                check_cancellation(config)
                with open(filepath, "rb") as f:
                    file_content = BytesIO(f.read())
                work_items.append((file_content, filepath.name, meta))

        if files is not None:
            meta_list = self._normalize_metadata(metadata, len(files))
            for i, (file, meta) in enumerate(zip(files, meta_list)):
                check_cancellation(config)
                filename = meta.get("filename", f"file_{i}")
                file_copy = BytesIO(file.getvalue())
                original_name = getattr(file, "name", None)
                if original_name is not None:
                    file_copy.name = original_name
                work_items.append((file_copy, filename, meta))

        return work_items

    def _convert_work_items_parallel(
        self,
        work_items: list[tuple[BytesIO, str, dict]],
        base_run_depends: list[dict],
        config: RunnableConfig,
        **kwargs,
    ) -> list[tuple[list, list[dict]]]:
        """Convert work items concurrently, preserving input order and first-failure semantics.

        Returns a list aligned with ``work_items`` of ``(documents, run_depends)`` tuples.
        Results are collected in submission order and the first worker exception is re-raised,
        matching the sequential "fail on first error" behavior. The main thread polls for
        cancellation so a cancel signal stops pending files without waiting for the whole batch.
        """
        if not work_items:
            return []

        check_cancellation(config)

        # Single file: no need to spin up a thread pool.
        if len(work_items) == 1:
            file, filename, meta = work_items[0]
            return [self._convert_one(file, filename, meta, base_run_depends, config, **kwargs)]

        max_workers = self._resolve_max_workers(len(work_items), config)

        executor = ContextAwareThreadPoolExecutor(max_workers=max_workers)
        try:
            futures = [
                executor.submit(self._convert_one, file, filename, meta, base_run_depends, config, **kwargs)
                for file, filename, meta in work_items
            ]
            # Poll so cancellation is honored promptly, and stop at the first worker failure so
            # later/unstarted files are not run — matching the sequential loop's fail-fast behavior.
            pending = set(futures)
            failed = False
            while pending and not failed:
                check_cancellation(config)
                done, pending = wait(pending, timeout=_CANCELLATION_POLL_INTERVAL, return_when=FIRST_EXCEPTION)
                failed = any(future.exception() is not None for future in done)
            if failed:
                # Re-raise the earliest-index failure; ``finally`` cancels the rest.
                for future in futures:
                    if future.done() and not future.cancelled() and future.exception() is not None:
                        raise future.exception()
            results = [future.result() for future in futures]
        except CanceledException:
            if config and getattr(config, "cancellation", None) and config.cancellation.token:
                config.cancellation.token.cancel()
            raise
        finally:
            # cancel_futures drops files that have not started; in-flight ones cannot be interrupted.
            executor.shutdown(wait=False, cancel_futures=True)

        return results

    def _resolve_max_workers(self, item_count: int, config: RunnableConfig | None) -> int:
        """Determine the thread-pool size, honoring the node's and RunnableConfig's limits."""
        max_workers = item_count if self.max_workers is None else self.max_workers
        config_limit = config.max_node_workers if config else None
        if config_limit:
            max_workers = min(max_workers, config_limit)
        return max(1, min(max_workers, item_count))

    def _convert_one(
        self,
        file: BytesIO,
        filename: str,
        meta: dict,
        base_run_depends: list[dict],
        config: RunnableConfig,
        **kwargs,
    ) -> tuple[list, list[dict]]:
        """Convert a single work item on a worker thread.

        Each call uses a private copy of ``base_run_depends`` so concurrent workers never race
        on shared state. Returns the produced documents alongside the dependency chain the
        sub-converters generated, which the caller folds back into ``self._run_depends``.
        """
        check_cancellation(config)
        run_depends = list(base_run_depends)
        try:
            documents, run_depends = self._process_single_file(file, filename, meta, run_depends, config, **kwargs)
        except Exception as e:
            logger.error(f"Failed to process file {filename}: {str(e)}")
            raise
        return documents, run_depends

    def call_file_type_extractor(
        self,
        file_type_extractor: FileTypeExtractor,
        input_data: FileTypeExtractorInputSchema,
        run_depends: list[dict],
        config: RunnableConfig,
        **run_kwargs,
    ) -> tuple[dict, list[dict]]:
        """
        Call the file type extractor.

        Runs the per-worker ``file_type_extractor`` clone (never the shared
        ``self.file_type_extractor``) with the supplied ``run_depends`` chain (local to the file
        being processed) so it is safe to call from concurrent worker threads. Returns the
        extractor output along with the updated dependency chain.
        """
        file_type_extractor_result = file_type_extractor.run(
            input_data=input_data, config=config, run_depends=run_depends, **run_kwargs
        )
        run_depends = [NodeDependency(node=file_type_extractor).to_dict(for_tracing=True)]

        if file_type_extractor_result.status != RunnableStatus.SUCCESS:
            logger.error(
                f"Node {self.name} - {self.id}: FileTypeExtractor execution failed: "
                f"{file_type_extractor_result.error.to_dict()}"
            )
            raise ValueError("FileTypeExtractor execution failed")
        return file_type_extractor_result.output, run_depends

    def call_converter(
        self, converter: Node, input_data: dict, run_depends: list[dict], config: RunnableConfig, **run_kwargs
    ) -> tuple[dict, list[dict]]:
        """
        Call the converter.

        Runs the per-worker ``converter`` clone (never the shared mapped instance) with the
        supplied ``run_depends`` chain so it is safe to call concurrently. Returns the converter
        output along with the updated dependency chain.
        """
        converter_result = converter.run(input_data=input_data, config=config, run_depends=run_depends, **run_kwargs)
        run_depends = [NodeDependency(node=converter).to_dict(for_tracing=True)]

        converter_name = converter.name

        if converter_result.status != RunnableStatus.SUCCESS:
            logger.error(
                f"Node {self.name} - {self.id}: {converter_name} execution failed: "
                f"{converter_result.error.to_dict()}"
            )
            raise ValueError(f"{converter_name} execution failed")
        return converter_result.output, run_depends

    def call_fallback_converter(
        self, fallback_converter: Node, input_data: dict, run_depends: list[dict], config: RunnableConfig, **run_kwargs
    ) -> tuple[dict, list[dict]]:
        """
        Call the fallback converter.

        Runs the per-worker ``fallback_converter`` clone (never the shared
        ``self.fallback_converter``) with the supplied ``run_depends`` chain so it is safe to call
        concurrently. Returns the fallback converter output along with the updated dependency chain.
        """
        fallback_converter_result = fallback_converter.run(
            input_data=input_data, config=config, run_depends=run_depends, **run_kwargs
        )
        run_depends = [NodeDependency(node=fallback_converter).to_dict(for_tracing=True)]

        fallback_converter_name = fallback_converter.name

        if fallback_converter_result.status != RunnableStatus.SUCCESS:
            logger.error(
                f"Node {self.name} - {self.id}: {fallback_converter_name} execution failed: "
                f"{fallback_converter_result.error.to_dict()}"
            )
            raise ValueError(f"{fallback_converter_name} execution failed")
        return fallback_converter_result.output, run_depends

    def _process_single_file(
        self, file: BytesIO, filename: str, metadata: dict, run_depends: list[dict], config: RunnableConfig, **kwargs
    ) -> tuple[list, list[dict]]:
        """
        Process a single file by routing to appropriate converter based on file type.

        Args:
            file: The file to convert
            filename: The filename for file type detection
            metadata: Metadata for the file
            run_depends: The dependency chain local to this file (never the shared
                ``self._run_depends``), so this method is safe to run concurrently.
            config: Runtime configuration
            **kwargs: Additional arguments

        Returns:
            tuple: ``(documents, run_depends)`` where documents is the list of documents from
                the conversion and run_depends is the dependency chain the sub-converters produced.
        """
        try:
            file_type_extractor = self.file_type_extractor.clone()
            file_type_extractor_result, run_depends = self.call_file_type_extractor(
                file_type_extractor=file_type_extractor,
                input_data={"file": file, "filename": filename},
                run_depends=run_depends,
                config=config,
                **kwargs,
            )

            detected_type = file_type_extractor_result.get("type")
            logger.info(f"Detected file type: {detected_type} for file: {filename}")

            if detected_type and detected_type in self.converter_mapping:
                converter = self.converter_mapping[detected_type].clone()
                converter_name = converter.name

                try:
                    if not hasattr(file, "name"):
                        file.name = filename

                    converter_input = {"files": [file]}
                    if metadata:
                        converter_input["metadata"] = [metadata]

                    check_cancellation(config)
                    result, run_depends = self.call_converter(
                        converter=converter,
                        input_data=converter_input,
                        run_depends=run_depends,
                        config=config,
                        **kwargs,
                    )
                    logger.info(f"Successfully converted using {converter_name}")
                    return result.get("documents", []), run_depends

                except CanceledException:
                    raise
                except Exception as e:
                    logger.warning(f"Failed to convert with {converter_name}: {str(e)}")
                    if self.fallback_converter:
                        return self._convert_with_fallback_converter(
                            file, filename, metadata, run_depends, config, **kwargs
                        )
                    else:
                        raise

            elif self.fallback_converter:
                return self._convert_with_fallback_converter(file, filename, metadata, run_depends, config, **kwargs)

            else:
                raise ValueError(f"Unsupported file type: {detected_type}")

        except CanceledException:
            raise
        except Exception as e:
            error_msg = f"Failed to convert document {filename}: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _convert_with_fallback_converter(
        self, file: BytesIO, filename: str, metadata: dict, run_depends: list[dict], config: RunnableConfig, **kwargs
    ) -> tuple[list, list[dict]]:
        """
        Try to convert using the fallback converter.

        Args:
            file: The file to convert
            filename: The filename
            metadata: Metadata for the file
            run_depends: The dependency chain local to this file.
            config: Runtime configuration
            **kwargs: Additional arguments

        Returns:
            tuple: ``(documents, run_depends)`` from the conversion.
        """
        try:
            logger.info(f"Attempting conversion with fallback converter for {filename}")

            converter_input = {"files": [file]}
            if metadata:
                converter_input["metadata"] = [metadata]

            fallback_converter = self.fallback_converter.clone()
            check_cancellation(config)
            result, run_depends = self.call_fallback_converter(
                fallback_converter=fallback_converter,
                input_data=converter_input,
                run_depends=run_depends,
                config=config,
                **kwargs,
            )

            logger.info(
                f"Successfully converted using fallback converter: {fallback_converter.__class__.__name__}"
            )
            return result.get("documents", []), run_depends

        except CanceledException:
            raise
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
