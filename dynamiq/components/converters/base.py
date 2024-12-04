import copy
import os
from abc import abstractmethod
from io import BytesIO
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from dynamiq.types import Document, DocumentCreationMode


class BaseConverter(BaseModel):
    document_creation_mode: DocumentCreationMode = DocumentCreationMode.ONE_DOC_PER_FILE

    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

    def run(
        self,
        file_paths: list[str] | list[os.PathLike] | None = None,
        files: list[BytesIO] | None = None,
        metadata: dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> dict[str, list[Any]]:
        """
        Converts files to Documents using the PyPDF.

        Processes file paths or BytesIO objects into Documents. Handles directories and files.

        Args:
            paths: List of file or directory paths to convert.
            files: List of BytesIO objects to convert.
            metadata: Metadata for documents. Can be a dict for all or a list of dicts for each.

        Returns:
            Dict with 'documents' key containing a list of created Documents.

        Raises:
            ValueError: If neither paths nor files provided, or if metadata is a list with
                directory paths.
        """

        documents = []

        if file_paths is not None:
            paths_obj = [Path(path) for path in file_paths]
            filepaths = [path for path in paths_obj if path.is_file()]
            filepaths_in_directories = [
                filepath for path in paths_obj if path.is_dir() for filepath in path.glob("*.*") if filepath.is_file()
            ]
            if filepaths_in_directories and isinstance(metadata, list):
                raise ValueError(
                    "If providing directories in the `paths` parameter, "
                    "`metadata` can only be a dictionary (metadata applied to every file), "
                    "and not a list. To specify different metadata for each file, "
                    "provide an explicit list of direct paths instead."
                )

            all_filepaths = set(filepaths + filepaths_in_directories)
            meta_list = self._normalize_metadata(metadata, len(all_filepaths))

            for filepath, meta in zip(all_filepaths, meta_list):
                documents.extend(self._process_file(filepath, meta))

        if files is not None:
            meta_list = self._normalize_metadata(metadata, len(files))
            for file, meta in zip(files, meta_list):
                documents.extend(self._process_file(file, meta))

        return {"documents": documents}

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

    @abstractmethod
    def _create_documents(
        self,
        filepath: str,
        elements: Any,
        document_creation_mode: DocumentCreationMode,
        metadata: dict[str, Any],
        **kwargs,
    ) -> list[Document]:
        pass

    @abstractmethod
    def _process_file(self, file: Path | BytesIO, metadata: dict[str, Any]) -> list[Any]:
        pass
