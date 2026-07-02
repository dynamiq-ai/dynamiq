import copy
import csv
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, Literal

from openpyxl import load_workbook

from dynamiq.components.converters.base import BaseConverter
from dynamiq.components.converters.utils import get_filename_for_bytesio
from dynamiq.types import Document, DocumentCreationMode


class ExcelFileConverter(BaseConverter):
    """
    A component for converting spreadsheet files (xlsx, csv, tsv) to Documents.

    Excel workbooks are read with openpyxl and every sheet is rendered as a markdown
    table (one section per sheet). CSV/TSV files are parsed with the stdlib csv module
    and rendered as a single markdown table.

    Args:
        document_creation_mode (Literal["one-doc-per-file"], optional):
            Determines how to create Documents from the spreadsheet content. Currently only
            supports:
            - `"one-doc-per-file"`: Creates one Document per file.
            Defaults to `"one-doc-per-file"`.

    Usage example:
        ```python
        from dynamiq.components.converters.excel import ExcelFileConverter

        converter = ExcelFileConverter()
        documents = converter.run(file_paths=["a/file/path.xlsx"])["documents"]
        ```
    """

    document_creation_mode: Literal[DocumentCreationMode.ONE_DOC_PER_FILE] = DocumentCreationMode.ONE_DOC_PER_FILE

    def _process_file(self, file: Path | BytesIO, metadata: dict[str, Any]) -> list[Any]:
        """
        Process a file and return a list of Documents.

        Args:
            file: Path to a file or BytesIO object
            metadata: Metadata to attach to the documents

        Returns:
            List of Documents
        """
        if isinstance(file, BytesIO):
            filepath = get_filename_for_bytesio(file)
            file.seek(0)
            data = file.read()
            file.seek(0)
        else:
            filepath = str(file)
            with open(file, "rb") as f:
                data = f.read()

        extension = Path(filepath).suffix.lower()
        if extension in {".csv", ".tsv"}:
            content = self._convert_delimited(data, delimiter="\t" if extension == ".tsv" else ",")
        else:
            content = self._convert_workbook(data)

        return self._create_documents(
            filepath=filepath,
            content=content,
            document_creation_mode=self.document_creation_mode,
            metadata=metadata,
        )

    def _convert_workbook(self, data: bytes) -> str:
        """Convert an Excel workbook to markdown, one table section per sheet."""
        workbook = load_workbook(BytesIO(data), read_only=True, data_only=True)
        try:
            sections = []
            for worksheet in workbook.worksheets:
                rows = [
                    ["" if cell is None else str(cell) for cell in row] for row in worksheet.iter_rows(values_only=True)
                ]
                table = self._to_markdown_table(rows)
                if table:
                    sections.append(f"## {worksheet.title}\n\n{table}" if len(workbook.worksheets) > 1 else table)
            return "\n\n".join(sections)
        finally:
            workbook.close()

    def _convert_delimited(self, data: bytes, delimiter: str) -> str:
        """Convert CSV/TSV content to a markdown table."""
        text = data.decode("utf-8-sig", errors="replace")
        rows = list(csv.reader(StringIO(text), delimiter=delimiter))
        return self._to_markdown_table(rows)

    @staticmethod
    def _to_markdown_table(rows: list[list[str]]) -> str:
        """Render rows as a markdown table, treating the first row as the header."""
        rows = [row for row in rows if any(cell.strip() for cell in row)]
        if not rows:
            return ""

        width = max(len(row) for row in rows)

        def render_row(row: list[str]) -> str:
            padded = row + [""] * (width - len(row))
            cells = [cell.replace("|", "\\|").replace("\n", " ") for cell in padded]
            return "| " + " | ".join(cells) + " |"

        lines = [render_row(rows[0]), "| " + " | ".join(["---"] * width) + " |"]
        lines.extend(render_row(row) for row in rows[1:])
        return "\n".join(lines)

    def _create_documents(
        self,
        filepath: str,
        content: str,
        document_creation_mode: DocumentCreationMode,
        metadata: dict[str, Any],
        **kwargs,
    ) -> list[Document]:
        """
        Create Documents from the spreadsheet content.
        """
        if document_creation_mode != DocumentCreationMode.ONE_DOC_PER_FILE:
            raise ValueError("ExcelFileConverter only supports one-doc-per-file mode")

        metadata = copy.deepcopy(metadata)
        metadata["file_path"] = filepath

        return [Document(content=content.strip(), metadata=metadata)]
