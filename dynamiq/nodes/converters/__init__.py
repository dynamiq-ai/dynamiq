from typing import TYPE_CHECKING

from dynamiq.utils.lazy_imports import lazy_exports

if TYPE_CHECKING:
    from .csv import CSVConverter, CSVDocumentCreationMode
    from .docx import DOCXFileConverter
    from .excel import ExcelFileConverter
    from .google_document_ai import GoogleDocumentAIFileConverter
    from .html import HTMLConverter
    from .llm_text_extractor import LLMImageConverter, LLMPDFConverter
    from .multi_file_type_converter import MultiFileTypeConverter
    from .pptx import PPTXFileConverter
    from .pypdf import PyPDFConverter
    from .text import TextFileConverter
    from .textract import TextractFileConverter
    from .unstructured import UnstructuredFileConverter

_LAZY_EXPORTS = {
    "CSVConverter": ".csv",
    "CSVDocumentCreationMode": ".csv",
    "DOCXFileConverter": ".docx",
    "ExcelFileConverter": ".excel",
    "GoogleDocumentAIFileConverter": ".google_document_ai",
    "HTMLConverter": ".html",
    "LLMImageConverter": ".llm_text_extractor",
    "LLMPDFConverter": ".llm_text_extractor",
    "MultiFileTypeConverter": ".multi_file_type_converter",
    "PPTXFileConverter": ".pptx",
    "PyPDFConverter": ".pypdf",
    "TextFileConverter": ".text",
    "TextractFileConverter": ".textract",
    "UnstructuredFileConverter": ".unstructured",
}

lazy_exports(__name__, _LAZY_EXPORTS, globals())

__all__ = [
    "CSVConverter",
    "CSVDocumentCreationMode",
    "DOCXFileConverter",
    "ExcelFileConverter",
    "GoogleDocumentAIFileConverter",
    "HTMLConverter",
    "LLMImageConverter",
    "LLMPDFConverter",
    "MultiFileTypeConverter",
    "PPTXFileConverter",
    "PyPDFConverter",
    "TextFileConverter",
    "TextractFileConverter",
    "UnstructuredFileConverter",
]
