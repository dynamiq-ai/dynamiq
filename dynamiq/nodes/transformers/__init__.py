from typing import TYPE_CHECKING

from dynamiq.utils.lazy_imports import lazy_exports

if TYPE_CHECKING:
    from .csv import CSVToList, JSONToCSV
    from .dynamiq_document import TextToDynamiqDocument
    from .excel import ExcelToList
    from .json import AnyToJSON, JSONToAny
    from .markdown import HTMLMarkdownToText, MarkdownToPDF
    from .sort_list import SortList
    from .text import TextTemplate

_LAZY_EXPORTS = {
    "CSVToList": ".csv",
    "JSONToCSV": ".csv",
    "TextToDynamiqDocument": ".dynamiq_document",
    "ExcelToList": ".excel",
    "AnyToJSON": ".json",
    "JSONToAny": ".json",
    "HTMLMarkdownToText": ".markdown",
    "MarkdownToPDF": ".markdown",
    "SortList": ".sort_list",
    "TextTemplate": ".text",
}

lazy_exports(__name__, _LAZY_EXPORTS, globals())

__all__ = [
    "CSVToList",
    "JSONToCSV",
    "TextToDynamiqDocument",
    "ExcelToList",
    "AnyToJSON",
    "JSONToAny",
    "HTMLMarkdownToText",
    "MarkdownToPDF",
    "SortList",
    "TextTemplate",
]
