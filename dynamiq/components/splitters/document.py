import enum
from copy import deepcopy

from more_itertools import windowed
from pydantic import BaseModel, ConfigDict, Field

from dynamiq.types import Document


class DocumentSplitBy(str, enum.Enum):
    """Enum class for document splitting methods."""

    WORD = "word"
    SENTENCE = "sentence"
    PAGE = "page"
    PASSAGE = "passage"
    TITLE = "title"
    CHARACTER = "character"


SPLIT_STR_BY_SPLIT_TYPE = {
    DocumentSplitBy.PAGE: "\f",
    DocumentSplitBy.PASSAGE: "\n\n",
    DocumentSplitBy.SENTENCE: ".",
    DocumentSplitBy.WORD: " ",
    DocumentSplitBy.TITLE: "\n#",
    DocumentSplitBy.CHARACTER: "",
}


class DocumentSplitter(BaseModel):
    """
    Splits a list of text documents into a list of text documents with shorter texts.

    Splitting documents with long texts is a common preprocessing step during indexing.
    This allows Embedders to create significant semantic representations
    and avoids exceeding the maximum context length of language models.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    split_by: DocumentSplitBy = DocumentSplitBy.PASSAGE
    split_length: int = Field(default=10, gt=0)
    split_overlap: int = Field(default=0, ge=0)

    def run(self, documents: list[Document]) -> dict:
        """
        Splits the provided documents into smaller parts based on the specified configuration.

        Args:
            documents (list[Document]): The list of documents to be split.

        Returns:
            dict: A dictionary containing one key, 'documents', which is a list of the split Documents.

        Raises:
            TypeError: If the input is not a list of Document instances.
            ValueError: If the content of any document is None.
        """
        if not isinstance(documents, list) or (
            documents and not isinstance(documents[0], Document)
        ):
            raise TypeError("DocumentSplitter expects a List of Documents as input.")

        split_docs = []
        for doc in documents:
            if doc.content is None:
                raise ValueError(
                    f"DocumentSplitter only works with text documents but document.content for document "
                    f"ID {doc.id} is None."
                )
            units = self._split_into_units(doc.content, self.split_by)
            text_splits = self._concatenate_units(
                units, self.split_length, self.split_overlap
            )
            if doc.metadata is None:
                doc.metadata = {}
            metadata = deepcopy(doc.metadata)
            metadata["source_id"] = doc.id
            split_docs += [
                Document(content=txt, metadata=metadata) for txt in text_splits
            ]
        return {"documents": split_docs}

    def _split_into_units(self, text: str, split_by: DocumentSplitBy) -> list[str]:
        """
        Splits the input text into units based on the specified split_by method.

        Args:
            text (str): The input text to be split.
            split_by (DocumentSplitBy): The method to use for splitting the text.

        Returns:
            list[str]: A list of text units after splitting.
        """
        split_at = SPLIT_STR_BY_SPLIT_TYPE[split_by]
        if split_by == DocumentSplitBy.CHARACTER:
            return [char for char in text]
        else:
            units = text.split(split_at)
        # Add the delimiter back to all units except the last one
        for i in range(len(units) - 1):
            if split_at == "\n#":
                units[i] = "\n# " + units[i]
            else:
                units[i] += split_at
        return units

    def _concatenate_units(
        self, elements: list[str], split_length: int, split_overlap: int
    ) -> list[str]:
        """
        Concatenates the elements into parts of split_length units.

        Args:
            elements (list[str]): The list of text units to be concatenated.
            split_length (int): The maximum number of units in each split.
            split_overlap (int): The number of overlapping units between splits.

        Returns:
            list[str]: A list of concatenated text splits.
        """
        text_splits = []
        segments = windowed(elements, n=split_length, step=split_length - split_overlap)
        for seg in segments:
            current_units = [unit for unit in seg if unit is not None]
            txt = "".join(current_units)
            if len(txt) > 0:
                text_splits.append(txt)
        return text_splits
