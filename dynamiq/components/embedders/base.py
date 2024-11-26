from typing import Any, Callable

from pydantic import BaseModel, PrivateAttr

from dynamiq.connections import BaseConnection
from dynamiq.types import Document


class BaseEmbedder(BaseModel):
    """
    Initializes the Embedder component with given configuration.

    Attributes:
        connection (Optional[BaseConnection]): The connection to the  API. A new connection
            is created if none is provided.
        model (str): The model name to use for embedding.
        prefix (str): A prefix string to prepend to each document text before embedding.
        suffix (str): A suffix string to append to each document text after embedding.
        batch_size (int): The number of documents to encode in a single batch.
        meta_fields_to_embed (Optional[list[str]]): A list of document meta fields to embed alongside
            the document text.
        embedding_separator (str): The separator string used to join document text with meta fields
            for embedding.
        truncate(str): truncate embeddings that are too long from start or end, ("NONE"|"START"|"END").
            Passing "START" will discard the start of the input. "END" will discard the end of the input. In both
            cases, input is discarded until the remaining input is exactly the maximum input token length
            for the model. If "NONE" is selected, when the input exceeds the maximum input token length
            an error will be returned.
        input_type(str):specifies the type of input you're giving to the model. Supported values are
            "search_document", "search_query", "classification" and "clustering".
        dimensions(int):he number of dimensions the resulting output embeddings should have.
            Only supported in OpenAI/Azure text-embedding-3 and later models.

    """
    model: str
    connection: BaseConnection
    prefix: str = ""
    suffix: str = ""
    batch_size: int = 32
    meta_fields_to_embed: list[str] | None = []
    embedding_separator: str = "\n"
    truncate: str | None = None
    input_type: str | None = None
    dimensions: int | None = None
    client: Any | None = None

    _embedding: Callable = PrivateAttr()

    def __init__(self, *args, **kwargs):
        # Import in runtime to save memory
        super().__init__(**kwargs)
        from litellm import embedding

        self._embedding = embedding

    @property
    def embed_params(self) -> dict:
        params = self.connection.conn_params
        if self.client:
            params = {"client": self.client}
        return params

    def embed_text(self, text: str) -> dict:
        """
        Embeds a single string using the Embedder model specified during the initialization of the component.

        Args:
            text (str): The text string to be embedded.

        Returns:
            dict: A dictionary containing:
                - 'embedding': A list representing the embedding vector of the input text.
                - 'meta': A dictionary with metadata information about the model usage.
        """
        if not isinstance(text, str):
            msg = (
                "TextEmbedder expects a string as input."
                "In case you want to embed a list of Documents, please use the DocumentEmbedder."
            )
            raise TypeError(msg)

        text_to_embed = self.prefix + text + self.suffix
        text_to_embed = text_to_embed.replace("\n", " ")

        response = self._embedding(
            model=self.model, input=[text_to_embed], **self.embed_params
        )

        meta = {"model": response.model, "usage": dict(response.usage)}

        return {"embedding": response.data[0]["embedding"], "meta": meta}

    def _prepare_documents_to_embed(self, documents: list[Document]) -> list[str]:
        """
        Prepare the texts to embed by concatenating the Document text with the metadata fields to embed.

        Args:
            documents (list[Document]): A list of Document objects to prepare for embedding.

        Returns:
            list[str]: A list of concatenated strings ready for embedding.
        """
        texts_to_embed: list[str] = []
        for doc in documents:
            meta_values_to_embed = [
                str(doc.meta[key])
                for key in self.meta_fields_to_embed
                if doc.meta.get(key) is not None
            ]

            text_to_embed = self.embedding_separator.join(
                meta_values_to_embed + [doc.content or ""]
            )
            texts_to_embed.append(text_to_embed)
        return texts_to_embed

    def _embed_texts_batch(
        self, texts_to_embed: list[str], batch_size: int
    ) -> tuple[list[list[float]], dict[str, Any]]:
        """
        Embed a list of texts in batches.
        """
        all_embeddings = []
        meta: dict[str, Any] = {}
        embed_params = self.embed_params
        for i in range(0, len(texts_to_embed), batch_size):
            batch = texts_to_embed[i : i + batch_size]
            response = self._embedding(model=self.model, input=batch, **embed_params)
            embeddings = [el["embedding"] for el in response.data]
            all_embeddings.extend(embeddings)

            if "model" not in meta:
                meta["model"] = response.model
            if "usage" not in meta:
                meta["usage"] = dict(response.usage)
            else:
                meta["usage"]["prompt_tokens"] += response.usage.prompt_tokens
                meta["usage"]["total_tokens"] += response.usage.total_tokens

        return all_embeddings, meta

    def embed_documents(self, documents: list[Document]) -> dict:
        """
        Embeds a list of documents and returns the embedded documents along with meta information.

        Args:
            documents (list[Document]): The documents to be embedded.

        Returns:
            dict: A dictionary containing:
                - 'documents' (list[Document]): The input documents with their embeddings populated.
                - 'meta' (dict): Metadata information about the embedding process.
        """
        if (
            not isinstance(documents, list)
            or documents
            and not isinstance(documents[0], Document)
        ):
            msg = (
                "DocumentEmbedder expects a list of Documents as input."
                "In case you want to embed a string, please use the embed_text."
            )
            raise TypeError(msg)

        if not documents:
            # return early if we were passed an empty list
            return {"documents": [], "meta": {}}

        texts_to_embed = self._prepare_documents_to_embed(documents=documents)

        embeddings, meta = self._embed_texts_batch(
            texts_to_embed=texts_to_embed, batch_size=self.batch_size
        )

        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb

        return {"documents": documents, "meta": meta}
