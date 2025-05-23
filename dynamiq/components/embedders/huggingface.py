from typing import Any, ClassVar

from dynamiq.components.embedders.base import BaseEmbedder, InvalidEmbeddingError
from dynamiq.connections import HuggingFace as HuggingFaceConnection
from dynamiq.utils.logger import logger


class HuggingFaceEmbedder(BaseEmbedder):
    """
    Initializes the HuggingFaceEmbedder component with given configuration.

    Attributes:
        connection (HuggingFaceConnection): The connection to the  HuggingFace API. A new connection
            is created if none is provided.
        model (str): The model name to use for embedding. Defaults to "huggingface/BAAI/bge-large-zh"
    """

    API_BASE_URL: ClassVar[str] = "https://api-inference.huggingface.co/models"
    connection: HuggingFaceConnection
    model: str = "huggingface/BAAI/bge-large-zh"

    def __init__(self, **kwargs):
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = HuggingFaceConnection()
        super().__init__(**kwargs)

    @property
    def embed_params(self) -> dict:
        params = super().embed_params

        if self.model.startswith("huggingface/"):
            model_id = self.model[len("huggingface/") :]
        else:
            model_id = self.model

        params["api_base"] = f"{self.API_BASE_URL}/{model_id}"

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

        Raises:
            TypeError: If input is not a string
            ValueError: If the embedding response is invalid
        """
        if not isinstance(text, str):
            msg = (
                "TextEmbedder expects a string as input."
                "In case you want to embed a list of Documents, please use the DocumentEmbedder."
            )
            raise TypeError(msg)

        text_to_embed = f"{self.prefix}{text}{self.suffix}"
        text_to_embed = text_to_embed.replace("\n", " ")

        response = self._embedding(model=self.model, input=text_to_embed, **self.embed_params)

        meta = {"model": response.model, "usage": dict(response.usage)}
        embedding = response.data[0]["embedding"]

        try:
            self.validate_embedding(embedding)
        except InvalidEmbeddingError as e:
            logger.error(f"Invalid embedding returned by model {self.model}: {str(e)}")
            raise ValueError(f"Invalid embedding returned by the model: {str(e)}")

        return {"embedding": embedding, "meta": meta}

    def _embed_texts_batch(
        self, texts_to_embed: list[str], batch_size: int
    ) -> tuple[list[list[float]], dict[str, Any]]:
        """
        Embed a list of texts one by one (non-batched API).
        """
        all_embeddings = []
        meta: dict[str, Any] = {}
        embed_params = self.embed_params

        for i in range(0, len(texts_to_embed), batch_size):
            batch = texts_to_embed[i : i + batch_size]

            for text in batch:
                response = self._embedding(model=self.model, input=text, **embed_params)

                embedding = response.data[0]["embedding"]
                all_embeddings.append(embedding)

                if "model" not in meta:
                    meta["model"] = response.model

                if "usage" not in meta:
                    meta["usage"] = dict(response.usage)
                else:
                    meta["usage"]["prompt_tokens"] += response.usage.prompt_tokens
                    meta["usage"]["total_tokens"] += response.usage.total_tokens

        return all_embeddings, meta
