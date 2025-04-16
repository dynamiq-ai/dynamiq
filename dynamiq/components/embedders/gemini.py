from dynamiq.components.embedders.base import BaseEmbedder
from dynamiq.connections import Gemini as GeminiConnection


class GeminiEmbedder(BaseEmbedder):
    """
    Initializes the GeminiEmbedder component with given configuration.

    Attributes:
        connection (GeminiConnection): The connection to the Gemini API. A new connection
            is created if none is provided.
        model (str): The model name to use for embedding. Defaults to "gemini/gemini-embedding-exp-03-07"
        input_type (str): Specifies the type of embedding task. Defaults to "search_query"
    """

    connection: GeminiConnection
    model: str = "gemini/gemini-embedding-exp-03-07"
    input_type: str = "search_query"

    def __init__(self, **kwargs):
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = GeminiConnection()
        super().__init__(**kwargs)

    @property
    def embed_params(self) -> dict:
        """
        Returns the embedding parameters for the Gemini API.

        Returns:
            dict: A dictionary containing the parameters for the embedding call.
        """
        params = super().embed_params

        input_to_task_mapping = {
            "search_document": "RETRIEVAL_DOCUMENT",
            "search_query": "RETRIEVAL_QUERY",
            "classification": "CLASSIFICATION",
            "clustering": "CLUSTERING",
        }
        params["task_type"] = input_to_task_mapping.get(self.input_type)

        if self.truncate:
            params["truncate"] = self.truncate

        if self.dimensions:
            params["dimensions"] = self.dimensions

        return params
