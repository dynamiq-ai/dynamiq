from typing import TYPE_CHECKING

from dynamiq.utils.lazy_imports import lazy_exports

if TYPE_CHECKING:
    from .bedrock import BedrockDocumentEmbedder, BedrockTextEmbedder
    from .cohere import CohereDocumentEmbedder, CohereTextEmbedder
    from .gemini import GeminiDocumentEmbedder, GeminiTextEmbedder
    from .huggingface import HuggingFaceDocumentEmbedder, HuggingFaceTextEmbedder
    from .mistral import MistralDocumentEmbedder, MistralTextEmbedder
    from .openai import OpenAIDocumentEmbedder, OpenAITextEmbedder
    from .vertexai import VertexAIDocumentEmbedder, VertexAITextEmbedder
    from .watsonx import WatsonXDocumentEmbedder, WatsonXTextEmbedder

_LAZY_EXPORTS = {
    "BedrockDocumentEmbedder": ".bedrock",
    "BedrockTextEmbedder": ".bedrock",
    "CohereDocumentEmbedder": ".cohere",
    "CohereTextEmbedder": ".cohere",
    "GeminiDocumentEmbedder": ".gemini",
    "GeminiTextEmbedder": ".gemini",
    "HuggingFaceDocumentEmbedder": ".huggingface",
    "HuggingFaceTextEmbedder": ".huggingface",
    "MistralDocumentEmbedder": ".mistral",
    "MistralTextEmbedder": ".mistral",
    "OpenAIDocumentEmbedder": ".openai",
    "OpenAITextEmbedder": ".openai",
    "VertexAIDocumentEmbedder": ".vertexai",
    "VertexAITextEmbedder": ".vertexai",
    "WatsonXDocumentEmbedder": ".watsonx",
    "WatsonXTextEmbedder": ".watsonx",
}

lazy_exports(__name__, _LAZY_EXPORTS, globals())

__all__ = [
    "BedrockDocumentEmbedder",
    "BedrockTextEmbedder",
    "CohereDocumentEmbedder",
    "CohereTextEmbedder",
    "GeminiDocumentEmbedder",
    "GeminiTextEmbedder",
    "HuggingFaceDocumentEmbedder",
    "HuggingFaceTextEmbedder",
    "MistralDocumentEmbedder",
    "MistralTextEmbedder",
    "OpenAIDocumentEmbedder",
    "OpenAITextEmbedder",
    "VertexAIDocumentEmbedder",
    "VertexAITextEmbedder",
    "WatsonXDocumentEmbedder",
    "WatsonXTextEmbedder",
]
