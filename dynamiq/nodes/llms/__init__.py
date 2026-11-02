from typing import TYPE_CHECKING

from dynamiq.utils.lazy_imports import lazy_exports

if TYPE_CHECKING:
    from .ai21 import AI21
    from .anthropic import Anthropic, AnthropicCacheControl
    from .anyscale import Anyscale
    from .atlascloud import AtlasCloud
    from .azureai import AzureAI
    from .base import BaseLLM
    from .bedrock import Bedrock, BedrockCacheControl
    from .cerebras import Cerebras
    from .cohere import Cohere
    from .custom_llm import CustomLLM
    from .databricks import Databricks
    from .deepinfra import DeepInfra
    from .deepseek import DeepSeek
    from .fireworksai import FireworksAI
    from .gemini import Gemini, GeminiCacheControl
    from .groq import Groq
    from .huggingface import HuggingFace
    from .mistral import Mistral
    from .novita import Novita
    from .nvidia_nim import NvidiaNIM
    from .ollama import Ollama
    from .openai import OpenAI
    from .openrouter import OpenRouter
    from .perplexity import Perplexity
    from .replicate import Replicate
    from .sambanova import SambaNova
    from .togetherai import TogetherAI
    from .vertexai import VertexAI
    from .watsonx import WatsonX
    from .xai import xAI

_LAZY_EXPORTS = {
    "AI21": ".ai21",
    "Anthropic": ".anthropic",
    "AnthropicCacheControl": ".anthropic",
    "Anyscale": ".anyscale",
    "AtlasCloud": ".atlascloud",
    "AzureAI": ".azureai",
    "BaseLLM": ".base",
    "Bedrock": ".bedrock",
    "BedrockCacheControl": ".bedrock",
    "Cerebras": ".cerebras",
    "Cohere": ".cohere",
    "CustomLLM": ".custom_llm",
    "Databricks": ".databricks",
    "DeepInfra": ".deepinfra",
    "DeepSeek": ".deepseek",
    "FireworksAI": ".fireworksai",
    "Gemini": ".gemini",
    "GeminiCacheControl": ".gemini",
    "Groq": ".groq",
    "HuggingFace": ".huggingface",
    "Mistral": ".mistral",
    "Novita": ".novita",
    "NvidiaNIM": ".nvidia_nim",
    "Ollama": ".ollama",
    "OpenAI": ".openai",
    "OpenRouter": ".openrouter",
    "Perplexity": ".perplexity",
    "Replicate": ".replicate",
    "SambaNova": ".sambanova",
    "TogetherAI": ".togetherai",
    "VertexAI": ".vertexai",
    "WatsonX": ".watsonx",
    "xAI": ".xai",
}

lazy_exports(__name__, _LAZY_EXPORTS, globals())

__all__ = [
    "AI21",
    "Anthropic",
    "AnthropicCacheControl",
    "Anyscale",
    "AtlasCloud",
    "AzureAI",
    "BaseLLM",
    "Bedrock",
    "BedrockCacheControl",
    "Cerebras",
    "Cohere",
    "CustomLLM",
    "Databricks",
    "DeepInfra",
    "DeepSeek",
    "FireworksAI",
    "Gemini",
    "GeminiCacheControl",
    "Groq",
    "HuggingFace",
    "Mistral",
    "Novita",
    "NvidiaNIM",
    "Ollama",
    "OpenAI",
    "OpenRouter",
    "Perplexity",
    "Replicate",
    "SambaNova",
    "TogetherAI",
    "VertexAI",
    "WatsonX",
    "xAI",
]
