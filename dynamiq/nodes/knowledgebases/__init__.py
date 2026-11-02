from typing import TYPE_CHECKING

from dynamiq.utils.lazy_imports import lazy_exports

if TYPE_CHECKING:
    from .bedrock import BedrockKnowledgeBaseSearch
    from .knowledgebase_graph import DynamiqKnowledgebaseGraphSearch
    from .knowledgebase_hybrid import DynamiqKnowledgebaseHybridSearch
    from .knowledgebase_vector import DynamiqKnowledgebaseVectorSearch
    from .vertex import VertexAIRagSearch

_LAZY_EXPORTS = {
    "BedrockKnowledgeBaseSearch": ".bedrock",
    "DynamiqKnowledgebaseVectorSearch": ".knowledgebase_vector",
    "DynamiqKnowledgebaseGraphSearch": ".knowledgebase_graph",
    "DynamiqKnowledgebaseHybridSearch": ".knowledgebase_hybrid",
    "VertexAIRagSearch": ".vertex",
}

lazy_exports(__name__, _LAZY_EXPORTS, globals())

__all__ = [
    "BedrockKnowledgeBaseSearch",
    "DynamiqKnowledgebaseVectorSearch",
    "DynamiqKnowledgebaseGraphSearch",
    "DynamiqKnowledgebaseHybridSearch",
    "VertexAIRagSearch",
]
