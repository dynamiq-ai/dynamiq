from typing import Any, Literal

from pydantic import ConfigDict

from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.flows import Flow
from dynamiq.nodes import InputTransformer, NodeGroup, llms
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.embedders import OpenAITextEmbedder
from dynamiq.nodes.node import Node, NodeDependency
from dynamiq.nodes.retrievers import PineconeDocumentRetriever
from dynamiq.prompts import Message, Prompt
from dynamiq.runnables import RunnableConfig
from dynamiq.storages.vector import PineconeVectorStore

# Constants
EMBEDDING_MODEL = "text-embedding-ada-002"
LLM_MODEL = "gpt-4o-mini"
VECTOR_STORE_INDEX = "default"
VECTOR_DIMENSION = 1536


class BankRAGTool(Node):
    """
    A tool that uses RAG (Retrieval-Augmented Generation) to answer questions
    based on internal bank system documents and policies.

    This tool integrates with Pinecone for document retrieval and OpenAI for
    text embedding and answer generation.
    """

    name: str = "Bank RAG Tool"
    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    description: str = (
        """A tool with access to Internal Bank System Documents and Policies.
        Provide request with key 'input'.
        """
    )
    vector_store: PineconeVectorStore = None
    text_embedder_node: OpenAITextEmbedder = None
    document_retriever_node: PineconeDocumentRetriever = None
    answer_generation_node: llms.OpenAI = None
    retriever_flow: Flow = None

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.vector_store = PineconeVectorStore(
            index_name=VECTOR_STORE_INDEX, dimension=VECTOR_DIMENSION
        )
        self.text_embedder_node = OpenAITextEmbedder(model=EMBEDDING_MODEL)
        self.document_retriever_node = self._create_document_retriever()
        self.answer_generation_node = self._create_answer_generation_node()
        self.retriever_flow = self._create_retriever_flow()

    def _create_document_retriever(self) -> PineconeDocumentRetriever:
        return PineconeDocumentRetriever(
            vector_store=self.vector_store,
            depends=[NodeDependency(self.text_embedder_node)],
            input_transformer=InputTransformer(
                selector={
                    "embedding": f"${[self.text_embedder_node.id]}.output.embedding"
                }
            ),
        )

    def _create_answer_generation_node(self) -> llms.OpenAI:
        prompt = Prompt(
            id="1",
            messages=[Message(content=self._default_prompt_template(), role="user")],
        )
        return llms.OpenAI(
            id="1",
            name="OpenAI Answer Generation",
            model=LLM_MODEL,
            prompt=prompt,
            connection=OpenAIConnection(),
            depends=[NodeDependency(self.document_retriever_node)],
            input_transformer=InputTransformer(
                selector={
                    "documents": f"${[self.document_retriever_node.id]}.output.documents"
                }
            ),
        )

    def _create_retriever_flow(self) -> Flow:
        return Flow(
            id="retriever_flow_pinecone_default",
            nodes=[
                self.text_embedder_node,
                self.document_retriever_node,
                self.answer_generation_node,
            ],
        )

    @staticmethod
    def _default_prompt_template() -> str:
        """
        Returns the default prompt template for the LLM.

        The template includes instructions for answering the question based on
        the provided context, formatting guidelines, and placeholders for the
        query and retrieved documents.
        """
        return r"""
        Please answer the following question based on the information found
        within the sections enclosed by triplet quotes (\`\`\`).
        Your response should be concise, well-written, and follow markdown formatting guidelines:

        - Use bullet points for list items.
        - Use **bold** text for emphasis where necessary.

        **Question:** {{query}}

        Thank you for your detailed attention to the request
        **Context information**:
        ```
        {% for document in documents %}
            ---
            Document title: {{ document.metadata["title"] }}
            Document information: {{ document.content }}
            ---
        {% endfor %}
        ```

        **User Question:** {{query}}
        Answer:
        """

    def _use(self, input_text: str) -> str:
        """
        Core method to process the input query and generate an answer using RAG.

        Args:
            input_text (str): The input query to be answered.

        Returns:
            str: The generated answer based on retrieved documents.
        """

        flow_result = self.retriever_flow.run(input_data={"query": input_text})
        answer = flow_result.output.get(self.answer_generation_node.id).get("output")
        return answer

    def execute(self, input_data: dict[str, Any], _: RunnableConfig = None, **kwargs) -> dict[str, Any]:
        """
        Execute the RAG tool to generate an answer based on the input query.

        Args:
            input_data (dict[str, Any]): A dictionary containing the input query
                under the key 'input'.
            config (RunnableConfig, optional): Configuration for the execution.
            **kwargs: Additional keyword arguments.

        Returns:
            dict[str, Any]: A dictionary containing the generated answer under
                the key 'content'.
        """

        if input_text := input_data.get("input"):
            result = self._use(str(input_text))
        else:
            raise ToolExecutionException("Error: Provide request with key 'input'.", recoverable=True)
        return {"content": result}
