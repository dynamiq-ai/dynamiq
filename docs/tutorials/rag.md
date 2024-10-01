# RAG Tutorial

## RAG - Document Indexing Flow

This workflow takes input PDF files, pre-processes them, converts them to vector embeddings, and stores them in the Pinecone vector database.

### Step-by-Step Guide

**Import Necessary Libraries**

```python
from io import BytesIO
from dynamiq import Workflow
from dynamiq.nodes import InputTransformer
from dynamiq.connections import OpenAI as OpenAIConnection, Pinecone as PineconeConnection
from dynamiq.nodes.converters import PyPDFConverter
from dynamiq.nodes.splitters.document import DocumentSplitter
from dynamiq.nodes.embedders import OpenAIDocumentEmbedder
from dynamiq.nodes.writers import PineconeDocumentWriter
```

**Initialize the RAG Workflow**

```python
rag_wf = Workflow()
```

**PyPDF Document Converter**

Convert the PDF documents into a format suitable for processing.

```python
converter = PyPDFConverter(document_creation_mode="one-doc-per-page")
rag_wf.flow.add_nodes(converter)  # Add node to the DAG
```

**Document Splitter**

Split the documents into smaller chunks for better processing.

```python
document_splitter = DocumentSplitter(
    split_by="sentence",
    split_length=10,
    split_overlap=1,
    input_transformer=InputTransformer(
        selector={
            "documents": f"${[converter.id]}.output.documents",
        },  # Map output of the previous node to the expected input of the current node
    ),
).depends_on(converter)
rag_wf.flow.add_nodes(document_splitter)
```

**OpenAI Vector Embeddings**

Convert the document chunks into vector embeddings using OpenAI.

```python
embedder = OpenAIDocumentEmbedder(
    connection=OpenAIConnection(api_key="$OPENAI_API_KEY"),
    model="text-embedding-3-small",
    input_transformer=InputTransformer(
        selector={
            "documents": f"${[document_splitter.id]}.output.documents",
        },
    ),
).depends_on(document_splitter)
rag_wf.flow.add_nodes(embedder)
```

**Pinecone Vector Storage**

Store the vector embeddings in the Pinecone vector database.

```python
vector_store = PineconeDocumentWriter(
    connection=PineconeConnection(api_key="$PINECONE_API_KEY"),
    index_name="default",
    dimension=1536,
    input_transformer=InputTransformer(
        selector={
            "documents": f"${[embedder.id]}.output.documents",
        },
    ),
).depends_on(embedder)
rag_wf.flow.add_nodes(vector_store)
```

**Prepare Input PDF Files**

Prepare the input PDF files for processing.

```python
file_paths = ["example.pdf"]
input_data = {
    "files": [
        BytesIO(open(path, "rb").read()) for path in file_paths
    ],
    "metadata": [
        {"filename": path} for path in file_paths
    ],
}
```

**Run RAG Indexing Flow**

Execute the workflow to process and store the documents.

```python
rag_wf.run(input_data=input_data)
```

---

## RAG - Document Retrieval Flow

This simple retrieval RAG flow searches for relevant documents and answers the original user question using the retrieved documents.

### Step-by-Step Guide

**Import Necessary Libraries**

```python
from dynamiq import Workflow
from dynamiq.nodes import InputTransformer
from dynamiq.connections import OpenAI as OpenAIConnection, Pinecone as PineconeConnection
from dynamiq.nodes.embedders import OpenAITextEmbedder
from dynamiq.nodes.retrievers import PineconeDocumentRetriever
from dynamiq.nodes.llms import OpenAI
from dynamiq.prompts import Message, Prompt
```

**Initialize the RAG Retrieval Workflow**

```python
retrieval_wf = Workflow()
```

**Shared OpenAI Connection**

Set up a shared connection to OpenAI.

```python
openai_connection = OpenAIConnection(api_key="$OPENAI_API_KEY")
```

**OpenAI Text Embedder for Query Embedding**

Embed the user query into a vector format.

```python
embedder = OpenAITextEmbedder(
    connection=openai_connection,
    model="text-embedding-3-small",
)
retrieval_wf.flow.add_nodes(embedder)
```

**Pinecone Document Retriever**

Retrieve relevant documents from the Pinecone vector database.

```python
document_retriever = PineconeDocumentRetriever(
    connection=PineconeConnection(api_key="$PINECONE_API_KEY"),
    index_name="default",
    dimension=1536,
    top_k=5,
    input_transformer=InputTransformer(
        selector={
            "embedding": f"${[embedder.id]}.output.embedding",
        },
    ),
).depends_on(embedder)
retrieval_wf.flow.add_nodes(document_retriever)
```

**Define the Prompt Template**

Create a template for generating answers based on the retrieved documents.

```python
prompt_template = """
Please answer the question based on the provided context.

Question: {{ query }}

Context:
{% for document in documents %}
- {{ document.content }}
{% endfor %}
"""
```

**OpenAI LLM for Answer Generation**

Generate an answer to the user query using OpenAI's language model.

```python
prompt = Prompt(messages=[Message(content=prompt_template, role="user")])

answer_generator = OpenAI(
    connection=openai_connection,
    model="gpt-4o",
    prompt=prompt,
    input_transformer=InputTransformer(
        selector={
            "documents": f"${[document_retriever.id]}.output.documents",
            "query": f"${[embedder.id]}.output.query",
        },  # Take documents from the vector store node and query from the embedder
    ),
).depends_on(document_retriever)
retrieval_wf.flow.add_nodes(answer_generator)
```

**Run the RAG Retrieval Flow**

Execute the workflow to retrieve and answer the user query.

```python
question = "What are the line items provided in the invoice?"
result = retrieval_wf.run(input_data={"query": question})

# Print the answer
answer = result.output.get(answer_generator.id).get("output", {}).get("content")
print(answer)
```
