# Basic Python RAG Example

## How to run the RAG Example

There are several options for running the example workflow.

If you would like to try the example using the default parameters without making any changes, simply execute the following command:

```
    python examples/components/rag/vector_stores/pinecone_flow.py main
```

It is also possible to specify the folder for uploading raw documents using `--folder-path` and to define the question to ask with `--question`:

```
    python examples/components/rag/vector_stores/pinecone_flow.py main --folder-path=examples/data --question="How to update an order?"
```

To run only the indexing workflow, use:

```
    python examples/components/rag/vector_stores/pinecone_flow.py indexing-flow --folder-path=examples/data
```

Alternatively, to execute only the inference workflow:

```
    python examples/components/rag/vector_stores/pinecone_flow.py retrieval-flow --question="How to update an order?"
```

Before running the example, please make sure to start Docker with the required services:

```
    make run_unstructured
```


# RAG example with filters
```
    python examples/components/rag/vector_stores/filters/filtering_example.py
```
