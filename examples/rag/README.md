# Basic Python RAG Example

## How to run the RAG Example

There are several options for running the example workflow.

If you would like to try the example using the default parameters without making any changes, simply execute the following command:

```
    python examples/rag/pinecone_flow.py main
```

It is also possible to specify the folder for uploading raw documents using `--folder-path` and to define the question to ask with `--question`:

```
    python examples/rag/pinecone_flow.py main --folder-path=examples/data --question="How to update an order?"
```

To run only the indexing workflow, use:

```
    python examples/rag/pinecone_flow.py indexing-flow --folder-path=examples/data
```

Alternatively, to execute only the inference workflow:

```
    python examples/rag/pinecone_flow.py retrieval-flow --question="How to update an order?"
```

Before running the example, please make sure to start Docker with the required services:

```
    make run_unstructured
```

# DAG RAG example from yaml

Please also make sure to create and fill in the `.env` file with the appropriate credentials, similar to what is shown in `.env.example`.

```
    python examples/rag/dag_yaml.py
```

# RAG example with filters
```
    python examples/rag/filters/filtering_example.py
```
