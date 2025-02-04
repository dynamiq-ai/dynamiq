# Basic Python RAG Example

## How to run the RAG Example

There are several options for running the example workflow.

### Pinecone Example

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

### Elasticsearch Example

Elasticsearch provides multiple search capabilities including dense vector, BM25 text, and hybrid search. Here are the different examples:

Basic RAG workflow with vector search:
```
    python examples/rag/elasticsearch_flow.py main --folder-path=examples/data --question="How to update an order?"
```

Hybrid search combining vector similarity with text matching:
```
    python examples/rag/elasticsearch_hybrid_flow.py main --folder-path=examples/data --question="How to update an order?"
```

Advanced features demonstration (BM25, combined search, document store operations):
```
    python examples/rag/elasticsearch_search_types.py
```

Before running the examples, make sure to:
1. Start Elasticsearch (you can use Docker):
```
    docker run -d --name elasticsearch -p 9200:9200 -e "discovery.type=single-node" -e "xpack.security.enabled=false" elasticsearch:8.12.0
```

2. Set up environment variables in `.env`:
```
ELASTICSEARCH_URL=http://localhost:9200
ELASTICSEARCH_USERNAME=elastic  # if security enabled
ELASTICSEARCH_PASSWORD=your_password  # if security enabled
```

Before running any example, please make sure to start Docker with the required services:

```
    make run_unstructured
```

# DAG RAG example from yaml

Please also make sure to create and fill in the `.env` file with the appropriate credentials, similar to what is shown in `.env.example`.

```
    python examples/rag/dag_yaml.py
```

For Elasticsearch DAG example:
```
    python examples/rag/dag_elasticsearch.py
```

# RAG example with filters

Using Pinecone:
```
    python examples/rag/filters/filtering_example.py
```

Using Elasticsearch with metadata filters:
```
    python examples/rag/elasticsearch_flow.py main --folder-path=examples/data --question="How to update an order?" --filters='{"category": "documentation"}'
