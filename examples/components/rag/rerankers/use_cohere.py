from dynamiq.connections import Cohere
from dynamiq.nodes.rankers import CohereReranker
from dynamiq.types import Document

if __name__ == "__main__":
    ranker = CohereReranker(connection=Cohere())

    input_data = {
        "query": "What is machine learning?",
        "documents": [
            Document(content="Machine learning is a branch of AI...", score=0.8),
            Document(content="Deep learning is a subset of machine learning...", score=0.7),
        ],
    }

    output = ranker.run(input_data=input_data)

    print(output)
