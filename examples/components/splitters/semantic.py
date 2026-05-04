import os

from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.embedders import OpenAITextEmbedder
from dynamiq.nodes.splitters import BreakpointThresholdType, SemanticSplitter
from dynamiq.types import Document


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY to run this example.")
        return

    embedder = OpenAITextEmbedder(connection=OpenAIConnection(), model="text-embedding-3-small")
    splitter = SemanticSplitter(
        embedder=embedder,
        breakpoint_threshold_type=BreakpointThresholdType.PERCENTILE,
        breakpoint_threshold_amount=65.0,
        buffer_size=1,
    )
    splitter.init_components()

    text = (
        "The cat sat on the mat. It was a sunny day. The kitten purred contentedly. "
        "Inflation in the eurozone fell again. Markets reacted positively to the news. "
        "Central banks signalled patience. Tomorrow we will write tests for the splitters."
    )
    output = splitter.execute(splitter.input_schema(documents=[Document(content=text)]))
    for chunk in output["documents"]:
        print(f"[chunk {chunk.metadata['chunk_index']}] {chunk.content}")


if __name__ == "__main__":
    main()
