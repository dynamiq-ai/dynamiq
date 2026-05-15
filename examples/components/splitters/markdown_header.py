from dynamiq.nodes.splitters import MarkdownHeaderSplitter
from dynamiq.types import Document


def main() -> None:
    splitter = MarkdownHeaderSplitter()
    splitter.init_components()

    markdown = (
        "# Project Overview\n"
        "Dynamiq orchestrates agentic AI workflows.\n\n"
        "## Splitters\n"
        "Built-in splitters cover characters, tokens, code, JSON, and HTML.\n\n"
        "### Recursive\n"
        "Walks a separator hierarchy and recurses on oversized pieces.\n\n"
        "## Embedders\n"
        "Wrap any provider for vectorization."
    )
    output = splitter.execute(splitter.input_schema(documents=[Document(content=markdown)]))
    for chunk in output["documents"]:
        path = " > ".join(value for key, value in chunk.metadata.items() if key.startswith("h"))
        print(f"[{path}] {chunk.content!r}")


if __name__ == "__main__":
    main()
