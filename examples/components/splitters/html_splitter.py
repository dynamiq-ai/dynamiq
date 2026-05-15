from importlib.util import find_spec

from dynamiq.nodes.splitters import HTMLHeaderSplitter
from dynamiq.types import Document


def main() -> None:
    if find_spec("bs4") is None:
        print(
            "HTMLHeaderSplitter requires beautifulsoup4. Install with: poetry install (or pip install beautifulsoup4)"
        )
        return

    splitter = HTMLHeaderSplitter()
    splitter.init_components()

    html = (
        "<html><body>"
        "<h1>Welcome</h1><p>Intro paragraph.</p>"
        "<h2>Features</h2><p>Splitters, embedders, retrievers.</p>"
        "<h2>Roadmap</h2><p>More language presets and tree-sitter.</p>"
        "</body></html>"
    )
    output = splitter.execute(splitter.input_schema(documents=[Document(content=html)]))
    for chunk in output["documents"]:
        print(f"{chunk.metadata} -> {chunk.content!r}")


if __name__ == "__main__":
    main()
