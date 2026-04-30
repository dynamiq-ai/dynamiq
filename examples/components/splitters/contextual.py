import os

from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.llms import OpenAI as OpenAILLM
from dynamiq.nodes.splitters import ContextualChunker, RecursiveCharacterSplitter
from dynamiq.types import Document


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY to run this example.")
        return

    inner = RecursiveCharacterSplitter(chunk_size=140, chunk_overlap=20)
    llm = OpenAILLM(connection=OpenAIConnection(), model="gpt-4o-mini")
    chunker = ContextualChunker(inner_splitter=inner, llm=llm, prepend=True)
    chunker.init_components()

    document = Document(
        content=(
            "Acme Corp Q3 revenue rose 12% year-on-year, driven by enterprise SaaS bookings. "
            "Gross margin expanded by 200 bps. Free cash flow improved due to reduced "
            "capex. Headcount grew 8% across engineering and go-to-market."
        ),
        metadata={"company": "Acme", "quarter": "Q3"},
    )
    output = chunker.execute(chunker.input_schema(documents=[document]))
    for chunk in output["documents"]:
        print("--- contextualized chunk ---")
        print(chunk.content)
        print(f"[context metadata]: {chunk.metadata.get('context')}")


if __name__ == "__main__":
    main()
