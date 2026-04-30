from dynamiq.nodes.splitters import TokenSplitter
from dynamiq.types import Document


def main() -> None:
    splitter = TokenSplitter(chunk_size=32, chunk_overlap=4, encoding_name="cl100k_base")
    splitter.init_components()

    text = (
        "Token-based splitting prevents accidental overflow of LLM context windows. "
        "It uses the same tokenizer the model uses, so chunk_size is honest."
    )
    output = splitter.execute(splitter.input_schema(documents=[Document(content=text)]))
    for chunk in output["documents"]:
        print(f"[{chunk.metadata['chunk_index']}] {chunk.content}")


if __name__ == "__main__":
    main()
