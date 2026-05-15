from dynamiq.nodes.splitters import RecursiveCharacterSplitter
from dynamiq.types import Document


def main() -> None:
    splitter = RecursiveCharacterSplitter(chunk_size=120, chunk_overlap=20)
    splitter.init_components()

    text = (
        "Dynamiq is an orchestration framework for agentic AI applications. "
        "It provides Nodes, Workflows, Agents, and Tools.\n\n"
        "Splitters break long documents into chunks that respect token / character "
        "budgets while preserving useful metadata for downstream retrieval."
    )
    documents = [Document(content=text, metadata={"source": "intro"})]
    output = splitter.execute(splitter.input_schema(documents=documents))
    for chunk in output["documents"]:
        print(f"[chunk {chunk.metadata['chunk_index']} @ {chunk.metadata['start_index']}] {chunk.content}")


if __name__ == "__main__":
    main()
