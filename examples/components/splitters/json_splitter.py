import json

from dynamiq.nodes.splitters import RecursiveJsonSplitter
from dynamiq.types import Document


def main() -> None:
    splitter = RecursiveJsonSplitter(max_chunk_size=80, convert_lists=True)
    splitter.init_components()

    payload = {
        "users": [
            {"id": "u-1", "name": "Alice", "role": "admin"},
            {"id": "u-2", "name": "Bob", "role": "editor"},
            {"id": "u-3", "name": "Carla", "role": "viewer"},
        ],
        "metadata": {"version": "0.48.0", "env": "staging"},
    }
    output = splitter.execute(splitter.input_schema(documents=[Document(content=json.dumps(payload))]))
    for chunk in output["documents"]:
        print(f"[chunk {chunk.metadata['chunk_index']}] {chunk.content}")


if __name__ == "__main__":
    main()
