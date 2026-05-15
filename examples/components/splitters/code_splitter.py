from dynamiq.nodes.splitters import CodeSplitter, Language
from dynamiq.types import Document

PYTHON_SOURCE = """\
def fetch_users():
    return [{"id": 1}, {"id": 2}]


class UserService:
    def __init__(self, store):
        self.store = store

    def get(self, user_id: int):
        return next((u for u in self.store if u["id"] == user_id), None)
"""


def main() -> None:
    splitter = CodeSplitter(language=Language.PYTHON, chunk_size=25, chunk_overlap=10)
    splitter.init_components()
    output = splitter.execute(splitter.input_schema(documents=[Document(content=PYTHON_SOURCE)]))
    for chunk in output["documents"]:
        print("--- chunk ---")
        print(chunk.content)


if __name__ == "__main__":
    main()
