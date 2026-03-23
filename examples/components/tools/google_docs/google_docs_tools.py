import os

from dynamiq.connections import GoogleOAuth2
from dynamiq.nodes.tools.google_docs.create_document import CreateDocument
from dynamiq.nodes.tools.google_docs.read_document import ReadDocument
from dynamiq.nodes.tools.google_docs.update_document import UpdateDocument

body = """
# Heading 1

Some **bold** and *italic* text with a [link](https://example.com).


Just a regular paragraph.

## Heading 2

1. First numbered item
2. Second numbered item

- Bullet item one
- Bullet item two

### Numbered List with Bullet Points ###

1. Item One
2. Item Two
    - Sub-item A
    - Sub-item B
3. Item Three

### Bullet Points with Numbered List ###

-   Item One
-   Item Two
    1. Sub-item A
    2. Sub-item B
-   Item Three

Regular paragraph text here.
"""


def create_document():
    """Creates a new Google Docs document with provided body text."""
    tool = CreateDocument(connection=GoogleOAuth2(access_token=os.getenv("GOOGLE_OAUTH2_ACCESS_TOKEN")))
    tool.run(input_data={"title": "Markdown Test Doc", "parse_markdown": False, "body": body})
    tool.close()


def update_document():
    """Updates an existing Google Docs document by parsing Markdown body text."""
    tool = UpdateDocument(connection=GoogleOAuth2(access_token=os.getenv("GOOGLE_OAUTH2_ACCESS_TOKEN")))
    tool.run(input_data={"document_id": "your_document_id", "parse_markdown": True, "body": body})
    tool.close()


def read_document():
    """Reads an existing Google Docs document and prints its content."""
    tool = ReadDocument(connection=GoogleOAuth2(access_token=os.getenv("GOOGLE_OAUTH2_ACCESS_TOKEN")))
    tool.run(
        input_data={
            "document_id": "your_document_id",
        }
    )
    tool.close()


if __name__ == "__main__":
    create_document()
    update_document()
    read_document()
