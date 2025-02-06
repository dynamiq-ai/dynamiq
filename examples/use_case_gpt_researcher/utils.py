from dynamiq.connections import Pinecone
from dynamiq.storages.vector.pinecone import PineconeVectorStore


def clean_pinecone_storage():
    """Deletes all documents in the Pinecone vector storage."""
    vector_store = PineconeVectorStore(connection=Pinecone(), index_name="gpt-researcher", create_if_not_exist=True)

    vector_store.delete_documents(delete_all=True)


def save_markdown_as_pdf(md_string: str, output_pdf: str):
    """Save a Markdown string as a PDF."""
    import markdown
    from weasyprint import HTML

    html_content = markdown.markdown(md_string)
    HTML(string=html_content).write_pdf(output_pdf)
