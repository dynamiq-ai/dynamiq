import typer
from fastapi import Body, FastAPI, HTTPException, Path
from pydantic import BaseModel
from vector_storages import (
    delete_chroma_documents_by_file_id,
    delete_pinecone_documents_by_file_id,
    delete_weaviate_documents_by_file_id,
)

app = FastAPI()


class PineconeDocumentsDeleteRequest(BaseModel):
    api_key: str
    index_name: str


class WeaviateDocumentsDeleteRequest(BaseModel):
    api_key: str
    url: str
    index_name: str


class ChromaDocumentsDeleteRequest(BaseModel):
    host: str
    port: str
    index_name: str


@app.delete("/pinecone/delete/{file_id}")
async def delete_pinecone_documents(
    file_id: str = Path(..., description="The ID of the file to delete"),
    request: PineconeDocumentsDeleteRequest = Body(...),
):
    """
    Endpoint to delete documents from Pinecone index by file_id.

    Args:
        file_id (str): The file ID to filter by.
        request (DeleteRequest): The request body containing api_key and index_name.

    Returns:
        dict: A success message.
    """
    try:
        delete_pinecone_documents_by_file_id(
            api_key=request.api_key, index_name=request.index_name, file_id=file_id
        )
        return {"message": "Documents deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/weaviate/delete/{file_id}")
async def delete_weaviate_documents(
    file_id: str = Path(..., description="The ID of the file to delete"),
    request: WeaviateDocumentsDeleteRequest = Body(...),
):
    """
    Endpoint to delete documents from Weaviate by file_id.

    Args:
        file_id (str): The file ID to filter by.
        request (DeleteRequest): The request body containing api_key, url, and index_name.

    Returns:
        dict: A success message.
    """
    try:
        delete_weaviate_documents_by_file_id(
            api_key=request.api_key,
            url=request.url,
            index_name=request.index_name,
            file_id=file_id,
        )
        return {"message": "Documents deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/chroma/delete/{file_id}")
async def delete_chroma_documents(
    file_id: str = Path(..., description="The ID of the file to delete"),
    request: ChromaDocumentsDeleteRequest = Body(...),
):
    """
    Endpoint to delete documents from Chroma by file_id.

    Args:
        file_id (str): The file ID to filter by.
        request (DeleteRequest): The request body containing host, port, and index_name.

    Returns:
        dict: A success message.
    """
    try:
        delete_chroma_documents_by_file_id(
            host=request.host,
            port=request.port,
            index_name=request.index_name,
            file_id=file_id,
        )
        return {"message": "Documents deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def main(host: str = "0.0.0.0", port: int = 8000):  # nosec
    """
    Main function to run the FastAPI server with specified host and port.

    Args:
        host (str): The host to run the server on.
        port (int): The port to run the server on.
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    typer.run(main)
