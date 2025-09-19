from pydantic import BaseModel, Field


class DryRunConfig(BaseModel):
    """Configuration for dry run.

    Attributes:
        enabled (bool): Whether dry run is enabled. Defaults to False.
        delete_documents: Whether to delete the ingested documents after the dry run. Defaults to True.
        delete_collection: Whether to delete the created collection after the dry run. Defaults to False.
    """

    enabled: bool = False
    delete_documents: bool = Field(default=True, description="Delete the ingested documents")
    delete_collection: bool = Field(default=True, description="Delete the created collection")
