from pydantic import BaseModel, Field


class Config(BaseModel):
    search_limit: int = Field(default=5, description="The number of relevant memories to retrieve during search.")
    search_filters: dict = Field(default={}, description="Filters to apply during vector search.")
