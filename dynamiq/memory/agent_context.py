from pydantic import BaseModel, Field


class ContextEntry(BaseModel):
    key: str
    data: str
    description: str


class Context(BaseModel):
    data: dict[str, ContextEntry] = Field(default_factory=dict, description="Storage")

    def add_entry(self, entry: ContextEntry) -> None:
        if entry.key not in self.data:
            self.data[entry.key] = ContextEntry(key=entry.key, data=entry.data, description=entry.description)
        else:
            self.data[entry.key].data += f"\n{entry.data}"

    def get_entry(self, key: str) -> str:
        if key not in self.data:
            raise ValueError(f"Error. No data found under key {key}")
        return self.data.get(key).data


class ContextConfig(BaseModel):
    context: Context = None
    max_context_length: int | None = None
    context_usage_ratio: float = 0.8

    @property
    def enabled(self) -> bool:
        return bool(self.context)
