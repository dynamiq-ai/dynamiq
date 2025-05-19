from pydantic import BaseModel, Field

class MemoryEntry(BaseModel):
    key: str
    data: str
    description: str

class InnerMemory(BaseModel):
    data: dict[str, MemoryEntry] = Field(default=dict(), description="Storage")

    def add_entry(self, entry: MemoryEntry) -> None:
        if entry.key not in self.data:
            self.data[entry.key] = MemoryEntry(key=entry.key, data=entry.data, description=entry.description)
        else:
            self.data[entry.key].data += f"\n{entry.data}"

    def get_entry(self, key: str) -> str:
        if key not in self.data:
            raise ValueError(f"Error. No data found under key {key}")
        return self.data.get(key).data

class InnerMemoryConfig(BaseModel):
    inner_memory: InnerMemory = None
    max_context_length: int | None = None
    context_usage_ratio: float = 0.8

    @property
    def enabled(self) -> bool:
        return bool(self.inner_memory)
