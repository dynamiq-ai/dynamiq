from pydantic import BaseModel, Field


class ContextEntry(BaseModel):
    key: str
    data: str
    description: str


class Context(BaseModel):
    data: dict[str, ContextEntry] = Field(default_factory=dict)

    def add_entry(self, entry: ContextEntry) -> None:
        """Adds entry to the context.

        Args:
            entry (ContextEntry): Entry to add to the context.
        """
        if entry.key not in self.data:
            self.data[entry.key] = ContextEntry(key=entry.key, data=entry.data, description=entry.description)
        else:
            self.data[entry.key].data += f"\n{entry.data}"

    def get_entry(self, key: str) -> str:
        """Returns entry information by key.

        Args:
            key (str): Key used to extract information.

        Returns:
            str: Data associated with the key.
        """
        if key not in self.data:
            raise ValueError(f"Error. No data found under key {key}")
        return self.data.get(key).data

    @property
    def formatted_data(self) -> str:
        """Returns the data stored in context in a readable format."""
        return "\n".join(
            f"{key}:\n Description: {value.description}\n Data: {value.data}\n" for key, value in self.data.items()
        )


class ContextConfig(BaseModel):
    context: Context = None
    max_context_length: int | None = None
    context_usage_ratio: float = 0.8

    @property
    def enabled(self) -> bool:
        """Returns whether context context is enabled."""
        return bool(self.context)
