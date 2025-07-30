import io
from typing import Any

from pydantic import BaseModel, Field


class FileOutput(BaseModel):
    """
    Represents a file output from a tool execution.
    
    This class standardizes how tools return file data, providing consistent
    metadata and content handling across the framework.
    
    Attributes:
        name (str): The filename including extension.
        content (bytes): The binary content of the file.
        mime_type (str | None): MIME type of the file (e.g., 'image/png').
        description (str | None): Human-readable description of the file.
        path (str | None): Original path where the file was located (e.g., in sandbox).
        size (int | None): Size of the file in bytes.
    """
    
    name: str = Field(..., description="Filename including extension")
    content: bytes = Field(..., description="Binary content of the file")
    mime_type: str | None = Field(None, description="MIME type of the file")
    description: str | None = Field(None, description="Human-readable description")
    path: str | None = Field(None, description="Original file path")
    size: int | None = Field(None, description="File size in bytes")
    
    def __init__(self, **data: Any):
        super().__init__(**data)
        # Auto-calculate size if not provided
        if self.size is None and self.content:
            self.size = len(self.content)
    
    @classmethod
    def from_bytes_io(
        cls,
        file_obj: io.BytesIO,
        name: str,
        mime_type: str | None = None,
        description: str | None = None,
        path: str | None = None,
    ) -> "FileOutput":
        """
        Create FileOutput from a BytesIO object.
        
        Args:
            file_obj: BytesIO object containing file data
            name: Filename including extension
            mime_type: MIME type of the file
            description: Human-readable description
            path: Original file path
            
        Returns:
            FileOutput: New FileOutput instance
        """
        content = file_obj.getvalue()
        return cls(
            name=name,
            content=content,
            mime_type=mime_type,
            description=description,
            path=path,
        )
    
    @classmethod
    def from_file_path(
        cls,
        file_path: str,
        name: str | None = None,
        mime_type: str | None = None,
        description: str | None = None,
    ) -> "FileOutput":
        """
        Create FileOutput from a file path.
        
        Args:
            file_path: Path to the file to read
            name: Filename (defaults to basename of file_path)
            mime_type: MIME type of the file
            description: Human-readable description
            
        Returns:
            FileOutput: New FileOutput instance
        """
        import os
        
        if name is None:
            name = os.path.basename(file_path)
            
        with open(file_path, "rb") as f:
            content = f.read()
            
        return cls(
            name=name,
            content=content,
            mime_type=mime_type,
            description=description,
            path=file_path,
        )
    
    def to_bytes_io(self) -> io.BytesIO:
        """
        Convert FileOutput to BytesIO object.
        
        Returns:
            io.BytesIO: BytesIO object containing file content
        """
        file_obj = io.BytesIO(self.content)
        file_obj.name = self.name
        return file_obj
    
    def save_to_path(self, file_path: str) -> None:
        """
        Save FileOutput content to a file path.
        
        Args:
            file_path: Path where to save the file
        """
        with open(file_path, "wb") as f:
            f.write(self.content)
    
    def model_dump_for_serialization(self, **kwargs) -> dict[str, Any]:
        """
        Model dump excluding binary content for serialization.
        
        Returns:
            dict: Model data without binary content
        """
        data = self.model_dump(**kwargs)
        data["content"] = f"<binary data: {self.size} bytes>"
        return data