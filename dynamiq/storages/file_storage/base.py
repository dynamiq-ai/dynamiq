"""Base file storage interface and common data structures."""

import abc
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO, Dict, Iterator, List, Optional, Union
from dataclasses import dataclass, field


@dataclass
class FileInfo:
    """Information about a stored file."""
    name: str
    path: str
    size: int
    content_type: str = "text/plain"
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    

class StorageError(Exception):
    """Base exception for storage operations."""
    def __init__(self, message: str, operation: str = None, path: str = None):
        self.message = message
        self.operation = operation
        self.path = path
        super().__init__(self.message)


class FileNotFoundError(StorageError):
    """Raised when a file is not found in storage."""
    pass


class FileExistsError(StorageError):
    """Raised when trying to create a file that already exists."""
    pass


class PermissionError(StorageError):
    """Raised when permission is denied for a storage operation."""
    pass


class FileStorage(abc.ABC):
    """Abstract base class for file storage implementations.
    
    This interface provides a unified way to interact with different
    file storage backends (in-memory, file system, cloud storage, etc.).
    """
    
    @abc.abstractmethod
    def store(
        self, 
        file_path: Union[str, Path], 
        content: Union[str, bytes, BinaryIO],
        content_type: str = None,
        metadata: Dict[str, Any] = None,
        overwrite: bool = False
    ) -> FileInfo:
        """Store a file in the storage backend.
        
        Args:
            file_path: Path where the file should be stored
            content: File content as string, bytes, or file-like object
            content_type: MIME type of the file content
            metadata: Additional metadata to store with the file
            overwrite: Whether to overwrite existing files
            
        Returns:
            FileInfo object with details about the stored file
            
        Raises:
            FileExistsError: If file exists and overwrite=False
            PermissionError: If storage operation is not permitted
            StorageError: For other storage-related errors
        """
        pass
    
    @abc.abstractmethod
    def retrieve(self, file_path: Union[str, Path]) -> bytes:
        """Retrieve file content from storage.
        
        Args:
            file_path: Path of the file to retrieve
            
        Returns:
            File content as bytes
            
        Raises:
            FileNotFoundError: If file doesn't exist
            PermissionError: If retrieval is not permitted
            StorageError: For other storage-related errors
        """
        pass

    
    @abc.abstractmethod
    def exists(self, file_path: Union[str, Path]) -> bool:
        """Check if a file exists in storage.
        
        Args:
            file_path: Path of the file to check
            
        Returns:
            True if file exists, False otherwise
        """
        pass


    @abc.abstractmethod
    def delete(self, file_path: Union[str, Path]) -> bool:
        """Delete a file from storage.
        
        Args:
            file_path: Path of the file to delete
            
        Returns:
            True if file was deleted, False if it didn't exist
            
        Raises:
            PermissionError: If deletion is not permitted
            StorageError: For other storage-related errors
        """
        pass
    
    @abc.abstractmethod
    def list_files(
        self, 
        directory: Union[str, Path] = "", 
        recursive: bool = False,
        pattern: str = None
    ) -> List[FileInfo]:
        """List files in storage.
        
        Args:
            directory: Directory to list (empty string for root)
            recursive: Whether to list files recursively
            pattern: Glob pattern to filter files
            
        Returns:
            List of FileInfo objects
        """
        pass

