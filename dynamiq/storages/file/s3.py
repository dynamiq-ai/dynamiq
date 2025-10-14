"""S3 file storage implementation."""

import mimetypes
import os
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from pydantic import Field, ConfigDict

from dynamiq.connections import AWS
from dynamiq.utils.logger import logger

from .base import FileInfo, FileNotFoundError, FileStore, StorageError


class S3FileStore(FileStore):
    """S3 file storage implementation.

    This implementation provides:
    - Direct S3 storage without caching
    - Basic compression support
    - Storage class configuration
    - Batch operations for efficiency

    Args:
        bucket: S3 bucket name
        connection: AWS connection instance
        prefix: S3 key prefix for all files
        enable_compression: Whether to compress files before upload
        storage_class: S3 storage class (STANDARD, STANDARD_IA, GLACIER, etc.)
        use_transfer_acceleration: Enable S3 Transfer Acceleration
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # S3 Configuration
    bucket: str = Field(..., description="S3 bucket name")
    connection: AWS = Field(default_factory=AWS, description="AWS connection")
    prefix: str = Field(default="", description="S3 key prefix for all files")
    
    # S3 Optimization
    enable_compression: bool = Field(default=True, description="Enable file compression")
    storage_class: str = Field(default="STANDARD", description="S3 storage class")
    use_transfer_acceleration: bool = Field(default=False, description="Enable S3 Transfer Acceleration")
    
    # Private attributes
    _s3_client: Any = None
    _s3_resource: Any = None

    def __init__(self, **kwargs):
        """Initialize S3 file store."""
        super().__init__(**kwargs)
        self._init_s3_client()

    def _init_s3_client(self):
        """Initialize S3 client and resource."""
        try:
            session = self.connection.get_boto3_session()
            
            # Configure S3 client
            client_config = {}
            if self.use_transfer_acceleration:
                client_config['use_accelerate_endpoint'] = True
            
            self._s3_client = session.client('s3', **client_config)
            self._s3_resource = session.resource('s3')
            
            # Verify bucket access
            self._s3_client.head_bucket(Bucket=self.bucket)
                
            logger.info(f"S3FileStore initialized with bucket: {self.bucket}")
            
        except NoCredentialsError:
            raise StorageError("AWS credentials not found", operation="init")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchBucket':
                raise StorageError(f"S3 bucket '{self.bucket}' does not exist", operation="init")
            elif error_code == 'AccessDenied':
                raise StorageError(f"Access denied to S3 bucket '{self.bucket}'", operation="init")
            else:
                raise StorageError(f"S3 error: {e}", operation="init")

    def _get_s3_key(self, file_path: str | Path) -> str:
        """Convert file path to S3 key."""
        file_path = str(file_path).lstrip('/')
        if self.prefix:
            return f"{self.prefix.rstrip('/')}/{file_path}"
        return file_path

    def _compress_content(self, content: bytes) -> bytes:
        """Compress content if compression is enabled."""
        if not self.enable_compression:
            return content
        
        # Only compress if it's worth it (content > 1KB)
        if len(content) < 1024:
            return content
            
        try:
            import gzip
            compressed = gzip.compress(content)
            # Only use compressed version if it's actually smaller
            if len(compressed) < len(content):
                return compressed
        except Exception as e:
            logger.warning(f"Compression failed: {e}")
        
        return content

    def _decompress_content(self, content: bytes) -> bytes:
        """Decompress content if it was compressed."""
        try:
            import gzip
            return gzip.decompress(content)
        except (gzip.BadGzipFile, OSError):
            # Not compressed or not gzip, return as-is
            return content

    def list_files_bytes(self) -> list[BytesIO]:
        """List files in storage and return content as BytesIO objects."""
        try:
            response = self._s3_client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=self.prefix
            )
            
            files = []
            for obj in response.get('Contents', []):
                key = obj['Key']
                file_path = key[len(self.prefix):].lstrip('/') if self.prefix else key
                
                # Fetch from S3
                try:
                    response = self._s3_client.get_object(Bucket=self.bucket, Key=key)
                    content = response['Body'].read()
                    content = self._decompress_content(content)
                    
                    file_io = BytesIO(content)
                    file_io.name = file_path
                    file_io.content_type = obj.get('ContentType', 'application/octet-stream')
                    files.append(file_io)
                except ClientError as e:
                    logger.warning(f"Failed to fetch {key}: {e}")
                    continue
            
            return files
            
        except ClientError as e:
            raise StorageError(f"Failed to list files: {e}", operation="list")

    def store(
        self,
        file_path: str | Path,
        content: str | bytes | BinaryIO,
        content_type: str = None,
        metadata: dict[str, Any] = None,
        overwrite: bool = False,
    ) -> FileInfo:
        """Store a file in S3."""
        file_path = str(file_path)
        s3_key = self._get_s3_key(file_path)
        
        # Check if file exists and overwrite is False
        if not overwrite and self.exists(file_path):
            logger.info(f"File '{file_path}' already exists. Skipping...")
            return self._get_file_info(file_path)
        
        # Convert content to bytes
        if isinstance(content, str):
            content_bytes = content.encode("utf-8")
        elif isinstance(content, bytes):
            content_bytes = content
        elif hasattr(content, "read"):  # BinaryIO-like object
            content_bytes = content.read()
            if hasattr(content, "seek"):
                content.seek(0)
        else:
            raise StorageError(f"Unsupported content type: {type(content)}", operation="store", path=file_path)
        
        # Compress content if enabled
        compressed_content = self._compress_content(content_bytes)
        
        # Determine content type
        if content_type is None:
            content_type, _ = mimetypes.guess_type(file_path)
            if content_type is None:
                content_type = "application/octet-stream"
        
        # Prepare S3 metadata
        s3_metadata = {
            'original_size': str(len(content_bytes)),
            'compressed': str(self.enable_compression and len(compressed_content) < len(content_bytes)),
            'created_at': datetime.now().isoformat(),
        }
        if metadata:
            s3_metadata.update(metadata)
        
        # Upload to S3
        try:
            extra_args = {
                'StorageClass': self.storage_class,
                'Metadata': s3_metadata,
                'ContentType': content_type,
            }
            
            if self.enable_compression and len(compressed_content) < len(content_bytes):
                extra_args['ContentEncoding'] = 'gzip'
            
            self._s3_client.put_object(
                Bucket=self.bucket,
                Key=s3_key,
                Body=compressed_content,
                **extra_args
            )
            
            logger.info(f"File '{file_path}' stored in S3 bucket '{self.bucket}'")
            
            return FileInfo(
                name=os.path.basename(file_path),
                path=file_path,
                size=len(content_bytes),
                content_type=content_type,
                created_at=datetime.now(),
                metadata=metadata or {},
            )
            
        except ClientError as e:
            raise StorageError(f"Failed to store file '{file_path}': {e}", operation="store", path=file_path)

    def retrieve(self, file_path: str | Path) -> bytes:
        """Retrieve file content from S3."""
        file_path = str(file_path)
        s3_key = self._get_s3_key(file_path)
        
        try:
            response = self._s3_client.get_object(Bucket=self.bucket, Key=s3_key)
            content = response['Body'].read()
            
            # Decompress if needed
            content = self._decompress_content(content)
            
            return content
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                raise FileNotFoundError(f"File '{file_path}' not found", operation="retrieve", path=file_path)
            else:
                raise StorageError(f"Failed to retrieve file '{file_path}': {e}", operation="retrieve", path=file_path)

    def exists(self, file_path: str | Path) -> bool:
        """Check if file exists in S3."""
        file_path = str(file_path)
        s3_key = self._get_s3_key(file_path)
        
        try:
            self._s3_client.head_object(Bucket=self.bucket, Key=s3_key)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            else:
                raise StorageError(f"Failed to check file existence '{file_path}': {e}", operation="exists", path=file_path)

    def delete(self, file_path: str | Path) -> bool:
        """Delete file from S3."""
        file_path = str(file_path)
        s3_key = self._get_s3_key(file_path)
        
        try:
            self._s3_client.delete_object(Bucket=self.bucket, Key=s3_key)
            logger.info(f"File '{file_path}' deleted from S3")
            return True
            
        except ClientError as e:
            logger.error(f"Failed to delete file '{file_path}': {e}")
            return False

    def list_files(
        self,
        directory: str | Path = "",
        recursive: bool = False,
        pattern: str = None,
    ) -> list[FileInfo]:
        """List files in S3 with optional filtering."""
        directory = str(directory).strip('/')
        prefix = self._get_s3_key(directory) if directory else self.prefix
        
        try:
            response = self._s3_client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=prefix
            )
            
            files = []
            for obj in response.get('Contents', []):
                key = obj['Key']
                file_path = key[len(self.prefix):].lstrip('/') if self.prefix else key
                
                # Apply directory filtering
                if directory and not file_path.startswith(directory):
                    continue
                
                # Apply recursive filtering
                if not recursive and '/' in file_path[len(directory):].lstrip('/'):
                    continue
                
                # Apply pattern filtering
                if pattern and not self._matches_pattern(file_path, pattern):
                    continue
                
                files.append(self._get_file_info(file_path, obj))
            
            return files
            
        except ClientError as e:
            raise StorageError(f"Failed to list files: {e}", operation="list")

    def _matches_pattern(self, file_path: str, pattern: str) -> bool:
        """Check if file path matches pattern (simple glob-like matching)."""
        import fnmatch
        return fnmatch.fnmatch(file_path, pattern)

    def _get_file_info(self, file_path: str, s3_object: dict = None) -> FileInfo:
        """Get FileInfo for a file, optionally using S3 object metadata."""
        if s3_object:
            return FileInfo(
                name=os.path.basename(file_path),
                path=file_path,
                size=s3_object.get('Size', 0),
                content_type=s3_object.get('ContentType', 'application/octet-stream'),
                created_at=s3_object.get('LastModified', datetime.now()),
                metadata=s3_object.get('Metadata', {}),
            )
        else:
            # Fallback to basic info
            return FileInfo(
                name=os.path.basename(file_path),
                path=file_path,
                size=0,
                content_type='application/octet-stream',
                created_at=datetime.now(),
                metadata={},
            )
