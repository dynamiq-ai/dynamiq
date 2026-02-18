# RFC-001-06: Storage Backends

**Status:** Final Draft v7.0  
**Created:** January 6, 2026  
**Part:** 6 of 9

---

## 1. Overview

This document defines the storage backend interface and implementations for checkpoint persistence.

**Backend Progression:**
1. **File** - Development, debugging, simple deployments
2. **SQLite** - Testing, single-process production
3. **Redis** - Production (fast, distributed, TTL support)
4. **PostgreSQL** - Production (durable, queryable, existing in runtime)

---

## 2. Backend Interface

```python
# dynamiq/checkpoint/backends/base.py

from abc import ABC, abstractmethod
from datetime import datetime
from typing import AsyncIterator, Iterator

from dynamiq.checkpoint.models import FlowCheckpoint, CheckpointStatus

class CheckpointBackend(ABC):
    """
    Abstract base class for checkpoint storage backends.
    
    Implementations must:
    - Be thread-safe for concurrent access
    - Handle their own connection management
    - Support both sync and async operations (async optional)
    
    Example:
        >>> backend = FileCheckpointBackend(".checkpoints")
        >>> checkpoint = FlowCheckpoint(flow_id="f1", run_id="r1")
        >>> checkpoint_id = backend.save(checkpoint)
        >>> loaded = backend.load(checkpoint_id)
    """
    
    @abstractmethod
    def save(self, checkpoint: FlowCheckpoint) -> str:
        """
        Save a checkpoint.
        
        Args:
            checkpoint: The checkpoint to save
            
        Returns:
            The checkpoint ID
            
        Raises:
            StorageError: If save fails
        """
        pass
    
    @abstractmethod
    def load(self, checkpoint_id: str) -> FlowCheckpoint | None:
        """
        Load a checkpoint by ID.
        
        Args:
            checkpoint_id: The checkpoint ID to load
            
        Returns:
            The checkpoint if found, None otherwise
        """
        pass
    
    @abstractmethod
    def update(self, checkpoint: FlowCheckpoint) -> None:
        """
        Update an existing checkpoint.
        
        Args:
            checkpoint: The checkpoint with updated state
            
        Raises:
            StorageError: If update fails
            NotFoundError: If checkpoint doesn't exist
        """
        pass
    
    @abstractmethod
    def delete(self, checkpoint_id: str) -> bool:
        """
        Delete a checkpoint.
        
        Args:
            checkpoint_id: The checkpoint ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        pass
    
    @abstractmethod
    def list_by_flow(
        self, 
        flow_id: str, 
        *, 
        status: CheckpointStatus | None = None,
        limit: int = 10,
        before: datetime | None = None,
    ) -> list[FlowCheckpoint]:
        """
        List checkpoints for a flow.
        
        Args:
            flow_id: The flow ID to query
            status: Optional status filter
            limit: Maximum results
            before: Only checkpoints before this time
            
        Returns:
            List of checkpoints, newest first
        """
        pass
    
    @abstractmethod
    def get_latest(
        self, 
        flow_id: str, 
        *, 
        status: CheckpointStatus | None = None
    ) -> FlowCheckpoint | None:
        """
        Get the most recent checkpoint for a flow.
        
        Args:
            flow_id: The flow ID to query
            status: Optional status filter
            
        Returns:
            The latest checkpoint if found
        """
        pass
    
    def cleanup(
        self, 
        flow_id: str, 
        *, 
        keep_count: int = 10,
        older_than_days: int | None = None
    ) -> int:
        """
        Remove old checkpoints.
        
        Default implementation iterates and deletes.
        Backends may override for efficiency.
        
        Args:
            flow_id: The flow ID to clean up
            keep_count: Number of recent checkpoints to keep
            older_than_days: Delete checkpoints older than this
            
        Returns:
            Number of checkpoints deleted
        """
        checkpoints = self.list_by_flow(flow_id, limit=1000)
        deleted = 0
        
        for i, cp in enumerate(checkpoints):
            should_delete = i >= keep_count
            
            if older_than_days and not should_delete:
                age = (datetime.utcnow() - cp.created_at).days
                should_delete = age > older_than_days
            
            if should_delete:
                if self.delete(cp.id):
                    deleted += 1
        
        return deleted
    
    # Async variants (optional, for async backends)
    async def asave(self, checkpoint: FlowCheckpoint) -> str:
        """Async save - default delegates to sync."""
        return self.save(checkpoint)
    
    async def aload(self, checkpoint_id: str) -> FlowCheckpoint | None:
        """Async load - default delegates to sync."""
        return self.load(checkpoint_id)
    
    async def aupdate(self, checkpoint: FlowCheckpoint) -> None:
        """Async update - default delegates to sync."""
        return self.update(checkpoint)
    
    async def adelete(self, checkpoint_id: str) -> bool:
        """Async delete - default delegates to sync."""
        return self.delete(checkpoint_id)
```

---

## 3. File Backend

```python
# dynamiq/checkpoint/backends/file.py

import json
import fcntl
from pathlib import Path
from datetime import datetime
from typing import Any

from dynamiq.checkpoint.backends.base import CheckpointBackend
from dynamiq.checkpoint.models import FlowCheckpoint, CheckpointStatus
from dynamiq.checkpoint.serializers import checkpoint_json_encoder

class FileCheckpointBackend(CheckpointBackend):
    """
    File-based storage for development and simple deployments.
    
    Features:
    - Each checkpoint is a JSON file
    - Flow index for fast listing
    - File locking for thread safety
    - Human-readable for debugging
    
    Directory structure:
        .checkpoints/
        ├── checkpoints/
        │   ├── {checkpoint_id_1}.json
        │   └── {checkpoint_id_2}.json
        └── indexes/
            └── flow_{flow_id}.json  # List of checkpoint IDs
    
    Example:
        >>> backend = FileCheckpointBackend(".checkpoints")
        >>> checkpoint = FlowCheckpoint(flow_id="my-flow", run_id="run-1")
        >>> checkpoint_id = backend.save(checkpoint)
    """
    
    def __init__(self, directory: str = ".dynamiq_checkpoints"):
        """
        Initialize file backend.
        
        Args:
            directory: Base directory for checkpoint files
        """
        self.base_dir = Path(directory)
        self.checkpoints_dir = self.base_dir / "checkpoints"
        self.indexes_dir = self.base_dir / "indexes"
        
        # Create directories
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.indexes_dir.mkdir(parents=True, exist_ok=True)
    
    def _checkpoint_path(self, checkpoint_id: str) -> Path:
        """Get path for a checkpoint file."""
        return self.checkpoints_dir / f"{checkpoint_id}.json"
    
    def _index_path(self, flow_id: str) -> Path:
        """Get path for a flow's index file."""
        # Sanitize flow_id for filesystem
        safe_id = flow_id.replace("/", "_").replace("\\", "_")
        return self.indexes_dir / f"flow_{safe_id}.json"
    
    def save(self, checkpoint: FlowCheckpoint) -> str:
        """Save checkpoint to file."""
        checkpoint.updated_at = datetime.utcnow()
        path = self._checkpoint_path(checkpoint.id)
        
        # Write checkpoint file with lock
        with open(path, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(
                    checkpoint.model_dump(), 
                    f, 
                    indent=2, 
                    default=checkpoint_json_encoder,
                    ensure_ascii=False,
                )
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        
        # Update flow index
        self._update_index(checkpoint.flow_id, checkpoint.id, checkpoint.created_at)
        
        return checkpoint.id
    
    def load(self, checkpoint_id: str) -> FlowCheckpoint | None:
        """Load checkpoint from file."""
        path = self._checkpoint_path(checkpoint_id)
        if not path.exists():
            return None
        
        with open(path, "r") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
                data = json.load(f)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        
        return FlowCheckpoint(**data)
    
    def update(self, checkpoint: FlowCheckpoint) -> None:
        """Update existing checkpoint."""
        if not self._checkpoint_path(checkpoint.id).exists():
            raise FileNotFoundError(f"Checkpoint {checkpoint.id} not found")
        self.save(checkpoint)
    
    def delete(self, checkpoint_id: str) -> bool:
        """Delete checkpoint file."""
        path = self._checkpoint_path(checkpoint_id)
        if path.exists():
            # Load to get flow_id for index cleanup
            checkpoint = self.load(checkpoint_id)
            path.unlink()
            
            # Update index
            if checkpoint:
                self._remove_from_index(checkpoint.flow_id, checkpoint_id)
            
            return True
        return False
    
    def list_by_flow(
        self, 
        flow_id: str, 
        *, 
        status: CheckpointStatus | None = None,
        limit: int = 10,
        before: datetime | None = None,
    ) -> list[FlowCheckpoint]:
        """List checkpoints for a flow."""
        index_path = self._index_path(flow_id)
        if not index_path.exists():
            return []
        
        with open(index_path, "r") as f:
            index_data = json.load(f)
        
        # Index stores: [{"id": "...", "created_at": "..."}, ...]
        # Sorted by created_at descending (newest first)
        
        checkpoints = []
        for entry in index_data:
            cp_id = entry["id"]
            cp_created = datetime.fromisoformat(entry["created_at"])
            
            # Filter by before
            if before and cp_created >= before:
                continue
            
            # Load checkpoint
            cp = self.load(cp_id)
            if cp is None:
                continue
            
            # Filter by status
            if status and cp.status != status:
                continue
            
            checkpoints.append(cp)
            
            if len(checkpoints) >= limit:
                break
        
        return checkpoints
    
    def get_latest(
        self, 
        flow_id: str, 
        *, 
        status: CheckpointStatus | None = None
    ) -> FlowCheckpoint | None:
        """Get most recent checkpoint."""
        results = self.list_by_flow(flow_id, status=status, limit=1)
        return results[0] if results else None
    
    def _update_index(
        self, 
        flow_id: str, 
        checkpoint_id: str,
        created_at: datetime,
    ) -> None:
        """Add checkpoint to flow index."""
        index_path = self._index_path(flow_id)
        
        # Load existing index
        if index_path.exists():
            with open(index_path, "r") as f:
                index_data = json.load(f)
        else:
            index_data = []
        
        # Add new entry at beginning (newest first)
        new_entry = {
            "id": checkpoint_id,
            "created_at": created_at.isoformat(),
        }
        
        # Remove if already exists (update case)
        index_data = [e for e in index_data if e["id"] != checkpoint_id]
        
        # Insert at beginning
        index_data.insert(0, new_entry)
        
        # Write back
        with open(index_path, "w") as f:
            json.dump(index_data, f, indent=2)
    
    def _remove_from_index(self, flow_id: str, checkpoint_id: str) -> None:
        """Remove checkpoint from flow index."""
        index_path = self._index_path(flow_id)
        if not index_path.exists():
            return
        
        with open(index_path, "r") as f:
            index_data = json.load(f)
        
        index_data = [e for e in index_data if e["id"] != checkpoint_id]
        
        with open(index_path, "w") as f:
            json.dump(index_data, f, indent=2)
```

---

## 4. SQLite Backend

```python
# dynamiq/checkpoint/backends/sqlite.py

import sqlite3
import json
from datetime import datetime
from contextlib import contextmanager
from threading import Lock

from dynamiq.checkpoint.backends.base import CheckpointBackend
from dynamiq.checkpoint.models import FlowCheckpoint, CheckpointStatus
from dynamiq.checkpoint.serializers import checkpoint_json_encoder

class SQLiteCheckpointBackend(CheckpointBackend):
    """
    SQLite-based storage for testing and light production.
    
    Features:
    - Single-file database
    - Automatic schema creation
    - Thread-safe with connection pooling
    - Efficient querying with indexes
    
    Example:
        >>> backend = SQLiteCheckpointBackend("checkpoints.db")
        >>> # Or in-memory for testing:
        >>> backend = SQLiteCheckpointBackend(":memory:")
    """
    
    def __init__(self, db_path: str = ".dynamiq_checkpoints.db"):
        """
        Initialize SQLite backend.
        
        Args:
            db_path: Path to SQLite database file, or ":memory:" for in-memory
        """
        self.db_path = db_path
        self._lock = Lock()
        self._init_schema()
    
    @contextmanager
    def _connection(self):
        """Get a database connection with automatic commit/rollback."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def _init_schema(self) -> None:
        """Create database schema if not exists."""
        with self._connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    id TEXT PRIMARY KEY,
                    flow_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            # Indexes for common queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_checkpoints_flow_created 
                ON checkpoints(flow_id, created_at DESC)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_checkpoints_flow_status 
                ON checkpoints(flow_id, status, created_at DESC)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_checkpoints_run 
                ON checkpoints(run_id)
            """)
    
    def save(self, checkpoint: FlowCheckpoint) -> str:
        """Save checkpoint to database."""
        checkpoint.updated_at = datetime.utcnow()
        
        data_json = json.dumps(
            checkpoint.model_dump(),
            default=checkpoint_json_encoder,
            ensure_ascii=False,
        )
        
        with self._lock:
            with self._connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO checkpoints 
                    (id, flow_id, run_id, status, data_json, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    checkpoint.id,
                    checkpoint.flow_id,
                    checkpoint.run_id,
                    checkpoint.status.value,
                    data_json,
                    checkpoint.created_at.isoformat(),
                    checkpoint.updated_at.isoformat(),
                ))
        
        return checkpoint.id
    
    def load(self, checkpoint_id: str) -> FlowCheckpoint | None:
        """Load checkpoint from database."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT data_json FROM checkpoints WHERE id = ?",
                (checkpoint_id,)
            ).fetchone()
        
        if row:
            return FlowCheckpoint(**json.loads(row["data_json"]))
        return None
    
    def update(self, checkpoint: FlowCheckpoint) -> None:
        """Update existing checkpoint."""
        self.save(checkpoint)
    
    def delete(self, checkpoint_id: str) -> bool:
        """Delete checkpoint from database."""
        with self._lock:
            with self._connection() as conn:
                cursor = conn.execute(
                    "DELETE FROM checkpoints WHERE id = ?",
                    (checkpoint_id,)
                )
                return cursor.rowcount > 0
    
    def list_by_flow(
        self, 
        flow_id: str, 
        *, 
        status: CheckpointStatus | None = None,
        limit: int = 10,
        before: datetime | None = None,
    ) -> list[FlowCheckpoint]:
        """List checkpoints for a flow."""
        query = "SELECT data_json FROM checkpoints WHERE flow_id = ?"
        params: list = [flow_id]
        
        if status:
            query += " AND status = ?"
            params.append(status.value)
        
        if before:
            query += " AND created_at < ?"
            params.append(before.isoformat())
        
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        
        with self._connection() as conn:
            rows = conn.execute(query, params).fetchall()
        
        return [FlowCheckpoint(**json.loads(row["data_json"])) for row in rows]
    
    def get_latest(
        self, 
        flow_id: str, 
        *, 
        status: CheckpointStatus | None = None
    ) -> FlowCheckpoint | None:
        """Get most recent checkpoint."""
        results = self.list_by_flow(flow_id, status=status, limit=1)
        return results[0] if results else None
    
    def cleanup(
        self, 
        flow_id: str, 
        *, 
        keep_count: int = 10,
        older_than_days: int | None = None
    ) -> int:
        """Efficient bulk cleanup."""
        deleted = 0
        
        with self._lock:
            with self._connection() as conn:
                # Get IDs to keep (most recent)
                keep_ids = [
                    row["id"] for row in conn.execute(
                        "SELECT id FROM checkpoints WHERE flow_id = ? "
                        "ORDER BY created_at DESC LIMIT ?",
                        (flow_id, keep_count)
                    ).fetchall()
                ]
                
                # Delete by count
                if keep_ids:
                    placeholders = ",".join("?" * len(keep_ids))
                    cursor = conn.execute(
                        f"DELETE FROM checkpoints WHERE flow_id = ? "
                        f"AND id NOT IN ({placeholders})",
                        [flow_id] + keep_ids
                    )
                    deleted += cursor.rowcount
                
                # Delete by age
                if older_than_days:
                    cutoff = datetime.utcnow()
                    cutoff = cutoff.replace(
                        day=cutoff.day - older_than_days
                    )
                    cursor = conn.execute(
                        "DELETE FROM checkpoints WHERE flow_id = ? "
                        "AND created_at < ?",
                        (flow_id, cutoff.isoformat())
                    )
                    deleted += cursor.rowcount
        
        return deleted
```

---

## 5. Redis Backend

```python
# dynamiq/checkpoint/backends/redis.py

import json
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import redis

from dynamiq.checkpoint.backends.base import CheckpointBackend
from dynamiq.checkpoint.models import FlowCheckpoint, CheckpointStatus
from dynamiq.checkpoint.serializers import checkpoint_json_encoder

class RedisCheckpointBackend(CheckpointBackend):
    """
    Redis-based storage for production deployments.
    
    Features:
    - Fast reads/writes (sub-millisecond)
    - TTL support for automatic cleanup
    - Sorted sets for efficient listing
    - Distributed-safe (multiple pods)
    
    Key structure:
    - checkpoint:{id} → JSON checkpoint data
    - flow:{flow_id}:checkpoints → Sorted set (score=timestamp, member=id)
    - flow:{flow_id}:status:{status} → Sorted set for status filtering
    
    Example:
        >>> backend = RedisCheckpointBackend(
        ...     host="redis.prod.internal",
        ...     password=os.environ["REDIS_PASSWORD"],
        ... )
    """
    
    def __init__(
        self, 
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: str | None = None,
        prefix: str = "dynamiq:checkpoint:",
        default_ttl: int | None = 86400 * 7,  # 7 days
        **kwargs,
    ):
        """
        Initialize Redis backend.
        
        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password
            prefix: Key prefix for namespacing
            default_ttl: Default TTL in seconds (None for no expiry)
            **kwargs: Additional redis.Redis arguments
        """
        try:
            import redis
        except ImportError:
            raise ImportError(
                "redis package required: pip install redis"
            )
        
        self._redis = redis.Redis(
            host=host, 
            port=port, 
            db=db, 
            password=password,
            decode_responses=True,
            **kwargs,
        )
        self._prefix = prefix
        self._default_ttl = default_ttl
    
    def _checkpoint_key(self, checkpoint_id: str) -> str:
        """Key for checkpoint data."""
        return f"{self._prefix}checkpoint:{checkpoint_id}"
    
    def _flow_index_key(self, flow_id: str) -> str:
        """Key for flow's checkpoint index."""
        return f"{self._prefix}flow:{flow_id}:checkpoints"
    
    def _flow_status_key(self, flow_id: str, status: str) -> str:
        """Key for flow's status-filtered index."""
        return f"{self._prefix}flow:{flow_id}:status:{status}"
    
    def save(self, checkpoint: FlowCheckpoint) -> str:
        """Save checkpoint to Redis."""
        checkpoint.updated_at = datetime.utcnow()
        
        key = self._checkpoint_key(checkpoint.id)
        data = json.dumps(
            checkpoint.model_dump(),
            default=checkpoint_json_encoder,
            ensure_ascii=False,
        )
        
        # Use pipeline for atomic operations
        pipe = self._redis.pipeline()
        
        # Save checkpoint data
        if self._default_ttl:
            pipe.setex(key, self._default_ttl, data)
        else:
            pipe.set(key, data)
        
        # Update flow index (sorted set with timestamp score)
        index_key = self._flow_index_key(checkpoint.flow_id)
        score = checkpoint.created_at.timestamp()
        pipe.zadd(index_key, {checkpoint.id: score})
        
        # Update status index
        status_key = self._flow_status_key(
            checkpoint.flow_id, 
            checkpoint.status.value
        )
        pipe.zadd(status_key, {checkpoint.id: score})
        
        # Set TTL on indexes
        if self._default_ttl:
            pipe.expire(index_key, self._default_ttl)
            pipe.expire(status_key, self._default_ttl)
        
        pipe.execute()
        
        return checkpoint.id
    
    def load(self, checkpoint_id: str) -> FlowCheckpoint | None:
        """Load checkpoint from Redis."""
        key = self._checkpoint_key(checkpoint_id)
        data = self._redis.get(key)
        
        if data:
            return FlowCheckpoint(**json.loads(data))
        return None
    
    def update(self, checkpoint: FlowCheckpoint) -> None:
        """Update existing checkpoint."""
        self.save(checkpoint)
    
    def delete(self, checkpoint_id: str) -> bool:
        """Delete checkpoint from Redis."""
        # Load to get flow_id for index cleanup
        checkpoint = self.load(checkpoint_id)
        if not checkpoint:
            return False
        
        pipe = self._redis.pipeline()
        pipe.delete(self._checkpoint_key(checkpoint_id))
        pipe.zrem(
            self._flow_index_key(checkpoint.flow_id), 
            checkpoint_id
        )
        pipe.zrem(
            self._flow_status_key(checkpoint.flow_id, checkpoint.status.value),
            checkpoint_id
        )
        results = pipe.execute()
        
        return results[0] > 0
    
    def list_by_flow(
        self, 
        flow_id: str, 
        *, 
        status: CheckpointStatus | None = None,
        limit: int = 10,
        before: datetime | None = None,
    ) -> list[FlowCheckpoint]:
        """List checkpoints for a flow."""
        # Choose index based on status filter
        if status:
            index_key = self._flow_status_key(flow_id, status.value)
        else:
            index_key = self._flow_index_key(flow_id)
        
        # Get checkpoint IDs from sorted set (newest first)
        if before:
            max_score = before.timestamp()
            checkpoint_ids = self._redis.zrevrangebyscore(
                index_key, 
                max_score, 
                "-inf", 
                start=0, 
                num=limit
            )
        else:
            checkpoint_ids = self._redis.zrevrange(
                index_key, 
                0, 
                limit - 1
            )
        
        # Load checkpoints (use pipeline for efficiency)
        if not checkpoint_ids:
            return []
        
        pipe = self._redis.pipeline()
        for cp_id in checkpoint_ids:
            pipe.get(self._checkpoint_key(cp_id))
        
        results = pipe.execute()
        
        checkpoints = []
        for data in results:
            if data:
                checkpoints.append(FlowCheckpoint(**json.loads(data)))
        
        return checkpoints
    
    def get_latest(
        self, 
        flow_id: str, 
        *, 
        status: CheckpointStatus | None = None
    ) -> FlowCheckpoint | None:
        """Get most recent checkpoint."""
        results = self.list_by_flow(flow_id, status=status, limit=1)
        return results[0] if results else None
    
    def cleanup(
        self, 
        flow_id: str, 
        *, 
        keep_count: int = 10,
        older_than_days: int | None = None
    ) -> int:
        """Efficient cleanup using Redis operations."""
        deleted = 0
        index_key = self._flow_index_key(flow_id)
        
        # Get all checkpoint IDs
        all_ids = self._redis.zrevrange(index_key, 0, -1)
        
        # Determine which to delete
        ids_to_delete = []
        
        if len(all_ids) > keep_count:
            ids_to_delete.extend(all_ids[keep_count:])
        
        if older_than_days:
            cutoff = datetime.utcnow().timestamp() - (older_than_days * 86400)
            old_ids = self._redis.zrangebyscore(index_key, "-inf", cutoff)
            ids_to_delete.extend(old_ids)
        
        # Deduplicate
        ids_to_delete = list(set(ids_to_delete))
        
        # Delete in batches
        for cp_id in ids_to_delete:
            if self.delete(cp_id):
                deleted += 1
        
        return deleted
```

---

## 6. PostgreSQL Backend

```python
# dynamiq/checkpoint/backends/postgres.py

import json
from datetime import datetime
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import psycopg2
    from psycopg2 import pool

from dynamiq.checkpoint.backends.base import CheckpointBackend
from dynamiq.checkpoint.models import FlowCheckpoint, CheckpointStatus
from dynamiq.checkpoint.serializers import checkpoint_json_encoder

class PostgresCheckpointBackend(CheckpointBackend):
    """
    PostgreSQL-based storage for production with strong durability.
    
    Features:
    - ACID transactions
    - JSONB for efficient storage and querying
    - Connection pooling
    - Works with existing runtime PostgreSQL
    
    Example:
        >>> backend = PostgresCheckpointBackend(
        ...     connection_string="postgresql://user:pass@host:5432/db"
        ... )
        
        # Or with existing SQLAlchemy session (runtime integration):
        >>> backend = PostgresCheckpointBackend.from_session(session)
    """
    
    def __init__(
        self,
        connection_string: str | None = None,
        table_name: str = "dynamiq_checkpoints",
        pool_size: int = 5,
        pool: "pool.ThreadedConnectionPool | None" = None,
    ):
        """
        Initialize PostgreSQL backend.
        
        Args:
            connection_string: PostgreSQL connection string
            table_name: Table name for checkpoints
            pool_size: Connection pool size
            pool: Existing connection pool (for runtime integration)
        """
        try:
            import psycopg2
            from psycopg2 import pool as pg_pool
        except ImportError:
            raise ImportError(
                "psycopg2 package required: pip install psycopg2-binary"
            )
        
        self._table_name = table_name
        
        if pool:
            self._pool = pool
            self._owns_pool = False
        else:
            self._pool = pg_pool.ThreadedConnectionPool(
                1, pool_size, connection_string
            )
            self._owns_pool = True
        
        self._init_schema()
    
    @contextmanager
    def _connection(self):
        """Get a connection from pool."""
        conn = self._pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._pool.putconn(conn)
    
    def _init_schema(self) -> None:
        """Create table and indexes if not exist."""
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self._table_name} (
                        id TEXT PRIMARY KEY,
                        flow_id TEXT NOT NULL,
                        run_id TEXT NOT NULL,
                        status TEXT NOT NULL,
                        data JSONB NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL,
                        updated_at TIMESTAMPTZ NOT NULL
                    )
                """)
                
                # Indexes
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self._table_name}_flow_created 
                    ON {self._table_name}(flow_id, created_at DESC)
                """)
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self._table_name}_flow_status 
                    ON {self._table_name}(flow_id, status, created_at DESC)
                """)
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self._table_name}_run 
                    ON {self._table_name}(run_id)
                """)
    
    def save(self, checkpoint: FlowCheckpoint) -> str:
        """Save checkpoint to PostgreSQL."""
        checkpoint.updated_at = datetime.utcnow()
        data = json.dumps(
            checkpoint.model_dump(),
            default=checkpoint_json_encoder,
            ensure_ascii=False,
        )
        
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"""
                    INSERT INTO {self._table_name} 
                    (id, flow_id, run_id, status, data, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        status = EXCLUDED.status,
                        data = EXCLUDED.data,
                        updated_at = EXCLUDED.updated_at
                """, (
                    checkpoint.id,
                    checkpoint.flow_id,
                    checkpoint.run_id,
                    checkpoint.status.value,
                    data,
                    checkpoint.created_at,
                    checkpoint.updated_at,
                ))
        
        return checkpoint.id
    
    def load(self, checkpoint_id: str) -> FlowCheckpoint | None:
        """Load checkpoint from PostgreSQL."""
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT data FROM {self._table_name} WHERE id = %s",
                    (checkpoint_id,)
                )
                row = cur.fetchone()
        
        if row:
            # PostgreSQL returns JSONB as dict
            data = row[0] if isinstance(row[0], dict) else json.loads(row[0])
            return FlowCheckpoint(**data)
        return None
    
    def update(self, checkpoint: FlowCheckpoint) -> None:
        """Update existing checkpoint."""
        self.save(checkpoint)
    
    def delete(self, checkpoint_id: str) -> bool:
        """Delete checkpoint from PostgreSQL."""
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"DELETE FROM {self._table_name} WHERE id = %s",
                    (checkpoint_id,)
                )
                return cur.rowcount > 0
    
    def list_by_flow(
        self, 
        flow_id: str, 
        *, 
        status: CheckpointStatus | None = None,
        limit: int = 10,
        before: datetime | None = None,
    ) -> list[FlowCheckpoint]:
        """List checkpoints for a flow."""
        query = f"SELECT data FROM {self._table_name} WHERE flow_id = %s"
        params: list = [flow_id]
        
        if status:
            query += " AND status = %s"
            params.append(status.value)
        
        if before:
            query += " AND created_at < %s"
            params.append(before)
        
        query += " ORDER BY created_at DESC LIMIT %s"
        params.append(limit)
        
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
        
        checkpoints = []
        for row in rows:
            data = row[0] if isinstance(row[0], dict) else json.loads(row[0])
            checkpoints.append(FlowCheckpoint(**data))
        
        return checkpoints
    
    def get_latest(
        self, 
        flow_id: str, 
        *, 
        status: CheckpointStatus | None = None
    ) -> FlowCheckpoint | None:
        """Get most recent checkpoint."""
        results = self.list_by_flow(flow_id, status=status, limit=1)
        return results[0] if results else None
    
    def close(self) -> None:
        """Close connection pool if we own it."""
        if self._owns_pool:
            self._pool.closeall()
```

---

## 7. Backend Selection Guide

| Scenario | Recommended Backend | Reason |
|----------|---------------------|--------|
| Local development | File | Human-readable, easy debugging |
| Unit tests | SQLite (`:memory:`) | Fast, isolated, no cleanup |
| Integration tests | SQLite (file) | Persistent, inspectable |
| Single-pod production | SQLite/PostgreSQL | Simple, durable |
| Multi-pod production | Redis | Fast, distributed |
| High durability needs | PostgreSQL | ACID, queryable |
| Runtime integration | PostgreSQL | Existing infrastructure |

---

## 8. Performance Benchmarks

Expected performance (approximate):

| Operation | File | SQLite | Redis | PostgreSQL |
|-----------|------|--------|-------|------------|
| Save (small) | 5-10ms | 2-5ms | 1-2ms | 3-8ms |
| Save (large 1MB) | 50-100ms | 20-50ms | 10-30ms | 30-80ms |
| Load | 2-5ms | 1-3ms | 1-2ms | 2-5ms |
| List (10 items) | 20-50ms | 5-10ms | 2-5ms | 5-15ms |
| Delete | 2-5ms | 1-3ms | 1-2ms | 2-5ms |

---

**Previous:** [05-DATA-MODELS.md](./05-DATA-MODELS.md)  
**Next:** [07-FLOW-INTEGRATION.md](./07-FLOW-INTEGRATION.md)
