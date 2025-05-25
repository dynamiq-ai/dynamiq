from datetime import datetime
from typing import Any

from .dry_run import DryRunMode


class DryRunResourceTracker:
    def __init__(self):
        self.collections = []
        self.documents = []
        self.metadata = {}

    def register_collection(self, name: str, vector_store: Any, mode: DryRunMode):
        self.collections.append(
            {
                "name": name,
                "vector_store": vector_store,
                "mode": mode,
                "created_at": datetime.now(),
                "cleanup_required": mode in [DryRunMode.TEMPORARY],
            }
        )

    def register_documents(self, doc_ids: list[str], vector_store: Any):
        self.documents.append({"doc_ids": doc_ids, "vector_store": vector_store, "uploaded_at": datetime.now()})

    def cleanup(self, mode: DryRunMode) -> dict[str, bool]:
        cleanup_results = {}

        if mode == DryRunMode.TEMPORARY:
            cleanup_results["documents"] = self._cleanup_documents()
            cleanup_results["collections"] = self._cleanup_collections()
        elif mode == DryRunMode.PERSISTENT:
            cleanup_results["documents"] = self._cleanup_documents()
            cleanup_results["collections_preserved"] = True
        else:
            cleanup_results["no_cleanup_required"] = True

        return cleanup_results

    def _cleanup_documents(self) -> bool:
        success = True
        for doc_batch in self.documents:
            try:
                if hasattr(doc_batch["vector_store"], "delete_documents"):
                    doc_batch["vector_store"].delete_documents(doc_batch["doc_ids"])
            except Exception:
                success = False
        return success

    def _cleanup_collections(self) -> bool:
        success = True
        for collection in self.collections:
            if collection["cleanup_required"]:
                try:
                    if hasattr(collection["vector_store"], "delete_collection"):
                        collection["vector_store"].delete_collection(collection["name"])
                except Exception:
                    success = False
        return success

    def get_collection_info(self, collection_name: str) -> dict[str, Any] | None:
        for collection in self.collections:
            if collection["name"] == collection_name:
                return collection
        return None

    def get_document_count(self) -> int:
        total = 0
        for doc_batch in self.documents:
            total += len(doc_batch["doc_ids"])
        return total

    def clear(self):
        self.collections.clear()
        self.documents.clear()
        self.metadata.clear()
