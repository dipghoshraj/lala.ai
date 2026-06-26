"""
vector/store.py — ChromaDB vector store wrapper for local development.

Supports two modes controlled by the ``chroma_mode`` config key:
  - ``"embedded"`` (default): persistent ChromaDB stored in a local directory.
    No server required — suitable for local dev and testing.
  - ``"server"``: connects to a standalone ChromaDB HTTP server
    (e.g. started via docker-compose).

Usage
-----
    from vector.store import VectorStore

    store = VectorStore.from_config({"mode": "embedded", "path": "./chroma_db"})
    store.add(ids=["doc1"], texts=["hello world"], metadatas=[{"source": "test"}])
    results = store.query("hello", n_results=5)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class VectorSearchResult:
    id: str
    text: str
    distance: float
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorStore:
    """
    Thin wrapper around a ChromaDB collection.

    All embedding is handled by ChromaDB's default embedding function
    (``sentence-transformers/all-MiniLM-L6-v2``), so no external embedding
    service is required for local development.
    """

    def __init__(self, client: Any, collection_name: str = "lala_vectors") -> None:
        self._client = client
        self._collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "VectorStore ready  collection=%s  count=%d",
            collection_name,
            self._collection.count(),
        )

    # ── Factory ──────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "VectorStore":
        """
        Build a VectorStore from a config dict.

        Config keys
        -----------
        mode            : "embedded" | "server"  (default: "embedded")
        path            : local directory for embedded mode            (default: "./chroma_db")
        host            : ChromaDB server host for server mode         (default: "localhost")
        port            : ChromaDB server port for server mode         (default: 8000)
        collection_name : name of the ChromaDB collection              (default: "lala_vectors")
        """
        import chromadb

        mode = cfg.get("mode", "embedded")
        collection_name = cfg.get("collection_name", "lala_vectors")

        if mode == "server":
            host = cfg.get("host", "localhost")
            port = int(cfg.get("port", 8000))
            logger.info("ChromaDB connecting to server  %s:%d", host, port)
            client = chromadb.HttpClient(host=host, port=port)
        else:
            path = cfg.get("path", "./chroma_db")
            logger.info("ChromaDB embedded  path=%s", path)
            client = chromadb.PersistentClient(path=path)

        return cls(client, collection_name)

    # ── Write ─────────────────────────────────────────────────────────────────

    def add(
        self,
        ids: list[str],
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """
        Add or update documents.  If an id already exists it is upserted.

        Parameters
        ----------
        ids       : unique string ID for each chunk
        texts     : raw text (ChromaDB will embed automatically)
        metadatas : optional per-chunk metadata dicts
        """
        if not ids:
            return
        metas = metadatas or [{}] * len(ids)
        self._collection.upsert(ids=ids, documents=texts, metadatas=metas)
        logger.debug("upserted %d chunks", len(ids))

    def delete(self, ids: list[str]) -> None:
        """Delete chunks by their IDs."""
        if ids:
            self._collection.delete(ids=ids)
            logger.debug("deleted %d chunks", len(ids))

    def delete_by_metadata(self, where: dict[str, Any]) -> None:
        """Delete chunks matching a metadata filter, e.g. ``{"source": "doc.md"}``."""
        self._collection.delete(where=where)

    # ── Read ──────────────────────────────────────────────────────────────────

    def query(
        self,
        text: str,
        n_results: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """
        Semantic search: embed ``text`` and return the closest ``n_results`` chunks.

        Parameters
        ----------
        text      : query string
        n_results : number of results to return
        where     : optional ChromaDB metadata filter

        Returns
        -------
        List of VectorSearchResult sorted by ascending cosine distance.
        """
        kwargs: dict[str, Any] = {
            "query_texts": [text],
            "n_results": min(n_results, max(self._collection.count(), 1)),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        raw = self._collection.query(**kwargs)

        results: list[VectorSearchResult] = []
        ids = raw.get("ids", [[]])[0]
        docs = raw.get("documents", [[]])[0]
        dists = raw.get("distances", [[]])[0]
        metas = raw.get("metadatas", [[]])[0]

        for chunk_id, doc, dist, meta in zip(ids, docs, dists, metas):
            results.append(
                VectorSearchResult(
                    id=chunk_id,
                    text=doc,
                    distance=dist,
                    metadata=meta or {},
                )
            )
        return results

    def count(self) -> int:
        """Return the number of chunks stored in the collection."""
        return self._collection.count()
