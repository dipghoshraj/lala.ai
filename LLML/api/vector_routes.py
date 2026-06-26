"""
api/vector_routes.py — FastAPI router for ChromaDB vector store operations.

Endpoints
---------
POST   /v1/vector/add             — upsert chunks into the vector store
POST   /v1/vector/search          — semantic similarity search
DELETE /v1/vector/chunks          — delete chunks by ID list
DELETE /v1/vector/documents/{source} — delete all chunks for a source document
GET    /v1/vector/count           — number of chunks in the collection
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

vector_router = APIRouter(prefix="/v1/vector", tags=["vector"])


# ── Request / response models ─────────────────────────────────────────────────

class AddRequest(BaseModel):
    ids: list[str] = Field(..., description="Unique ID for each chunk")
    texts: list[str] = Field(..., description="Text content for each chunk")
    metadatas: list[dict[str, Any]] | None = Field(
        default=None, description="Optional metadata per chunk"
    )


class AddResponse(BaseModel):
    added: int


class SearchRequest(BaseModel):
    query: str = Field(..., description="Query text for semantic similarity search")
    n_results: int = Field(default=5, ge=1, le=100)
    where: dict[str, Any] | None = Field(
        default=None, description="Optional ChromaDB metadata filter"
    )


class SearchResult(BaseModel):
    id: str
    text: str
    distance: float
    metadata: dict[str, Any]


class SearchResponse(BaseModel):
    results: list[SearchResult]


class DeleteChunksRequest(BaseModel):
    ids: list[str] = Field(..., description="Chunk IDs to delete")


class CountResponse(BaseModel):
    count: int


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_store(request: Request):
    store = getattr(request.app.state, "vector_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="Vector store not initialised")
    return store


# ── Routes ────────────────────────────────────────────────────────────────────

@vector_router.post("/add", response_model=AddResponse)
async def add_chunks(body: AddRequest, request: Request):
    """
    Upsert chunks into the vector store.

    ``ids``, ``texts``, and (optional) ``metadatas`` must all have the same length.
    Existing chunks with matching IDs are overwritten.
    """
    if len(body.ids) != len(body.texts):
        raise HTTPException(
            status_code=422,
            detail="ids and texts must have the same length",
        )
    if body.metadatas and len(body.metadatas) != len(body.ids):
        raise HTTPException(
            status_code=422,
            detail="metadatas length must match ids length",
        )

    store = _get_store(request)
    store.add(ids=body.ids, texts=body.texts, metadatas=body.metadatas)
    logger.info("vector/add  count=%d", len(body.ids))
    return AddResponse(added=len(body.ids))


@vector_router.post("/search", response_model=SearchResponse)
async def search_chunks(body: SearchRequest, request: Request):
    """
    Semantic similarity search.

    Returns up to ``n_results`` chunks ordered by ascending cosine distance
    (closer = more similar).
    """
    store = _get_store(request)
    raw = store.query(text=body.query, n_results=body.n_results, where=body.where)
    results = [
        SearchResult(
            id=r.id,
            text=r.text,
            distance=r.distance,
            metadata=r.metadata,
        )
        for r in raw
    ]
    logger.info("vector/search  query=%r  hits=%d", body.query[:60], len(results))
    return SearchResponse(results=results)


@vector_router.delete("/chunks")
async def delete_chunks(body: DeleteChunksRequest, request: Request):
    """Delete specific chunks by ID."""
    store = _get_store(request)
    store.delete(ids=body.ids)
    return {"deleted": len(body.ids)}


@vector_router.delete("/documents/{source:path}")
async def delete_document(source: str, request: Request):
    """
    Delete all vector chunks associated with a source document path.

    ``source`` is URL-decoded automatically by FastAPI.
    """
    store = _get_store(request)
    store.delete_by_metadata(where={"source": source})
    logger.info("vector/delete-document  source=%r", source)
    return {"deleted_source": source}


@vector_router.get("/count", response_model=CountResponse)
async def count_chunks(request: Request):
    """Return the total number of chunks in the vector store."""
    store = _get_store(request)
    return CountResponse(count=store.count())
