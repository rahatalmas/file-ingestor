"""
Qdrant Client Wrapper
Multi-tenant ingestion: one collection per company_id.
Handles collection creation, schema, and batched upserts.
"""

import os
import logging
import time
from typing import List, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    PayloadSchemaType,
    CreateCollection,
    OptimizersConfigDiff,
    HnswConfigDiff,
)

logger = logging.getLogger("qdrant")

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", None)   # None for local
VECTOR_DIM = 3072         # text-embedding-3-large
UPSERT_BATCH_SIZE = 100   # points per batch


def _collection_name(company_id: str) -> str:
    """Stable, safe collection name per tenant."""
    return f"company_{company_id.replace('-', '_')}"


def _get_client() -> QdrantClient:
    return QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=60,
    )


def _ensure_collection(client: QdrantClient, company_id: str) -> str:
    """Create the company collection if it doesn't exist. Returns collection name."""
    name = _collection_name(company_id)

    existing = {c.name for c in client.get_collections().collections}
    if name in existing:
        return name

    logger.info(f"Creating Qdrant collection: {name}")
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(
            size=VECTOR_DIM,
            distance=Distance.COSINE,
            on_disk=False,
        ),
        hnsw_config=HnswConfigDiff(
            m=16,
            ef_construct=100,
            full_scan_threshold=10000,
        ),
        optimizers_config=OptimizersConfigDiff(
            indexing_threshold=20000,
        ),
        on_disk_payload=True,   # keep payloads on disk to save RAM
    )

    # Create payload indexes for common filter fields
    indexed_fields = {
        "company_id": PayloadSchemaType.KEYWORD,
        "doc_id": PayloadSchemaType.KEYWORD,
        "file_name": PayloadSchemaType.KEYWORD,
        "source_type": PayloadSchemaType.KEYWORD,
        "intent": PayloadSchemaType.KEYWORD,
        "label": PayloadSchemaType.KEYWORD,
        "section_title": PayloadSchemaType.TEXT,
        "ingested_at": PayloadSchemaType.KEYWORD,
        "page_number": PayloadSchemaType.INTEGER,
        "chunk_index": PayloadSchemaType.INTEGER,
    }
    for field, schema in indexed_fields.items():
        try:
            client.create_payload_index(
                collection_name=name,
                field_name=field,
                field_schema=schema,
            )
        except Exception as e:
            logger.warning(f"Could not create index for '{field}': {e}")

    logger.info(f"Collection '{name}' created with payload indexes.")
    return name


def _chunk_to_point(chunk: Dict[str, Any]) -> PointStruct:
    """Convert a chunk dict to a Qdrant PointStruct."""
    embedding = chunk.pop("embedding")   # remove from payload

    # Qdrant point ID must be a UUID string or unsigned int
    point_id = chunk["chunk_id"]

    # Everything else goes into payload
    payload = {k: v for k, v in chunk.items() if v is not None}

    return PointStruct(
        id=point_id,
        vector=embedding,
        payload=payload,
    )


def upsert_chunks(chunks: List[Dict[str, Any]], company_id: str) -> int:
    """
    Upsert embedded chunks into the company's Qdrant collection.

    Args:
        chunks:     List of chunk dicts with 'embedding' field
        company_id: Tenant identifier

    Returns:
        Number of points successfully upserted

    Raises:
        Exception on unrecoverable Qdrant error
    """
    if not chunks:
        logger.warning("upsert_chunks called with empty list.")
        return 0

    client = _get_client()
    collection = _ensure_collection(client, company_id)
    total = len(chunks)
    upserted = 0

    logger.info(f"Upserting {total} points to collection '{collection}'...")

    for batch_start in range(0, total, UPSERT_BATCH_SIZE):
        batch = chunks[batch_start: batch_start + UPSERT_BATCH_SIZE]
        points = [_chunk_to_point(c) for c in batch]

        for attempt in range(1, 4):
            try:
                client.upsert(
                    collection_name=collection,
                    points=points,
                    wait=True,
                )
                upserted += len(points)
                logger.info(
                    f"  Upserted batch {batch_start // UPSERT_BATCH_SIZE + 1}/"
                    f"{(total + UPSERT_BATCH_SIZE - 1) // UPSERT_BATCH_SIZE}"
                )
                break
            except Exception as e:
                if attempt == 3:
                    raise
                wait = 2 ** attempt
                logger.warning(
                    f"Qdrant upsert error (attempt {attempt}/3): {e}. Retrying in {wait}s..."
                )
                time.sleep(wait)

    logger.info(f"Qdrant upsert complete: {upserted}/{total} points.")
    return upserted