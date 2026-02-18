"""
Enterprise RAG Ingestor
Orchestrates: File → Docling → Chunks → OpenAI Embeddings → Qdrant (multi-tenant)
"""

import os
import shutil
import logging
import traceback
from datetime import datetime, timezone
from pathlib import Path
import json
from docling_processor.converter import convert_file
from docling_processor.chunker import chunk_document
from docling_processor.embedder import embed_chunks
from docling_processor.qdrant_client_wrapper import upsert_chunks

logger = logging.getLogger("ingestor")


def _get_status_dir(file_path: str, status: str) -> str:
    """
    Given a file in .../notprocessed/<file>, return the sibling status folder.
    status: 'processed' | 'failed_to_process'
    """
    notprocessed_dir = os.path.dirname(file_path)
    parent_dir = os.path.dirname(notprocessed_dir)
    status_dir = os.path.join(parent_dir, status)
    os.makedirs(status_dir, exist_ok=True)
    return status_dir


def _move_file(file_path: str, dest_dir: str) -> str:
    """Move file to destination dir, handling name collisions with timestamp."""
    filename = os.path.basename(file_path)
    dest_path = os.path.join(dest_dir, filename)

    # Avoid overwrite collision
    if os.path.exists(dest_path):
        stem, ext = os.path.splitext(filename)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        dest_path = os.path.join(dest_dir, f"{stem}_{ts}{ext}")

    shutil.move(file_path, dest_path)
    return dest_path


def ingest_file(file_path: str, company_id: str) -> bool:
    """
    Full ingestion pipeline for a single file.

    Steps:
      1. Convert file to structured JSON via Docling
      2. Chunk the document intelligently
      3. Embed chunks via OpenAI
      4. Upsert into Qdrant under the company's collection (multi-tenant)
      5. Move file to processed/ or failed_to_process/

    Returns:
        True on success, False on failure
    """
    file_name = os.path.basename(file_path)
    file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
    logger.info(
        f"[INGEST START] company={company_id} | file={file_name} | size={file_size} bytes"
    )

    try:
        # ── Step 1: Convert ──────────────────────────────────────────────────
        logger.info(f"[1/4] Converting: {file_name}")
        doc_json, file_metadata = convert_file(file_path, company_id)

        # ── Step 2: Chunk ────────────────────────────────────────────────────
        logger.info(f"[2/4] Chunking: {file_name}")
        chunks = chunk_document(doc_json, file_metadata)
        with open("chunks.json", "w") as f:
            json.dump(chunks, f, indent=2)

        logger.info(f"      → {len(chunks)} chunks created")

        if not chunks:
            raise ValueError("Chunker produced 0 chunks — document may be empty or unsupported.")

        # ── Step 3: Embed ────────────────────────────────────────────────────
        logger.info(f"[3/4] Embedding {len(chunks)} chunks via OpenAI...")
        chunks_with_embeddings = embed_chunks(chunks)
        with open("embedded.json", "w", encoding="utf-8") as f:
            json.dump(chunks_with_embeddings, f, ensure_ascii=False, indent=2) 

        # ── Step 4: Upsert to Qdrant ─────────────────────────────────────────
        logger.info(f"[4/4] Upserting to Qdrant | collection=company_{company_id}")
        upsert_chunks(chunks_with_embeddings, company_id)

        # ── Move to processed ────────────────────────────────────────────────
        dest_dir = _get_status_dir(file_path, "processed")
        dest = _move_file(file_path, dest_dir)
        logger.info(f"[INGEST SUCCESS] Moved to: {dest}")
        return True

    except Exception as exc:
        logger.error(
            f"[INGEST FAILED] company={company_id} | file={file_name}\n"
            f"{traceback.format_exc()}"
        )
        try:
            dest_dir = _get_status_dir(file_path, "failed_to_process")
            dest = _move_file(file_path, dest_dir)
            logger.info(f"Moved failed file to: {dest}")
        except Exception as move_err:
            logger.error(f"Could not move failed file: {move_err}")
        return False