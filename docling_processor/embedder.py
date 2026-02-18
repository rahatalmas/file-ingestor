"""
OpenAI Embedder
Embeds chunks in batches using text-embedding-3-large with retry + rate-limit handling.
"""

import os
import time
import logging
from typing import List, Dict

import openai
from openai import OpenAI

logger = logging.getLogger("embedder")

EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM = 3072          # text-embedding-3-large dimension
BATCH_SIZE = 100              # OpenAI allows up to 2048 items; 100 is safe
MAX_RETRIES = 5
RETRY_BASE_DELAY = 2.0        # seconds, exponential backoff


def _get_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable is not set. "
            "Please set it before running the pipeline."
        )
    return OpenAI(api_key=api_key)


def _embed_batch(client: OpenAI, texts: List[str]) -> List[List[float]]:
    """Call OpenAI embedding API for a batch of texts with retry logic."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=texts,
                encoding_format="float",
            )
            # Sort by index to ensure order matches input
            sorted_data = sorted(response.data, key=lambda d: d.index)
            return [d.embedding for d in sorted_data]

        except openai.RateLimitError as e:
            wait = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            logger.warning(f"Rate limit hit (attempt {attempt}/{MAX_RETRIES}). Waiting {wait}s...")
            time.sleep(wait)

        except openai.APIConnectionError as e:
            wait = RETRY_BASE_DELAY * attempt
            logger.warning(f"Connection error (attempt {attempt}/{MAX_RETRIES}): {e}. Waiting {wait}s...")
            time.sleep(wait)

        except openai.APIStatusError as e:
            if e.status_code in (500, 502, 503, 504):
                wait = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(f"Server error {e.status_code} (attempt {attempt}/{MAX_RETRIES}). Waiting {wait}s...")
                time.sleep(wait)
            else:
                raise  # Non-retryable

    raise RuntimeError(f"Embedding failed after {MAX_RETRIES} attempts.")


def embed_chunks(chunks: List[Dict]) -> List[Dict]:
    """
    Add 'embedding' field to each chunk dict.

    Processes in batches of BATCH_SIZE to respect OpenAI limits.
    Truncates text to first 8000 chars as a safety guard (model limit is ~8191 tokens).

    Args:
        chunks: List of chunk dicts (must have 'text' field)

    Returns:
        Same list with 'embedding' field populated
    """
    if not chunks:
        return chunks

    client = _get_client()
    total = len(chunks)
    logger.info(f"Embedding {total} chunks using {EMBEDDING_MODEL}...")

    for batch_start in range(0, total, BATCH_SIZE):
        batch = chunks[batch_start: batch_start + BATCH_SIZE]
        texts = [c["text"][:8000] for c in batch]   # safety truncation

        logger.info(
            f"  Batch {batch_start // BATCH_SIZE + 1}/"
            f"{(total + BATCH_SIZE - 1) // BATCH_SIZE} "
            f"({len(texts)} items)"
        )

        embeddings = _embed_batch(client, texts)

        for chunk, embedding in zip(batch, embeddings):
            chunk["embedding"] = embedding

        # Polite pause between batches to avoid hammering rate limits
        if batch_start + BATCH_SIZE < total:
            time.sleep(0.2)

    logger.info(f"Embedding complete. Dimension: {len(chunks[0].get('embedding', []))}")
    return chunks