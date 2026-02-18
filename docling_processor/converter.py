"""
Document Converter
Wraps Docling to convert any supported file format into a structured dict + metadata.
"""

import os
import hashlib
import logging
import mimetypes
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Tuple

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

logger = logging.getLogger("converter")

# File extensions that Docling handles well
DOCLING_SUPPORTED = {
    ".pdf", ".docx", ".doc", ".pptx", ".ppt",
    ".xlsx", ".xls", ".html", ".htm", ".txt",
    ".md", ".rtf", ".odt", ".ods", ".odp",
    ".epub", ".xml"
}

# Extensions we handle with fallback plain-text reading
PLAINTEXT_FALLBACK = {".txt", ".md", ".csv", ".json", ".xml", ".log", ".yaml", ".yml"}


def _file_hash(path: str) -> str:
    """SHA-256 hash of file for dedup / change detection."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_metadata(file_path: str, company_id: str) -> Dict[str, Any]:
    """Collect file-level metadata to attach to every chunk."""
    stat = os.stat(file_path)
    ext = Path(file_path).suffix.lower()
    mime, _ = mimetypes.guess_type(file_path)
    return {
        "company_id": company_id,
        "file_name": os.path.basename(file_path),
        "file_path": file_path,
        "file_extension": ext,
        "mime_type": mime or "application/octet-stream",
        "file_size_bytes": stat.st_size,
        "file_hash_sha256": _file_hash(file_path),
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "source_type": _classify_source(ext),
    }


def _classify_source(ext: str) -> str:
    mapping = {
        ".pdf": "pdf",
        ".docx": "word", ".doc": "word", ".odt": "word", ".rtf": "word",
        ".pptx": "presentation", ".ppt": "presentation", ".odp": "presentation",
        ".xlsx": "spreadsheet", ".xls": "spreadsheet", ".ods": "spreadsheet",
        ".csv": "spreadsheet",
        ".html": "web", ".htm": "web",
        ".txt": "text", ".md": "markdown",
        ".json": "data", ".xml": "data", ".yaml": "data", ".yml": "data",
        ".epub": "ebook",
    }
    return mapping.get(ext, "unknown")


def _plaintext_fallback(file_path: str) -> Dict[str, Any]:
    """Read plain-text file and wrap in a minimal doc_json structure."""
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    texts = []
    for i, line in enumerate(content.splitlines()):
        line = line.strip()
        if not line:
            continue
        texts.append({
            "self_ref": f"#/texts/{i}",
            "label": "text",
            "text": line,
            "prov": []
        })

    return {
        "schema_name": "DoclingDocument",
        "version": "1.0.0",
        "name": os.path.basename(file_path),
        "texts": texts,
        "pictures": [],
        "tables": [],
        "groups": [],
        "body": {"children": [{"$ref": f"#/texts/{i}"} for i in range(len(texts))]},
    }


def convert_file(file_path: str, company_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Convert any supported file to Docling doc_json + file metadata.

    Args:
        file_path:  Absolute path to the file
        company_id: Tenant identifier

    Returns:
        (doc_json, file_metadata) tuple

    Raises:
        Exception on unrecoverable conversion failure
    """
    ext = Path(file_path).suffix.lower()
    metadata = _build_metadata(file_path, company_id)

    # ── Plain-text fast path ──────────────────────────────────────────────────
    if ext in PLAINTEXT_FALLBACK and ext not in {".html", ".htm", ".xml"}:
        logger.info(f"Using plain-text fallback for: {ext}")
        doc_json = _plaintext_fallback(file_path)
        doc_json["name"] = metadata["file_name"]
        return doc_json, metadata

    # ── Docling conversion ────────────────────────────────────────────────────
    try:
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True          # enable OCR for scanned PDFs
        pipeline_options.do_table_structure = True

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        result = converter.convert(file_path)
        doc_json = result.document.export_to_dict()
        doc_json["name"] = metadata["file_name"]
        return doc_json, metadata

    except Exception as e:
        # If docling fails for a non-critical reason and file is text-ish, try fallback
        if ext in PLAINTEXT_FALLBACK:
            logger.warning(f"Docling failed, using text fallback. Reason: {e}")
            doc_json = _plaintext_fallback(file_path)
            doc_json["name"] = metadata["file_name"]
            return doc_json, metadata
        raise