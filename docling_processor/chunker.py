"""
Enterprise Dynamic Chunker
Produces semantically coherent, metadata-rich chunks from a Docling doc_json.
Handles: text, headings, lists, tables, captions, code, formulas, figures.
"""

import re
import uuid
import logging
from typing import Any, Dict, List, Optional, Tuple

import tiktoken

logger = logging.getLogger("chunker")

# ── Token budget constants ─────────────────────────────────────────────────────
DEFAULT_MAX_TOKENS = 400        # soft ceiling per chunk
OVERLAP_TOKENS = 60             # previous-chunk context to prepend
MIN_CHUNK_TOKENS = 20           # don't emit tiny orphan chunks
TABLE_MAX_TOKENS = 600          # tables can be larger
CODE_MAX_TOKENS = 500

ENCODER = tiktoken.get_encoding("cl100k_base")  # matches text-embedding-3-*


# ── Label classification ───────────────────────────────────────────────────────
HEADING_LABELS = {"section_header", "subsection_header", "title"}
BODY_LABELS = {"text", "paragraph"}
LIST_LABELS = {"list_item", "list"}
TABLE_LABELS = {"table"}
CODE_LABELS = {"code", "formula", "equation"}
CAPTION_LABELS = {"caption", "figure_caption", "table_caption"}
SKIP_LABELS = {"page_header", "page_footer", "page_number", "picture"}


def _count_tokens(text: str) -> int:
    return len(ENCODER.encode(text))


def _chunk_id() -> str:
    return str(uuid.uuid4())


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ── Docling JSON flattener ─────────────────────────────────────────────────────

def _resolve_ref(ref: str, doc_json: Dict) -> Optional[Dict]:
    """Resolve a $ref pointer like '#/texts/3' or '#/tables/1'."""
    try:
        parts = ref.lstrip("#/").split("/")
        obj = doc_json
        for p in parts:
            if isinstance(obj, list):
                obj = obj[int(p)]
            else:
                obj = obj[p]
        return obj
    except (KeyError, IndexError, ValueError, TypeError):
        return None


def _flatten_group(group: Dict, doc_json: Dict) -> List[Dict]:
    """Recursively expand a group's children into a flat element list."""
    elements = []
    for child_ref in group.get("children", []):
        ref = child_ref.get("$ref", "")
        elem = _resolve_ref(ref, doc_json)
        if elem is None:
            continue
        if ref.startswith("#/groups/"):
            elements.extend(_flatten_group(elem, doc_json))
        else:
            elem["_ref"] = ref
            elements.append(elem)
    return elements


def _flatten_document(doc_json: Dict) -> List[Dict]:
    """Flatten Docling doc_json body into an ordered list of elements."""
    elements = []
    body_children = doc_json.get("body", {}).get("children", [])

    for child_ref in body_children:
        ref = child_ref.get("$ref", "")
        if not ref:
            continue
        elem = _resolve_ref(ref, doc_json)
        if elem is None:
            continue
        if ref.startswith("#/groups/"):
            elements.extend(_flatten_group(elem, doc_json))
        else:
            elem["_ref"] = ref
            elements.append(elem)

    return elements


# ── Table serialiser ───────────────────────────────────────────────────────────

def _table_to_markdown(table: Dict) -> str:
    """Convert a Docling table dict to a Markdown table string."""
    try:
        grid = table.get("data", {}).get("grid", [])
        if not grid:
            return table.get("text", "")

        rows = []
        for row in grid:
            cells = [c.get("text", "").replace("|", "\\|").strip() for c in row]
            rows.append("| " + " | ".join(cells) + " |")
            if len(rows) == 1:
                # Header separator
                rows.append("| " + " | ".join(["---"] * len(cells)) + " |")
        return "\n".join(rows)
    except Exception:
        return table.get("text", "")


# ── Intent classifier ──────────────────────────────────────────────────────────

def _infer_intent(text: str, label: str, section_path: str) -> str:
    """Heuristic intent tag to improve retrieval relevance."""
    text_lower = text.lower()
    path_lower = section_path.lower()

    if label in HEADING_LABELS:
        return "navigation"
    if label in TABLE_LABELS:
        return "tabular_data"
    if label in CODE_LABELS:
        return "code_or_formula"

    # Keyword heuristics
    if any(k in text_lower for k in ["policy", "rule", "regulation", "compliance", "must", "shall"]):
        return "policy_or_rule"
    if any(k in text_lower for k in ["how to", "step", "procedure", "process", "guide", "instruction"]):
        return "procedural"
    if any(k in text_lower for k in ["price", "cost", "fee", "rate", "$", "discount", "payment"]):
        return "pricing"
    if any(k in text_lower for k in ["contact", "phone", "email", "address", "location", "outlet"]):
        return "contact_or_location"
    if any(k in text_lower for k in ["feature", "benefit", "advantage", "offer", "product", "service"]):
        return "product_or_service"
    if any(k in text_lower for k in ["faq", "question", "answer", "q:", "a:"]):
        return "faq"
    if any(k in text_lower for k in ["introduction", "overview", "about", "background"]):
        return "overview"
    if any(k in text_lower for k in ["summary", "conclusion", "result", "finding"]):
        return "summary"
    return "informational"


# ── Main chunker ───────────────────────────────────────────────────────────────

def chunk_document(
    doc_json: Dict,
    file_metadata: Dict,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> List[Dict]:
    """
    Produce enterprise-grade RAG chunks from a Docling doc_json.

    Each chunk contains:
      - text content
      - rich metadata (company_id, doc_id, section path, intent, label, etc.)
      - position info for dedup and ordering

    Args:
        doc_json:      Docling export_to_dict() output
        file_metadata: Dict from converter._build_metadata()
        max_tokens:    Soft token ceiling per chunk
        overlap_tokens: Tokens to include from previous chunk for context

    Returns:
        List of chunk dicts ready for embedding + Qdrant upsert
    """
    company_id = file_metadata["company_id"]
    doc_id = file_metadata["file_hash_sha256"][:16]   # short stable id
    file_name = file_metadata["file_name"]

    elements = _flatten_document(doc_json)
    if not elements:
        logger.warning(f"No elements found in document: {file_name}")
        return []

    chunks: List[Dict] = []
    section_stack: List[str] = [file_name]   # breadcrumb path
    section_level: Dict[int, str] = {}       # level → heading text

    current_text = ""
    current_tokens = 0
    current_labels: List[str] = []
    current_page: Optional[int] = None
    overlap_buffer = ""     # last N tokens of previous chunk
    chunk_index = 0

    def _flush(force_label: Optional[str] = None) -> None:
        nonlocal current_text, current_tokens, current_labels, chunk_index, overlap_buffer

        text = _clean_text(current_text)
        if not text or current_tokens < MIN_CHUNK_TOKENS:
            current_text = ""
            current_tokens = 0
            current_labels = []
            return

        section_path = " > ".join(section_stack)
        dominant_label = force_label or (current_labels[0] if current_labels else "text")
        intent = _infer_intent(text, dominant_label, section_path)

        chunk = {
            # ── Identity ──────────────────────────────────────
            "chunk_id": _chunk_id(),
            "doc_id": doc_id,
            "company_id": company_id,
            "chunk_index": chunk_index,

            # ── Content ───────────────────────────────────────
            "text": text,
            "token_count": current_tokens,

            # ── Structure ─────────────────────────────────────
            "section_path": section_path,
            "section_title": section_stack[-1] if section_stack else "",
            "label": dominant_label,
            "intent": intent,

            # ── Document context ──────────────────────────────
            "prev_chunk_summary": overlap_buffer[:300] if overlap_buffer else "",
            "page_number": current_page,

            # ── File metadata ─────────────────────────────────
            "file_name": file_metadata["file_name"],
            "file_extension": file_metadata["file_extension"],
            "source_type": file_metadata["source_type"],
            "mime_type": file_metadata["mime_type"],
            "file_size_bytes": file_metadata["file_size_bytes"],
            "file_hash_sha256": file_metadata["file_hash_sha256"],
            "ingested_at": file_metadata["ingested_at"],
        }

        chunks.append(chunk)

        # Prepare overlap buffer snapped to word boundary.
        # Raw token decoding can split mid-word (e.g. "ensingh" from "Mymensingh").
        encoded = ENCODER.encode(text)
        overlap_tokens_actual = min(overlap_tokens, len(encoded))
        raw_overlap = ENCODER.decode(encoded[-overlap_tokens_actual:])
        if raw_overlap and not raw_overlap[0].isspace() and raw_overlap != text:
            space_idx = raw_overlap.find(" ")
            overlap_buffer = raw_overlap[space_idx + 1:] if space_idx != -1 else raw_overlap
        else:
            overlap_buffer = raw_overlap.lstrip()

        chunk_index += 1
        current_text = ""
        current_tokens = 0
        current_labels = []

    # NOTE: overlap_buffer is stored ONLY in prev_chunk_summary metadata.
    # It is intentionally NOT prepended to chunk text — that causes duplication
    # in both the text field and prev_chunk_summary, which degrades RAG quality.

    for elem in elements:
        label = elem.get("label", "text")
        prov = elem.get("prov", [{}])
        page = prov[0].get("page_no") if prov else None
        if page:
            current_page = page

        # ── Skip non-content labels ───────────────────────────────────────────
        if label in SKIP_LABELS:
            continue

        # ── Heading: flush current chunk and update breadcrumb ────────────────
        if label in HEADING_LABELS:
            _flush()
            heading_text = _clean_text(elem.get("text", ""))
            if not heading_text:
                continue

            # Determine heading depth from label or level field
            level = elem.get("level", 1)
            if label == "title":
                level = 0
            elif label == "subsection_header":
                level = 2

            # Trim stack to current level
            if level == 0:
                section_stack.clear()
                section_stack.append(file_name)
            else:
                # Pop headings at same or deeper level
                for lvl in sorted(list(section_level.keys()), reverse=True):
                    if lvl >= level:
                        if section_level[lvl] in section_stack:
                            section_stack.remove(section_level[lvl])
                        del section_level[lvl]

            section_stack.append(heading_text)
            section_level[level] = heading_text
            continue

        # ── Table: serialize and emit as standalone chunk ─────────────────────
        if label in TABLE_LABELS:
            _flush()
            table_md = _table_to_markdown(elem)
            table_tokens = _count_tokens(table_md)

            # Split large tables into row-batches
            if table_tokens > TABLE_MAX_TOKENS:
                rows = table_md.split("\n")
                header = rows[:2]  # markdown header + separator
                batch_rows = header[:]
                batch_tokens = _count_tokens("\n".join(batch_rows))

                for row in rows[2:]:
                    row_tokens = _count_tokens(row)
                    if batch_tokens + row_tokens > TABLE_MAX_TOKENS:
                        current_text = "\n".join(batch_rows)
                        current_tokens = batch_tokens
                        current_labels = ["table"]
                        _flush(force_label="table")
                        batch_rows = header + [row]
                        batch_tokens = _count_tokens("\n".join(batch_rows))
                    else:
                        batch_rows.append(row)
                        batch_tokens += row_tokens

                if len(batch_rows) > 2:
                    current_text = "\n".join(batch_rows)
                    current_tokens = batch_tokens
                    current_labels = ["table"]
                    _flush(force_label="table")
            else:
                current_text = table_md
                current_tokens = table_tokens
                current_labels = ["table"]
                _flush(force_label="table")
            continue

        # ── Code / Formula: emit standalone ───────────────────────────────────
        if label in CODE_LABELS:
            _flush()
            code_text = _clean_text(elem.get("text", ""))
            if code_text:
                current_text = f"```\n{code_text}\n```"
                current_tokens = _count_tokens(current_text)
                current_labels = [label]
                _flush(force_label=label)
            continue

        # ── Caption: append to current chunk ─────────────────────────────────
        if label in CAPTION_LABELS:
            cap_text = _clean_text(elem.get("text", ""))
            if cap_text:
                cap_tokens = _count_tokens(cap_text)
                current_text += (" " if current_text else "") + cap_text
                current_tokens += cap_tokens
                if "caption" not in current_labels:
                    current_labels.append("caption")
            continue

        # ── Body text / list items ─────────────────────────────────────────────
        raw_text = elem.get("text", "")
        if not raw_text:
            # Try to pull text from nested items
            for item in elem.get("items", []):
                t = _clean_text(item.get("text", ""))
                if t:
                    raw_text += (" " if raw_text else "") + t

        text = _clean_text(raw_text)
        if not text:
            continue

        text_tokens = _count_tokens(text)
        budget = TABLE_MAX_TOKENS if label in TABLE_LABELS else max_tokens

        # Oversized single element: must be split by sentence
        if text_tokens > budget:
            _flush()
            _split_long_text(text, label, budget, overlap_tokens, chunks,
                             file_metadata, doc_id, company_id,
                             " > ".join(section_stack), current_page,
                             overlap_buffer, chunk_index)
            chunk_index = len(chunks)  # sync index
            overlap_buffer = ""
            continue

        # Normal accumulation
        if current_tokens + text_tokens > budget:
            _flush()
            # Overlap is stored in prev_chunk_summary only — NOT prepended to text

        current_text += (" " if current_text else "") + text
        current_tokens += text_tokens
        if label not in current_labels:
            current_labels.append(label)

    # Flush remaining
    _flush()

    logger.info(f"Chunking complete: {len(chunks)} chunks from '{file_name}'")
    return chunks


def _split_long_text(
    text: str,
    label: str,
    max_tokens: int,
    overlap_tokens: int,
    chunks: List[Dict],
    file_metadata: Dict,
    doc_id: str,
    company_id: str,
    section_path: str,
    page_number: Optional[int],
    overlap_buffer: str,
    chunk_index_start: int,
) -> None:
    """Split a long text element into multiple chunks by sentence boundary."""
    # Split on sentence endings
    sentences = re.split(r'(?<=[.!?])\s+', text)
    buffer = overlap_buffer + "\n" if overlap_buffer else ""
    buffer_tokens = _count_tokens(buffer)
    idx = chunk_index_start

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        s_tokens = _count_tokens(sentence)

        if buffer_tokens + s_tokens > max_tokens and buffer_tokens > 0:
            # Flush
            clean = _clean_text(buffer)
            if _count_tokens(clean) >= MIN_CHUNK_TOKENS:
                chunks.append(_build_chunk(
                    clean, label, doc_id, company_id, idx,
                    section_path, page_number, overlap_buffer, file_metadata
                ))
                idx += 1
                # Overlap snapped to word boundary
                encoded = ENCODER.encode(clean)
                ov = min(overlap_tokens, len(encoded))
                raw_ov = ENCODER.decode(encoded[-ov:])
                if raw_ov and not raw_ov[0].isspace() and raw_ov != clean:
                    si = raw_ov.find(" ")
                    overlap_buffer = raw_ov[si + 1:] if si != -1 else raw_ov
                else:
                    overlap_buffer = raw_ov.lstrip()
                buffer = overlap_buffer + "\n" if overlap_buffer else ""
                buffer_tokens = _count_tokens(buffer)

        buffer += sentence + " "
        buffer_tokens += s_tokens

    clean = _clean_text(buffer)
    if _count_tokens(clean) >= MIN_CHUNK_TOKENS:
        chunks.append(_build_chunk(
            clean, label, doc_id, company_id, idx,
            section_path, page_number, overlap_buffer, file_metadata
        ))


def _build_chunk(
    text: str,
    label: str,
    doc_id: str,
    company_id: str,
    index: int,
    section_path: str,
    page_number: Optional[int],
    prev_summary: str,
    file_metadata: Dict,
) -> Dict:
    """Construct a single chunk dict."""
    intent = _infer_intent(text, label, section_path)
    return {
        "chunk_id": _chunk_id(),
        "doc_id": doc_id,
        "company_id": company_id,
        "chunk_index": index,
        "text": text,
        "token_count": _count_tokens(text),
        "section_path": section_path,
        "section_title": section_path.split(" > ")[-1] if section_path else "",
        "label": label,
        "intent": intent,
        "prev_chunk_summary": prev_summary[:300] if prev_summary else "",
        "page_number": page_number,
        "file_name": file_metadata["file_name"],
        "file_extension": file_metadata["file_extension"],
        "source_type": file_metadata["source_type"],
        "mime_type": file_metadata["mime_type"],
        "file_size_bytes": file_metadata["file_size_bytes"],
        "file_hash_sha256": file_metadata["file_hash_sha256"],
        "ingested_at": file_metadata["ingested_at"],
    }