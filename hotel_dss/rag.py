import json
import math
import re
from pathlib import Path

import numpy as np
from pypdf import PdfReader


BASE_DIR = Path(__file__).resolve().parent.parent
RAG_DIR = BASE_DIR / "rag"
RAG_DIR.mkdir(parents=True, exist_ok=True)
REFERENCE_TEXT_PATH = RAG_DIR / "rm_reference.txt"
CHUNKS_PATH = RAG_DIR / "rm_chunks.jsonl"
EMBEDDING_CACHE_PATH = RAG_DIR / "rm_embeddings.npz"


def extract_pdf_to_text(pdf_path: str | Path, output_path: str | Path | None = None) -> str:
    reader = PdfReader(str(pdf_path))
    pages = []
    for page_idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append(f"\n\n[PAGE {page_idx}]\n{text}")
    full_text = "\n".join(pages)
    if output_path is not None:
        Path(output_path).write_text(full_text, encoding="utf-8")
    return full_text


def load_reference_text() -> str:
    if not REFERENCE_TEXT_PATH.exists():
        return ""
    return REFERENCE_TEXT_PATH.read_text(encoding="utf-8")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_into_chunks(text: str, chunk_chars: int = 1800, overlap_chars: int = 250) -> list[dict]:
    text = text.replace("\r\n", "\n")
    sections = re.split(r"\n\s*\[PAGE (\d+)\]\s*\n", text)
    chunks = []
    if not sections:
        return chunks

    for idx in range(1, len(sections), 2):
        page_num = int(sections[idx])
        body = normalize_text(sections[idx + 1])
        if not body:
            continue
        start = 0
        while start < len(body):
            end = min(start + chunk_chars, len(body))
            chunk_text = body[start:end].strip()
            if chunk_text:
                chunks.append(
                    {
                        "chunk_id": f"page-{page_num}-offset-{start}",
                        "page_start": page_num,
                        "page_end": page_num,
                        "chapter": f"Page {page_num}",
                        "heading": f"Page {page_num}",
                        "text": chunk_text,
                    }
                )
            if end >= len(body):
                break
            start = max(end - overlap_chars, start + 1)
    return chunks


def save_chunks(chunks: list[dict]) -> None:
    with CHUNKS_PATH.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def load_chunks() -> list[dict]:
    if not CHUNKS_PATH.exists():
        return []
    chunks = []
    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def ensure_reference_chunks(pdf_path: str | Path) -> list[dict]:
    if not REFERENCE_TEXT_PATH.exists():
        extract_pdf_to_text(pdf_path, REFERENCE_TEXT_PATH)
    if CHUNKS_PATH.exists():
        return load_chunks()
    text = load_reference_text()
    chunks = split_into_chunks(text)
    save_chunks(chunks)
    return chunks


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))


def keyword_retrieve(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    query_tokens = tokenize(query)
    ranked = []
    for chunk in chunks:
        chunk_tokens = tokenize(chunk.get("text", ""))
        if not chunk_tokens:
            continue
        overlap = len(query_tokens & chunk_tokens)
        density = overlap / math.sqrt(len(chunk_tokens))
        ranked.append((density, overlap, chunk))
    ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [item[2] for item in ranked[:top_k] if item[0] > 0]


def cosine_similarity_matrix(query_embedding: np.ndarray, doc_embeddings: np.ndarray) -> np.ndarray:
    query_norm = np.linalg.norm(query_embedding) + 1e-12
    doc_norm = np.linalg.norm(doc_embeddings, axis=1) + 1e-12
    return (doc_embeddings @ query_embedding) / (doc_norm * query_norm)


def save_embedding_cache(chunks: list[dict], embeddings: np.ndarray) -> None:
    payload = {
        "chunk_ids": np.array([chunk["chunk_id"] for chunk in chunks], dtype=object),
        "embeddings": embeddings,
    }
    np.savez_compressed(EMBEDDING_CACHE_PATH, **payload)


def load_embedding_cache() -> tuple[list[str], np.ndarray] | None:
    if not EMBEDDING_CACHE_PATH.exists():
        return None
    data = np.load(EMBEDDING_CACHE_PATH, allow_pickle=True)
    chunk_ids = data["chunk_ids"].tolist()
    embeddings = data["embeddings"]
    return chunk_ids, embeddings


def embedding_retrieve(
    query_embedding: np.ndarray,
    chunks: list[dict],
    embedding_cache: tuple[list[str], np.ndarray] | None,
    top_k: int = 5,
) -> list[dict]:
    if embedding_cache is None:
        return []
    chunk_ids, embeddings = embedding_cache
    index = {chunk["chunk_id"]: chunk for chunk in chunks}
    scores = cosine_similarity_matrix(query_embedding, embeddings)
    top_idx = np.argsort(scores)[::-1][:top_k]
    out = []
    for idx in top_idx:
        chunk_id = chunk_ids[idx]
        chunk = index.get(chunk_id)
        if chunk is not None:
            chunk = dict(chunk)
            chunk["score"] = float(scores[idx])
            out.append(chunk)
    return out


def build_query_from_summary(summary_payload: dict) -> str:
    segments = ", ".join(summary_payload.get("top_segments", []))
    drivers = ", ".join(summary_payload.get("top_drivers", []))
    hotel_type = summary_payload.get("hotel_type", "")
    role = summary_payload.get("role", "")
    month = summary_payload.get("month", "")
    return (
        f"{hotel_type} hotel revenue management {role} month {month} "
        f"cancellation risk drivers {drivers} segment {segments} "
        "recommended intervention booking cancellation pricing distribution overbooking deposit"
    )
