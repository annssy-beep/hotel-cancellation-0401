import os
from typing import Any

import numpy as np
from openai import OpenAI

from hotel_dss.prompts import build_system_prompt, build_user_prompt
from hotel_dss.rag import (
    CHUNKS_PATH,
    REFERENCE_TEXT_PATH,
    build_query_from_summary,
    embedding_retrieve,
    ensure_reference_chunks,
    keyword_retrieve,
    load_chunks,
    load_embedding_cache,
    save_embedding_cache,
)


DEFAULT_CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
DEFAULT_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


def get_client() -> OpenAI | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def llm_status() -> dict[str, str | bool]:
    client = get_client()
    if client is None:
        return {"ready": False, "message": "LLM disabled: `OPENAI_API_KEY` is not set."}
    if not REFERENCE_TEXT_PATH.exists():
        return {
            "ready": False,
            "message": "LLM is available, but RAG is disabled because `rag/rm_reference.txt` is missing.",
        }
    return {"ready": True, "message": "LLM + RAG ready"}


def generate_role_narrative(summary_payload: dict) -> str:
    role = summary_payload["role"]
    hotel_type = summary_payload["hotel_type"]
    month = summary_payload["month"]
    pred_rate = summary_payload["pred_cancel_rate"]
    exposure = summary_payload["revenue_exposure"]
    opp = summary_payload["opportunity_score"]
    segments = ", ".join(summary_payload["top_segments"]) if summary_payload["top_segments"] else "segment evidence pending"
    return (
        f"For {month}, {hotel_type} shows a predicted cancellation rate of {pred_rate:.1%}, "
        f"revenue exposure of {exposure:,.0f}, and intervention opportunity of {opp:,.0f}. "
        f"The riskiest segments are {segments}."
    )


def get_text_embedding(client: OpenAI, text: str, model: str = DEFAULT_EMBEDDING_MODEL) -> np.ndarray:
    response = client.embeddings.create(model=model, input=text)
    return np.asarray(response.data[0].embedding, dtype=np.float32)


def ensure_embedding_cache(client: OpenAI, chunks: list[dict], model: str = DEFAULT_EMBEDDING_MODEL) -> tuple[list[str], np.ndarray]:
    cached = load_embedding_cache()
    if cached is not None:
        chunk_ids, embeddings = cached
        if len(chunk_ids) == len(chunks):
            return cached

    texts = [chunk["text"] for chunk in chunks]
    batch_size = 20
    max_chars_per_batch = 24000
    vectors: list[np.ndarray] = []
    batch: list[str] = []
    for text in texts:
        projected_chars = sum(len(item) for item in batch) + len(text)
        if batch and (len(batch) >= batch_size or projected_chars > max_chars_per_batch):
            response = client.embeddings.create(model=model, input=batch)
            batch_embeddings = np.asarray([row.embedding for row in response.data], dtype=np.float32)
            vectors.append(batch_embeddings)
            batch = []
        batch.append(text)
    if batch:
        response = client.embeddings.create(model=model, input=batch)
        batch_embeddings = np.asarray([row.embedding for row in response.data], dtype=np.float32)
        vectors.append(batch_embeddings)
    embeddings = np.vstack(vectors) if vectors else np.empty((0, 0), dtype=np.float32)
    save_embedding_cache(chunks, embeddings)
    return [chunk["chunk_id"] for chunk in chunks], embeddings


def retrieve_context(
    summary_payload: dict,
    pdf_path: str,
    use_embeddings: bool = False,
    top_k: int = 5,
) -> list[dict]:
    chunks = ensure_reference_chunks(pdf_path)
    query = build_query_from_summary(summary_payload)
    if not use_embeddings:
        return keyword_retrieve(query, chunks, top_k=top_k)

    client = get_client()
    if client is None:
        return keyword_retrieve(query, chunks, top_k=top_k)
    cache = ensure_embedding_cache(client, chunks)
    query_embedding = get_text_embedding(client, query)
    retrieved = embedding_retrieve(query_embedding, chunks, cache, top_k=top_k)
    return retrieved if retrieved else keyword_retrieve(query, chunks, top_k=top_k)


def generate_llm_decision_brief(
    summary_payload: dict,
    retrieved_chunks: list[dict],
    model: str = DEFAULT_CHAT_MODEL,
) -> str:
    client = get_client()
    if client is None:
        raise RuntimeError("OPENAI_API_KEY is not set")

    response = client.responses.create(
        model=model,
        max_output_tokens=220,
        input=[
            {"role": "system", "content": build_system_prompt(summary_payload["role"])},
            {"role": "user", "content": build_user_prompt(summary_payload, retrieved_chunks)},
        ],
    )
    return response.output_text


def build_brief_with_fallback(
    summary_payload: dict,
    pdf_path: str,
    use_embeddings: bool = False,
    top_k: int = 5,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "retrieved_chunks": [],
        "narrative_preview": generate_role_narrative(summary_payload),
        "llm_text": None,
        "mode": "rule_preview",
        "error": None,
    }

    try:
        retrieved_chunks = retrieve_context(
            summary_payload=summary_payload,
            pdf_path=pdf_path,
            use_embeddings=use_embeddings,
            top_k=top_k,
        )
        result["retrieved_chunks"] = retrieved_chunks
    except Exception as exc:
        result["error"] = f"retrieval_error: {type(exc).__name__}: {exc}"
        use_embeddings = False

    client = get_client()
    if client is None:
        return result

    try:
        result["llm_text"] = generate_llm_decision_brief(summary_payload, result["retrieved_chunks"])
        result["mode"] = "llm_keyword_rag" if not use_embeddings else "llm_embedding_rag"
    except Exception as exc:
        result["error"] = f"llm_error: {type(exc).__name__}: {exc}"
    return result
