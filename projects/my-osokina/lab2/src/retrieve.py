from __future__ import annotations

from collections import defaultdict

from src.chunking import Chunk
from src.config import AppConfig
from src.embeddings import BM25Index, CrossEncoderReranker, DenseEncoder
from src.store import client, qdrant_search, uses_qdrant


def rrf_fuse(*ranked_lists: list[str], k: int = 60) -> list[str]:
    """
        Каждому документу из каждого списка прибавляем 1 / (k + rank), где rank - позиция в ранжированном списке, начинается с 1. 
        Чем выше документ в списке — тем больше вклад. k (по дефолту 60) — сглаживающая константа: она ослабляет доминирование первых мест.
        Документы, которые встречаются в обоих списках, получают сумму двух вкладов и поднимаются выше.

        Возвращается отсортированный по убыванию fused-score список id.
    """
    scores: dict[str, float] = defaultdict(float)
    for ranking in ranked_lists:
        for rank, cid in enumerate(ranking, start=1):
            scores[cid] += 1.0 / (k + rank)
    return [cid for cid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def retrieve(
    cfg: AppConfig,
    question: str,
    chunks: list[Chunk],
    bm25: BM25Index | None = None,
    encoder: DenseEncoder | None = None,
    reranker: CrossEncoderReranker | None = None,
    top_k: int | None = None,
    qdrant=None,
) -> list[dict]:
    k = top_k or cfg.top_k
    fetch = k * 3
    by_id = {c.id: c.to_dict() for c in chunks}
    dense_ids: list[str] = []
    sparse_ids: list[str] = []
    scored: dict[str, dict] = {}

    own_client = False
    if uses_qdrant(cfg) and qdrant is None:
        qdrant = client(cfg)
        own_client = True
    try:
        if cfg.vectorization in {"dense", "hybrid"}:
            if encoder is None:
                raise RuntimeError("Dense-модель не загружена")
            qvec = encoder.encode([question], is_query=True, batch_size=1)[0]
            for hit in qdrant_search(qdrant, cfg, query_vector=qvec, top_k=fetch):
                dense_ids.append(hit["id"]) # сохраняем порядок результатов
                scored[hit["id"]] = hit  # складываем в общий scored по id

        if cfg.vectorization in {"sparse", "hybrid"}:
            if bm25 is None:
                raise RuntimeError("BM25-индекс не загружен")
            for idx, score in bm25.query(question, top_k=fetch):
                item = chunks[idx].to_dict() # достаём Chunk по индексу и приводим к словарю
                item["score"] = score # кладём BM25-score в словарь Chunk
                sparse_ids.append(item["id"]) # сохраняем порядок результатов BM25
                scored[item["id"]] = item # складываем полный словарь в общий scored по id
    finally:
        if own_client and qdrant is not None:
            qdrant.close()

    # итоговый порядок id
    if cfg.vectorization == "dense":
        ordered = dense_ids
    elif cfg.vectorization == "sparse":
        ordered = sparse_ids
    else:
        ordered = rrf_fuse(dense_ids, sparse_ids, k=cfg.rrf_k)

    # Берём top-(fetch, если есть реранкер, иначе k) кандидатов в нужном порядке
    # и достаём для каждого полный словарь (score + поля чанка)
    results = [dict(scored.get(cid) or by_id[cid]) for cid in ordered[: fetch if reranker else k]]
    if reranker:
        return reranker.rerank(question, results, top_k=k)
    return results[:k]
