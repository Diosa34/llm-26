from __future__ import annotations

from typing import Iterable

from src.sources import norm_source


def _gold_sources(gold: dict) -> set[str]:
    """Получаем множество релевантных источников."""
    return {norm_source(s) for s in gold.get("relevant_sources", []) if s}


def _hit_source(hit: dict) -> str:
    return norm_source(hit.get("source"))


def _is_relevant_source(hit: dict, gold: dict) -> bool:
    src = _hit_source(hit)
    return bool(src) and src in _gold_sources(gold)


def _is_relevant(hit: dict, gold: dict) -> bool:
    """Чанк релевантен: источник в relevant_sources и (если задано) must_contain в тексте."""
    if not _is_relevant_source(hit, gold):
        return False
    needles = [n.lower() for n in gold.get("must_contain", [])]
    if not needles:
        return True
    text = hit.get("text", "").lower()
    return any(n in text for n in needles)


def _as_dict(chunk) -> dict:
    return chunk.to_dict() if hasattr(chunk, "to_dict") else chunk


def precision_at_k_chunks(hits: list[dict], gold: dict, k: int) -> float:
    """Доля релевантных чанков среди топ-k: |релевантные ∩ топ-k| / k."""
    if k <= 0:
        return 0.0
    top = hits[:k]
    return sum(1 for h in top if _is_relevant(h, gold)) / k


def precision_at_k_docs(hits: list[dict], gold: dict, k: int) -> float:
    """|документы из relevant_sources ∩ топ-k| / |уникальные документы в топ-k|."""
    retrieved = {_hit_source(h) for h in hits[:k] if _hit_source(h)}
    if not retrieved:
        return 0.0
    return len(retrieved & _gold_sources(gold)) / len(retrieved)


def recall_at_k_docs(hits: list[dict], gold: dict, k: int) -> float:
    """Источник релевантен, если он в relevant_sources, даже без must_contain в чанке."""
    needed = _gold_sources(gold)
    if not needed:
        return 0.0
    found = {_hit_source(h) for h in hits[:k] if _is_relevant_source(h, gold)}
    return len(found & needed) / len(needed)


def recall_at_k_chunks(
    hits: list[dict],
    gold: dict,
    k: int,
    all_chunks: list | None = None,
) -> float:
    """Доля найденных релевантных чанков среди всех релевантных в корпусе.

    Считаем по всем all_chunks, отбирая релевантные через _is_relevant: 
    |найденные в топ-k| / |всего релевантных|.
    Если релевантных нет или all_chunks не передан — 0.0.
    """
    total = sum(1 for c in all_chunks if _is_relevant(_as_dict(c), gold))
    if total == 0:
        return 0.0
    found = sum(1 for h in hits[:k] if _is_relevant(h, gold))
    return found / total


def reciprocal_rank(hits: list[dict], gold: dict, k: int) -> float:
    """Обратный ранг первого релевантного хита в топ-k (1/rank), если релевантные нашлись, иначе 0.0."""
    for rank, hit in enumerate(hits[:k], start=1):
        if _is_relevant(hit, gold):
            return 1.0 / rank
    return 0.0


def mrr(rows: Iterable[tuple[dict, list[dict]]], k: int) -> float:
    """Mean Reciprocal Rank: среднее reciprocal_rank по всем запросам."""
    total, n = 0.0, 0
    for gold, hits in rows:
        total += reciprocal_rank(hits, gold, k)
        n += 1
    return total / max(n, 1)


def evaluate_queries(
    rows: Iterable[tuple[dict, list[dict]]],
    k: int,
    all_chunks: list | None = None,
) -> dict:
    p_c = r_c = p_d = r_d = rr = 0.0
    n = 0
    per_query: list[dict] = []

    for gold, hits in rows:
        pc = precision_at_k_chunks(hits, gold, k)
        rc = recall_at_k_chunks(hits, gold, k, all_chunks=all_chunks)
        pd = precision_at_k_docs(hits, gold, k)
        rd = recall_at_k_docs(hits, gold, k)
        m = reciprocal_rank(hits, gold, k)

        p_c += pc
        r_c += rc
        p_d += pd
        r_d += rd
        rr += m
        n += 1

        per_query.append(
            {
                "id": gold.get("id"),
                "question": gold.get("question"),
                "precision_chunks": pc,
                "recall_chunks": rc,
                "precision_docs": pd,
                "recall_docs": rd,
                "mrr": m,
                "hits": hits,
            }
        )

    n = max(n, 1)
    return {
        f"precision_chunks@{k}": p_c / n,
        f"recall_chunks@{k}": r_c / n,
        f"precision_docs@{k}": p_d / n,
        f"recall_docs@{k}": r_d / n,
        f"mrr@{k}": rr / n,
        "n_queries": n,
        "per_query": per_query,
    }


def match_gold(question: str, gold_items: list[dict]) -> dict | None:
    """Находит gold-элемент в списке запросов gold по точному совпадению question."""
    q = (question or "").strip().lower()
    for item in gold_items:
        if (item.get("question") or "").strip().lower() == q:
            return item
    return None
