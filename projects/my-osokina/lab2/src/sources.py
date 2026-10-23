from __future__ import annotations


def norm_source(value: str | None) -> str:
    """Единый вид имени источника: trim + lower."""
    return (value or "").strip().lower()


def serialize_hits(hits: list[dict], n: int | None = None) -> list[dict]:
    selected = hits if n is None else hits[:n]
    rows = []
    for i, ctx in enumerate(selected, start=1):
        page = ctx.get("page")
        snippet = " ".join((ctx.get("text") or "").split())[:220] # обрезаем чисто для лога
        rows.append(
            {
                "rank": i,
                "id": ctx.get("id"),
                "source": norm_source(ctx.get("source")),
                "page": page,
                "headers": ctx.get("headers") or [],
                "score": ctx.get("score"),
                "snippet": snippet,
            }
        )
    return rows
