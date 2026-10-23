from __future__ import annotations

import httpx

from src.config import AppConfig
from src.sources import norm_source

SYSTEM = """Ты ассистент по документации Arenadata Catalog.
Отвечай только по приведённому контексту. Если ответа нет в контексте — скажи об этом.
После ответа кратко перечисли использованные источники.
Пиши по-русски, кратко и по делу."""


def build_prompt(question: str, contexts: list[dict], n: int) -> str:
    head = "Контекст:\n"
    tail = f"\n\nВопрос: {question}\n\nОтвет с цитатами [номер]:"

    blocks = []
    for i, ctx in enumerate(contexts[:n], start=1):
        page = f", стр. {ctx['page']}" if ctx.get("page") else ""
        headers = " > ".join(ctx.get("headers") or [])
        loc = f"{norm_source(ctx.get('source'))}{page}"
        if headers:
            loc += f" | {headers}"
        snippet = ctx.get("text", "").strip()
        blocks.append(f"[{i}] {loc}\n{snippet}") # {source}{page} | {headers} \n {snippet}
    
    ctx_str = "\n\n".join(blocks) if blocks else "(контекст не найден)"
    return f"{head}{ctx_str}{tail}"


def format_citations(contexts: list[dict], n: int) -> str:
    lines = ["", "Цитаты:"]
    for i, ctx in enumerate(contexts[:n], start=1):
        page = f", стр. {ctx['page']}" if ctx.get("page") else ""
        snippet = " ".join(ctx.get("text", "").split())[:220]
        lines.append(f"[{i}] {norm_source(ctx.get('source'))}{page}: «{snippet}»")
    return "\n".join(lines)


def generate_answer(cfg: AppConfig, question: str, contexts: list[dict], k=None) -> str:
    """Формирует запрос к llm, отправляет, затем возвращает ответ."""
    n = k or cfg.top_k
    prompt = build_prompt(question, contexts, n)
    url = cfg.ollama_base_url.rstrip("/") + "/api/chat"
    payload = {
        "model": cfg.ollama_model,
        "stream": False,
        "options": {
            "temperature": cfg.temperature,
            "num_ctx": cfg.llm_max_tokens,
        },
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
    }
    try:
        with httpx.Client(timeout=180) as http:
            resp = http.post(url, json=payload)
            resp.raise_for_status()
            answer = resp.json().get("message", {}).get("content", "").strip()
    except httpx.HTTPError as exc:
        answer = f"Не удалось обратиться к Ollama ({cfg.ollama_base_url}): {exc}"
    return answer + format_citations(contexts, n)
