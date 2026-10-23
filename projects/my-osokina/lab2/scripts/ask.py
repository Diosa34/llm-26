from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.io_utils import append_jsonl
from src.metrics import evaluate_queries, match_gold
from src.rag import generate_answer
from src.retrieve import retrieve
from src.runtime import load_retriever, needs_qdrant
from src.sources import serialize_hits
from src.store import client


def answer_once(cfg, state, qdrant, question: str, gold: list[dict], k: int) -> str:
    hits: list[dict] = retrieve(
        cfg,
        question,
        state.chunks,
        bm25=state.bm25,
        encoder=state.dense,
        reranker=state.reranker,
        top_k=k,
        qdrant=qdrant,
    )
    serialized = serialize_hits(hits, k) # Отбрасываем path, date, section, token_count, run_id

    # Получаем ответ от llm
    answer = generate_answer(cfg, question, hits, k)

    # Сохраняем лог
    append_jsonl(
        cfg.run_dir / "ask.jsonl",
        {
            "ts": datetime.now(timezone.utc).isoformat(),
            "run_id": cfg.run_id,
            "question": question,
            "hits": serialized,
        },
    )

    return answer


def main(argv: list[str] | None = None) -> None:
    cfg, args = load_config(argv)
    k = args.k or cfg.top_k
    gold = []
    if cfg.gold_path.exists():
        gold = json.loads(cfg.gold_path.read_text(encoding="utf-8"))
    state = load_retriever(cfg)
    qdrant = client(cfg) if needs_qdrant(cfg) else None
    try:
        if args.question:
            print(answer_once(cfg, state, qdrant, args.question, gold, k))
            return
        print("RAG-агент (пустая строка — выход). Ollama:", cfg.ollama_model)
        while True:
            try:
                question = input("Вопрос: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not question:
                break
            print(answer_once(cfg, state, qdrant, question, gold, k))
            print()
    finally:
        if qdrant is not None:
            qdrant.close()


if __name__ == "__main__":
    main()
