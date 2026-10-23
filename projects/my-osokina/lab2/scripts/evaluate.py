from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.io_utils import log_experiment, write_json
from src.metrics import evaluate_queries
from src.retrieve import retrieve
from src.runtime import load_retriever, needs_qdrant
from src.sources import serialize_hits
from src.store import client


def main(argv: list[str] | None = None) -> None:
    cfg, args = load_config(argv)
    k = args.k or cfg.top_k
    gold = json.loads(cfg.gold_path.read_text(encoding="utf-8"))
    state = load_retriever(cfg)
    qdrant = client(cfg) if needs_qdrant(cfg) else None
    try:
        pairs = []
        for item in gold:
            hits = retrieve(
                cfg,
                item["question"],
                state.chunks,
                bm25=state.bm25,
                encoder=state.dense,
                reranker=state.reranker,
                top_k=k,
                qdrant=qdrant,
            )
            pairs.append((item, hits))
    finally:
        if qdrant is not None:
            qdrant.close()

    metrics = evaluate_queries(pairs, k=k, all_chunks=state.chunks)
    summary = {key: metrics[key] for key in metrics if key != "per_query"}
    per_query = []
    for row in metrics["per_query"]:
        per_query.append({**{k: v for k, v in row.items() if k != "hits"}, "hits": serialize_hits(row["hits"], k)})

    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "run_id": cfg.run_id,
        "k": k,
        "metrics": summary,
        "per_query": per_query,
    }
    out_path = cfg.run_dir / "eval.json"
    write_json(out_path, payload)
    log_experiment(cfg, {"stage": "evaluate", "k": k, "metrics": summary, "eval_path": str(out_path)})
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(per_query, ensure_ascii=False, indent=2))
    print(f"сохранено: {out_path}")


if __name__ == "__main__":
    main()
