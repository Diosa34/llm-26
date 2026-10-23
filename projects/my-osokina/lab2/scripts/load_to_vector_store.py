from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.embeddings import load_dense
from src.io_utils import load_chunks, log_experiment
from src.store import (
    client,
    drop_collection,
    ensure_collection,
    upsert_chunks,
    uses_dense_vectors,
    uses_qdrant,
)


def main(argv: list[str] | None = None) -> None:
    cfg, args = load_config(argv)
    chunks_path = cfg.run_dir / "chunks.jsonl"
    if not chunks_path.exists():
        raise SystemExit(f"Сначала соберите индекс: python scripts/build_index.py\nНет {chunks_path}")

    if not uses_qdrant(cfg):
        print("vectorization=sparse+bm25: Qdrant не нужен, поиск идёт по bm25.pkl")
        log_experiment(cfg, {"stage": "load_store", "collection": None, "n": 0})
        return

    qdrant = client(cfg)
    where = cfg.qdrant_url or str(cfg.qdrant_path)
    print(f"Qdrant: {where} (HNSW m={cfg.hnsw_m}, ef_construct={cfg.hnsw_ef_construct})")
    try:
        if args.drop_and_reindex:
            drop_collection(qdrant, cfg.collection_name)
            print(f"коллекция {cfg.collection_name} удалена")

        chunks = load_chunks(chunks_path)
        dense = None
        dim = None
        if uses_dense_vectors(cfg):
            emb_path = cfg.run_dir / "embeddings.npy"
            if not emb_path.exists():
                raise SystemExit(f"Нет эмбеддингов {emb_path}. Запустите build_index.py")
            dense = load_dense(emb_path)
            if len(chunks) != len(dense):
                raise SystemExit("Число чанков и dense-векторов не совпадает — build_index.py --rebuild")
            dim = int(dense.shape[1])

        ensure_collection(qdrant, cfg, dim=dim, rebuild=args.rebuild or args.drop_and_reindex)
        n = upsert_chunks(qdrant, cfg, chunks, dense_vectors=dense)
        print(f"загружено {n} точек в {cfg.collection_name} (HNSW m={cfg.hnsw_m}, ef_c={cfg.hnsw_ef_construct})")
        log_experiment(cfg, {"stage": "load_store", "collection": cfg.collection_name, "n": n})
    finally:
        qdrant.close()


if __name__ == "__main__":
    main()
