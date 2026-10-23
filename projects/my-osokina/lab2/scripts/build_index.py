from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.chunking import build_chunks
from src.config import load_config
from src.embeddings import BM25Index, DenseEncoder, save_dense
from src.io_utils import dump_run_config, load_chunks, log_experiment, save_chunks
from src.length import doc_prefix_for
from src.runtime import make_length_counter


def main(argv: list[str] | None = None) -> None:
    # Логируем конфигурацию эксперимента
    cfg, args = load_config(argv)
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    dump_run_config(cfg)

    # Определяем реальный допустимый размер чанка с учетом лимита моделей
    length_counter = make_length_counter(cfg)
    runtime_prefix = doc_prefix_for(cfg.dense_model) if cfg.vectorization in {"dense", "hybrid"} else ""
    budget = length_counter.content_budget(runtime_prefix) # относительно лимитов моделей
    chunk_size = min(cfg.chunk_size, budget)
    overlap = cfg.overlap if cfg.overlap < chunk_size else chunk_size // 2 
    if chunk_size != cfg.chunk_size:
        print(
            f"[{cfg.run_id}] chunk_size {cfg.chunk_size} → {chunk_size} "
            f"(лимит моделей {length_counter.limit}, prefix={runtime_prefix!r})"
        )

    chunks_path = cfg.run_dir / "chunks.jsonl"
    emb_path = cfg.run_dir / "embeddings.npy"
    bm25_path = cfg.run_dir / "bm25.pkl"

    if args.rebuild or not chunks_path.exists():
        print(
            f"[{cfg.run_id}] собираю чанки: splitter={cfg.splitter} size={chunk_size} "
            f"overlap={overlap} counter={ [m.name for m in length_counter.models] }"
        )
        chunks = build_chunks(
            cfg.markdown_dir,
            cfg.corpus_dir,
            cfg.splitter,
            chunk_size,
            overlap,
            cfg.headers_mode,
            length_counter,
            cfg.dense_model,
            runtime_prefix
        )
        save_chunks(chunks_path, chunks)
    else:
        chunks = load_chunks(chunks_path)
        print(f"[{cfg.run_id}] повторно использую {len(chunks)} чанков")

    # Получаем тексты чанков
    texts = [c.text for c in chunks]
    print(f"[{cfg.run_id}] документов-чанков: {len(chunks)}")

    # Кодируем тексты в векторы для dense модели
    if cfg.vectorization in {"dense", "hybrid"} and (args.rebuild or not emb_path.exists()):
        print(f"[{cfg.run_id}] dense: {cfg.dense_model}")
        encoder = DenseEncoder(cfg.dense_model, length_counter)
        vectors = encoder.encode(texts, is_query=False, batch_size=cfg.batch_size)
        save_dense(emb_path, vectors)
        print(f"[{cfg.run_id}] dense dim={vectors.shape[1]}")

    # Кодируем тексты в векторы для sparse модели
    if cfg.vectorization in {"sparse", "hybrid"}:
        print(f"[{cfg.run_id}] sparse: {cfg.sparse_model}")
        if args.rebuild or not bm25_path.exists():
            BM25Index(texts).save(bm25_path)

    log_experiment(
        cfg,
        {
            "stage": "build_index",
            "n_chunks": len(chunks),
            "effective_chunk_size": chunk_size,
            "length_limit": length_counter.limit,
        },
    )
    print(f"готово: {cfg.run_dir}")


if __name__ == "__main__":
    main()
