from __future__ import annotations

from dataclasses import dataclass

from src.chunking import Chunk
from src.config import AppConfig
from src.embeddings import BM25Index, CrossEncoderReranker, DenseEncoder
from src.io_utils import load_chunks
from src.length import LengthCounter, model_names
from src.store import uses_qdrant


@dataclass
class RetrieverState:
    chunks: list[Chunk]
    dense: DenseEncoder | None = None
    bm25: BM25Index | None = None
    reranker: CrossEncoderReranker | None = None
    length: LengthCounter | None = None


def make_length_counter(cfg: AppConfig) -> LengthCounter:
    return LengthCounter(
        model_names(cfg.dense_model, cfg.sparse_model, cfg.vectorization, cfg.cross_encoder_model)
    )


def load_retriever(cfg: AppConfig) -> RetrieverState:
    chunks_path = cfg.run_dir / "chunks.jsonl"
    if not chunks_path.exists():
        raise FileNotFoundError(f"Нет чанков {chunks_path}. Сначала scripts/build_index.py")
    length = make_length_counter(cfg)
    state = RetrieverState(chunks=load_chunks(chunks_path), length=length)

    if cfg.vectorization in {"sparse", "hybrid"}:
        bm25_path = cfg.run_dir / "bm25.pkl"
        if not bm25_path.exists():
            raise FileNotFoundError(f"Нет BM25 {bm25_path}")
        state.bm25 = BM25Index.load(bm25_path)

    if cfg.vectorization in {"dense", "hybrid"}:
        state.dense = DenseEncoder(cfg.dense_model, length)

    if cfg.cross_encoder_model.strip():
        state.reranker = CrossEncoderReranker(cfg.cross_encoder_model.strip(), length)
    return state


def needs_qdrant(cfg: AppConfig) -> bool:
    return uses_qdrant(cfg)
