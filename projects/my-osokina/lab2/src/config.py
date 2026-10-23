from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from huggingface_hub import login

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def apply_hf_token() -> None:
    token = os.getenv("HF_TOKEN")
    login(token=token, add_to_git_credential=False)


apply_hf_token()


@dataclass
class AppConfig:
    corpus_dir: Path
    markdown_dir: Path
    artifacts_dir: Path
    gold_path: Path
    splitter: str
    chunk_size: int
    overlap: int
    headers_mode: str
    vectorization: str
    dense_model: str
    sparse_model: str
    cross_encoder_model: str
    batch_size: int
    qdrant_path: Path
    qdrant_url: str
    qdrant_api_key: str
    hnsw_m: int
    hnsw_ef_construct: int
    hnsw_ef_search: int
    top_k: int
    rrf_k: int
    ollama_base_url: str
    ollama_model: str
    temperature: float
    llm_max_tokens: int

    @property
    def run_id(self) -> str:
        payload = {
            "splitter": self.splitter,
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "headers_mode": self.headers_mode,
            "vectorization": self.vectorization,
            "dense_model": self.dense_model,
            "sparse_model": self.sparse_model,
        }
        digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        return digest[:10]

    @property
    def run_dir(self) -> Path:
        return self.artifacts_dir / "runs" / self.run_id

    @property
    def collection_name(self) -> str:
        return f"rag_{self.run_id}"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key, value in data.items():
            if isinstance(value, Path):
                data[key] = str(value)
        data["run_id"] = self.run_id
        data["collection_name"] = self.collection_name
        if data.get("qdrant_api_key"):
            data["qdrant_api_key"] = "***"
        if data.get("qdrant_url"):
            data["qdrant_mode"] = "cloud"
        return data


def _deep_get(data: dict, *keys, default=None):
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _as_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


def load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--config", default=str(ROOT / "config_middle.yaml"))
    p.add_argument("--splitter", choices=["recursive", "markdown", "semantic", "hybrid"])
    p.add_argument("--chunk-size", type=int, default=None)
    p.add_argument("--overlap", type=int, default=None)
    p.add_argument("--headers-mode", choices=["none", "text", "metadata", "both"])
    p.add_argument("--vectorization", choices=["dense", "sparse", "hybrid"])
    p.add_argument("--dense-model")
    p.add_argument("--sparse-model")
    p.add_argument("--cross-encoder-model")
    p.add_argument("--top-k", type=int)
    p.add_argument("--hnsw-m", type=int)
    p.add_argument("--hnsw-ef-construct", type=int)
    p.add_argument("--hnsw-ef-search", type=int)
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--drop-and-reindex", action="store_true")
    p.add_argument("--k", type=int, help="k для метрик retrieval")
    p.add_argument("question", nargs="?", help="вопрос для RAG")
    return p.parse_args(argv)


def load_config(argv: list[str] | None = None) -> tuple[AppConfig, argparse.Namespace]:
    load_dotenv(ROOT / ".env")
    apply_hf_token()
    args = parse_args(argv)
    raw = load_yaml(Path(args.config))

    splitter = args.splitter or _deep_get(raw, "chunking", "splitter", default="hybrid")
    chunk_size = args.chunk_size or int(_deep_get(raw, "chunking", "chunk_size", default=400))
    overlap = args.overlap if args.overlap is not None else int(_deep_get(raw, "chunking", "overlap", default=80))
    if not 100 <= chunk_size <= 1000:
        raise ValueError("chunk_size должен быть в диапазоне 100–1000 токенов")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap должен быть >= 0 и меньше chunk_size")

    cfg = AppConfig(
        corpus_dir=_as_path(os.getenv("CORPUS_DIR") or _deep_get(raw, "paths", "corpus_dir", default="")),
        markdown_dir=_as_path(os.getenv("MARKDOWN_DIR") or _deep_get(raw, "paths", "markdown_dir", default="")),
        artifacts_dir=_as_path(os.getenv("ARTIFACTS_DIR") or _deep_get(raw, "paths", "artifacts_dir", default="./artifacts")),
        gold_path=_as_path(_deep_get(raw, "paths", "gold_path", default="./eval/gold.json")),
        splitter=splitter,
        chunk_size=chunk_size,
        overlap=overlap,
        headers_mode=args.headers_mode or _deep_get(raw, "chunking", "headers_mode", default="both"),
        vectorization=args.vectorization or _deep_get(raw, "embeddings", "vectorization", default="hybrid"),
        dense_model=args.dense_model or _deep_get(raw, "embeddings", "dense_model", default="intfloat/multilingual-e5-small"),
        sparse_model=args.sparse_model or _deep_get(raw, "embeddings", "sparse_model", default="bm25"),
        cross_encoder_model=args.cross_encoder_model
        if args.cross_encoder_model is not None
        else str(_deep_get(raw, "embeddings", "cross_encoder_model", default="") or ""),
        batch_size=int(_deep_get(raw, "embeddings", "batch_size", default=16)),
        qdrant_path=_as_path(_deep_get(raw, "store", "qdrant_path", default="./artifacts/qdrant")),
        qdrant_url=(os.getenv("QDRANT_URL") or "").strip(),
        qdrant_api_key=(os.getenv("QDRANT_API_KEY") or "").strip(),
        hnsw_m=args.hnsw_m or int(_deep_get(raw, "store", "hnsw_m", default=16)),
        hnsw_ef_construct=args.hnsw_ef_construct or int(_deep_get(raw, "store", "hnsw_ef_construct", default=64)),
        hnsw_ef_search=args.hnsw_ef_search or int(_deep_get(raw, "store", "hnsw_ef_search", default=32)),
        top_k=args.top_k or int(_deep_get(raw, "retrieval", "top_k", default=5)),
        rrf_k=int(_deep_get(raw, "retrieval", "rrf_k", default=60)),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL") or _deep_get(raw, "llm", "base_url", default="http://127.0.0.1:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL") or _deep_get(raw, "llm", "model", default="qwen2.5:3b"),
        temperature=float(_deep_get(raw, "llm", "temperature", default=0.1)),
        llm_max_tokens=int(os.getenv("OLLAMA_NUM_CTX") or _deep_get(raw, "llm", "llm_max_tokens", default=4096)),
    )
    return cfg, args
