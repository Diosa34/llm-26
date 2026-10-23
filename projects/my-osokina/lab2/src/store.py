from __future__ import annotations

import uuid
from typing import Iterable

from qdrant_client import QdrantClient
from qdrant_client.http import models

from src.chunking import Chunk
from src.config import AppConfig


def uses_qdrant(cfg: AppConfig) -> bool:
    return cfg.vectorization in {"dense", "hybrid"}


def uses_dense_vectors(cfg: AppConfig) -> bool:
    return cfg.vectorization in {"dense", "hybrid"}


def client(cfg: AppConfig) -> QdrantClient:
    if cfg.qdrant_url:
        return QdrantClient(url=cfg.qdrant_url, api_key=cfg.qdrant_api_key or None, timeout=60)
    cfg.qdrant_path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(cfg.qdrant_path))


def _point_id(collection: str, chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{collection}:{chunk_id}"))


def collection_exists(qdrant: QdrantClient, name: str) -> bool:
    return qdrant.collection_exists(collection_name=name)


def drop_collection(qdrant: QdrantClient, name: str) -> None:
    if collection_exists(qdrant, name):
        qdrant.delete_collection(name)


def ensure_collection(qdrant: QdrantClient, cfg: AppConfig, dim: int | None, rebuild: bool = False) -> None:
    name = cfg.collection_name
    if rebuild:
        drop_collection(qdrant, name)
    if collection_exists(qdrant, name):
        return

    vectors_config: dict | models.VectorParams = {}
    if uses_dense_vectors(cfg):
        if not dim:
            raise ValueError("Для dense-коллекции нужна размерность вектора")
        vectors_config = {
            "dense": models.VectorParams(
                size=dim,
                distance=models.Distance.COSINE,
                hnsw_config=models.HnswConfigDiff(m=cfg.hnsw_m, ef_construct=cfg.hnsw_ef_construct),
            )
        }
    if not vectors_config:
        raise ValueError("Нечего создавать в Qdrant: нет dense")

    qdrant.create_collection(
        collection_name=name,
        vectors_config=vectors_config or {},
    )


def upsert_chunks(
    qdrant: QdrantClient,
    cfg: AppConfig,
    chunks: list[Chunk],
    dense_vectors=None,
) -> int:
    """Вставить точки в коллекцию. Если точка с таким id уже есть — она перезаписывается, если нет — создаётся."""
    name = cfg.collection_name
    batch: list[models.PointStruct] = []
    dense_vectors = dense_vectors if dense_vectors is not None else [None] * len(chunks)

    for chunk, dense in zip(chunks, dense_vectors):
        payload = chunk.to_dict()
        payload["run_id"] = cfg.run_id
        vector: dict = {}
        if dense is not None:
            vector["dense"] = dense.tolist() if hasattr(dense, "tolist") else dense
        batch.append(models.PointStruct(id=_point_id(name, chunk.id), vector=vector, payload=payload))
        if len(batch) >= 128:
            qdrant.upsert(collection_name=name, points=batch)
            batch = []
    if batch:
        qdrant.upsert(collection_name=name, points=batch)
    return len(chunks)


def qdrant_search(
    qdrant: QdrantClient,
    cfg: AppConfig,
    query_vector=None,
    top_k: int = 5,
) -> list[dict]:
    params = models.SearchParams(hnsw_ef=cfg.hnsw_ef_search)
    query = query_vector.tolist() if hasattr(query_vector, "tolist") else query_vector
    hits = qdrant.query_points(
        collection_name=cfg.collection_name,
        query=query,
        using="dense",
        limit=top_k,
        search_params=params,
        with_payload=True,
    ).points
    results = []
    for hit in hits:
        item = dict(hit.payload)
        item["score"] = float(hit.score)
        results.append(item)
    return results


def iter_payloads(qdrant: QdrantClient, cfg: AppConfig) -> Iterable[dict]:
    offset = None
    while True:
        points, offset = qdrant.scroll(
            collection_name=cfg.collection_name,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            yield dict(point.payload)
        if offset is None:
            break
