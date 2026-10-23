from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.chunking import Chunk
from src.config import AppConfig
from src.sources import norm_source


def save_chunks(path: Path, chunks: list[Chunk]) -> None:
    """Преобразует чанки в словари и записывает в файл"""
    with path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")


def load_chunks(path: Path) -> list[Chunk]:
    """Построчно читает файл, формируя список чанков"""
    items = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            data = json.loads(line)
            data["source"] = norm_source(data.get("source"))
            items.append(Chunk(**data))
    return items


def dump_run_config(cfg: AppConfig) -> None:
    """
    Сохраняет конфигурацию эксперимента в файл config.yaml.
    """
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    payload = cfg.to_dict()
    payload["created_at"] = datetime.now(timezone.utc).isoformat()
    (cfg.run_dir / "config.yaml").write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def log_experiment(cfg: AppConfig, extra: dict) -> None:
    """Логирует метаданные и результаты эксперимента в файл со списком всех экспериментов."""
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    row = {"ts": datetime.now(timezone.utc).isoformat(), "config": cfg.to_dict(), **extra}
    with (cfg.artifacts_dir / "experiments.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
