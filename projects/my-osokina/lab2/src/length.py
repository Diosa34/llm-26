from __future__ import annotations

from dataclasses import dataclass

from transformers import AutoTokenizer

E5_PREFIX_DOC = "passage: "
E5_PREFIX_QUERY = "query: "
_DEFAULT_SPECIAL_TOKENS = 2


def _safe_max_length(tokenizer, fallback: int = 512) -> int:
    """Возвращает максимальное количество токенов, которое может быть использовано для модели."""
    ml = getattr(tokenizer, "model_max_length", None) or fallback
    if ml > 100_000: # Если модель поддерживает более 100 000 токенов, то возвращаем fallback
        return fallback
    return int(ml)


def _count_special_tokens(tokenizer, fallback: int = _DEFAULT_SPECIAL_TOKENS) -> int:
    """Сколько токенов токенайзер добавляет сам (обычно 2: [CLS]/[SEP])."""
    try:
        # Кодируем непустой текст, чтобы токенайзер точно добавил спецтокены
        probe = "a"
        with_special = len(tokenizer.encode(probe, add_special_tokens=True))
        without_special = len(tokenizer.encode(probe, add_special_tokens=False))
        return max(0, with_special - without_special)
    except Exception:
        return fallback


def needs_e5_prefix(model_name: str) -> bool:
    return "e5" in model_name.lower()


def doc_prefix_for(model_name: str) -> str:
    return E5_PREFIX_DOC if needs_e5_prefix(model_name) else ""


@dataclass
class ModelLimit:
    name: str
    tokenizer: object
    max_length: int
    special_tokens: int


class LengthCounter:
    """Один счётчик длины для чанкинга и моделей (dense / cross-encoder).

    count(text) = max(число токенов по всем подключенным токенайзерам).
    Лимит = min(model_max_length).
    """

    def __init__(self, model_names: list[str]):

        self.models: list[ModelLimit] = []
        seen: set[str] = set()
        for name in model_names:
            if name in seen:
                continue
            seen.add(name)
            # Загружаем токенайзер для модели
            tok = AutoTokenizer.from_pretrained(name, use_fast=True)
            self.models.append(ModelLimit(name=name, tokenizer=tok, max_length=_safe_max_length(tok), special_tokens=_count_special_tokens(tok),))
        self.limit = min(m.max_length for m in self.models)
        self.special_tokens = max(m.special_tokens for m in self.models)

    def count(self, text: str) -> int:
        """Возвращает максимальное количество токенов в тексте для всех моделей."""
        if not text:
            return 0
        return max(len(m.tokenizer.encode(text, add_special_tokens=False)) for m in self.models)

    def __call__(self, text: str) -> int:
        return self.count(text)

    def prefix_tokens(self, prefix: str) -> int:
        """Возвращает количество токенов в префиксе."""
        return self.count(prefix) if prefix else 0

    def content_budget(self, *prefixes: str) -> int:
        """Бюджет контента: лимит минус ВСЕ префиксы минус спец-токены."""
        used = sum(self.prefix_tokens(p) for p in prefixes if p)
        return max(8, self.limit - used - self.special_tokens)

    def full_length(self, text: str, *prefixes: str) -> int:
        """Длина того, что реально получит модель: префиксы + контент + спец-токены."""
        full = "".join(p for p in prefixes if p) + text
        return self.count(full) + self.special_tokens

    def fits(self, text: str, *prefixes: str) -> bool:
        """Проверяет, укладывается ли текст в лимит."""
        return self.full_length(text, *prefixes) <= self.limit


def model_names(dense_model: str, sparse_model: str, vectorization: str, cross_encoder_model: str) -> list[str]:
    """Возвращает список моделей."""
    names: list[str] = []
    if vectorization in {"dense", "hybrid"}:
        names.append(dense_model)
    if cross_encoder_model.strip():
        names.append(cross_encoder_model.strip())
    if not names:
        names.append(dense_model)
    return names
