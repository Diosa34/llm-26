from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from rank_bm25 import BM25Okapi
from transformers import AutoModelForMaskedLM, AutoTokenizer

import re
import unicodedata
from functools import lru_cache

from nltk.stem.snowball import SnowballStemmer

from src.length import E5_PREFIX_DOC, E5_PREFIX_QUERY, LengthCounter


_RU_STOP = {
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то",
    "все", "она", "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за",
    "бы", "по", "только", "ее", "мне", "было", "вот", "от", "меня", "еще",
    "нет", "о", "из", "ему", "теперь", "когда", "даже", "ну", "вдруг", "ли",
    "если", "уже", "или", "ни", "быть", "был", "него", "до", "вас", "нибудь",
    "опять", "уж", "вам", "ведь", "там", "потом", "себя", "ничего", "ей",
    "может", "они", "тут", "где", "есть", "надо", "ней", "для", "мы", "тебя",
    "их", "чем", "была", "сам", "чтоб", "без", "будто", "чего", "раз", "тоже",
    "себе", "под", "будет", "ж", "тогда", "кто", "этот", "того", "потому",
    "этого", "какой", "совсем", "ним", "здесь", "этом", "один", "почти",
    "мой", "тем", "чтобы", "нее", "сейчас", "были", "куда", "зачем", "всех",
    "никогда", "можно", "при", "наконец", "два", "об", "другой", "хоть",
    "после", "над", "больше", "тот", "через", "эти", "нас", "про", "всего",
    "них", "какая", "много", "разве", "три", "эту", "моя", "впрочем",
    "хорошо", "свою", "этой", "перед", "иногда", "лучше", "чуть", "том",
    "нельзя", "такой", "им", "более", "всегда", "конечно", "всю", "между",
    "эта", "эти", "это", "этих", "этим", "этими", "ко", "кто-то", "что-то",
}
_TOKEN_RE = re.compile(r"[а-яё]+(?:[-'][а-яё]+)*", re.IGNORECASE)
_STEMMER = SnowballStemmer("russian")


def _needs_e5_prefix(model_name: str) -> bool:
    return "e5" in model_name.lower()


@lru_cache(maxsize=50_000)
def _stem(tok: str) -> str:
    return _STEMMER.stem(tok)


def _tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).lower()
    out: list[str] = []
    for m in _TOKEN_RE.finditer(text):
        tok = m.group(0)
        if len(tok) < 2 or tok in _RU_STOP:
            continue
        stem = _stem(tok)
        if stem and stem not in _RU_STOP:
            out.append(stem)
    return out


class DenseEncoder:
    def __init__(self, model_name: str, length: LengthCounter | None = None):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.length = length

    @property
    def dim(self) -> int:
        """Возвращает размерность вектора для модели."""
        return self.model.get_embedding_dimension()

    def _prepare(self, texts: list[str], is_query: bool) -> list[str]:
        prefix = ""
        if _needs_e5_prefix(self.model_name):
            prefix = E5_PREFIX_QUERY if is_query else E5_PREFIX_DOC
        prepared = [prefix + t for t in texts]
        return prepared

    def encode(self, texts: list[str], is_query: bool = False, batch_size: int = 16) -> np.ndarray:
        vectors = self.model.encode(
            self._prepare(texts, is_query),
            batch_size=batch_size,
            show_progress_bar=len(texts) > 8,
            normalize_embeddings=True,
        )
        return np.asarray(vectors, dtype=np.float32)


class BM25Index:
    def __init__(self, texts: list[str]):
        self.corpus_tokens = [_tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def query(self, text: str, top_k: int) -> list[tuple[int, float]]:
        """
        Возвращает список (doc_id, score) для документов, у которых скор > 0.
        """
        scores = self.bm25.get_scores(_tokenize(text))
        ranked = np.argsort(scores)[::-1][:top_k] # сортируем индексы в порядке убывания скоров
        return [(int(i), float(scores[i])) for i in ranked if scores[i] > 0]

    def save(self, path: Path) -> None:
        path.write_bytes(pickle.dumps(self))

    @staticmethod
    def load(path: Path) -> "BM25Index":
        return pickle.loads(path.read_bytes())


class CrossEncoderReranker:
    def __init__(self, model_name: str, length: LengthCounter):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.length = length
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, hits: list[dict], top_k: int) -> list[dict]:
        if not hits:
            return []
        pairs = [[query, hit.get("text", "")] for hit in hits]
        scores = self.model.predict(pairs) # вычисялет скор релевантности для каждой пары (query, hit["text"])
        ranked = sorted(zip(hits, scores), key=lambda x: float(x[1]), reverse=True)
        out = []
        for hit, score in ranked[:top_k]:
            item = dict(hit)
            item["score"] = float(score)
            out.append(item)
        return out


def save_dense(path: Path, vectors: np.ndarray) -> None:
    np.save(path, vectors)


def load_dense(path: Path) -> np.ndarray:
    return np.load(path)
