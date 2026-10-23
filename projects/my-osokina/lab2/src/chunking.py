from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import yaml
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.length import LengthCounter
from src.sources import norm_source

PAGE_RE = re.compile(r"<!--\s*Страница\s+(\d+)\s*-->")
MD_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]
SEPARATORS = ["\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " ", ""]


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    path: str
    date: str
    section: str
    page: int | None
    headers: list[str]
    token_count: int

    def to_dict(self) -> dict:
        return asdict(self)


class _STEmbeddings(Embeddings):
    """Обёртка SentenceTransformer под интерфейс LangChain SemanticChunker."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()


def _parse_frontmatter(raw: str) -> tuple[dict, str]:
    """
    raw: str - текст документа с метаданными.
    return: tuple[dict, str] - словарь метаданных и тело документа без метаданных.
    """
    if not raw.startswith("---"):
        return {}, raw
    parts = raw.split("---", 2)
    if len(parts) < 3:
        return {}, raw
    return yaml.safe_load(parts[1]) or {}, parts[2]


def _page_blocks(body: str) -> list[tuple[int | None, str]]:
    """
    body: str - текст документа без метаданных.
    return: list[tuple[int | None, str]] - список кортежей (номер страницы, текст блока).

    Регулярным выражением ищет номер страницы в строке и запоминает соответствующий блок текста.
    """
    blocks: list[tuple[int | None, str]] = []
    page: int | None = None
    buf: list[str] = []
    for line in body.splitlines():
        m = PAGE_RE.search(line)
        if m:
            if buf:
                blocks.append((page, "\n".join(buf)))
            buf, page = [], int(m.group(1))
            continue
        buf.append(line)
    if buf:
        blocks.append((page, "\n".join(buf)))
    return blocks or [(None, body)]


def _get_recursive_splitter(length: LengthCounter, chunk_size: int, overlap: int) -> RecursiveCharacterTextSplitter:
    """
    chunk_size: int - размер чанка.
    overlap: int - перекрытие чанков.

    return: RecursiveCharacterTextSplitter - рекурсивный текстовый разделитель.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=SEPARATORS,
        is_separator_regex=False,
        length_fn=length,
    )


def _apply_headers(text: str, headers: list[str], mode: str) -> tuple[str, list[str]]:
    clean = [h for h in headers if h]
    if mode in {"none", "metadata"}:
        body = text
    else:
        prefix = " > ".join(clean)
        body = f"{prefix}\n\n{text}" if prefix else text
    meta_headers = clean if mode in {"metadata", "both"} else []
    return body.strip(), meta_headers


def _make_chunk(
    text: str,
    meta: dict,
    headers: list[str],
    page: int | None,
    mode: str,
    length: LengthCounter,
    runtime_prefix: str = "",
) -> Chunk | None:
    body, meta_headers = _apply_headers(text, headers, mode)
    if length(body) < 20:
        return None
    cid = hashlib.sha1(f"{meta.get('source')}|{page}|{body}".encode("utf-8")).hexdigest()[:16]
    return Chunk(
        id=cid,
        text=body,
        source=norm_source(str(meta.get("source", ""))),
        path=str(meta.get("path", "")),
        date=str(meta.get("date", "")),
        section=str(meta.get("section", "")),
        page=page,
        headers=meta_headers,
        token_count=length.full_length(body, runtime_prefix),
    )


def _headers_from_doc(doc: Document) -> list[str]:
    return [str(v).strip() for v in doc.metadata.values() if str(v).strip()]


def _split_oversized(docs: list[Document], splitter: RecursiveCharacterTextSplitter) -> list[Document]:
    out: list[Document] = []
    for doc in docs:
        pieces = splitter.split_text(doc.page_content)
        for piece in pieces:
            out.append(Document(page_content=piece, metadata=dict(doc.metadata)))
    return out


def _markdown_docs(text: str) -> list[Document]:
    md = MarkdownHeaderTextSplitter(headers_to_split_on=MD_HEADERS, strip_headers=True)
    docs = md.split_text(text)
    return docs or [Document(page_content=text, metadata={})]

# Хранит экземпляр модели
_SEMANTIC_EMB: tuple[str, _STEmbeddings] | None = None


def _semantic_docs(text: str, dense_model: str) -> list[Document]:
    global _SEMANTIC_EMB
    if _SEMANTIC_EMB is None or _SEMANTIC_EMB[0] != dense_model:
        _SEMANTIC_EMB = (dense_model, _STEmbeddings(dense_model))
    splitter = SemanticChunker(
        _SEMANTIC_EMB[1],
    )
    chunks = splitter.split_text(text)
    return [Document(page_content=c, metadata={}) for c in chunks] or [Document(page_content=text, metadata={})]


def _chunks_from_docs(docs, meta, page, headers_mode, length_counter, rec, chunk_size, runtime_prefix=""):
    chunks = []
    for doc in docs:
        headers = _headers_from_doc(doc)
        header_prefix = " > ".join(h for h in headers if h) if headers_mode in {"both", "prefix"} else ""
        if header_prefix:
            header_prefix += "\n\n"
        budget = min(chunk_size, length_counter.content_budget(runtime_prefix, header_prefix))
        splitter = _get_recursive_splitter(length_counter, budget, rec.chunk_overlap)
        for piece in _split_oversized([doc], splitter):
            item = _make_chunk(piece.page_content, meta, headers, page, headers_mode, length_counter, runtime_prefix)
            if item:
                chunks.append(item)
    return chunks


def chunk_markdown(
    raw: str,
    splitter: str,
    chunk_size: int,
    overlap: int,
    headers_mode: str,
    length_counter: LengthCounter,
    dense_model: str,
    runtime_prefix: str = "",
) -> list[Chunk]:
    meta, body = _parse_frontmatter(raw)
    window = overlap if splitter in {"recursive", "hybrid", "semantic"} else 0
    recursive_splitter = _get_recursive_splitter(length_counter, chunk_size, window)
    chunks: list[Chunk] = []

    for page, text in _page_blocks(body):
        text = text.strip()
        if not text:
            continue
        if splitter == "recursive":
            docs = [Document(page_content=text, metadata={})]
        elif splitter == "markdown":
            docs = _markdown_docs(text)
        elif splitter == "semantic":
            docs = _semantic_docs(text, dense_model)
        else:
            docs = _markdown_docs(text)
        chunks.extend(_chunks_from_docs(docs, meta, page, headers_mode, length_counter, recursive_splitter, chunk_size, runtime_prefix,))
    return chunks


def build_chunks(
    markdown_dir: Path,
    corpus_dir: Path,
    splitter: str,
    chunk_size: int,
    overlap: int,
    headers_mode: str,
    length_counter: LengthCounter,
    dense_model: str,
    runtime_prefix: str = "",
) -> list[Chunk]:
    out: list[Chunk] = []
    seen: set[str] = set()

    def add_all(items: list[Chunk]) -> None:
        """Добавляет чанки в список."""
        for chunk in items:
            if chunk.id in seen:
                continue
            seen.add(chunk.id)
            out.append(chunk)

    for md_path in sorted(markdown_dir.glob("*.md")):
        """Собираем чанки из markdown файлов."""
        add_all(
            chunk_markdown(
                md_path.read_text(encoding="utf-8"),
                splitter,
                chunk_size,
                overlap,
                headers_mode,
                length_counter,
                dense_model,
                runtime_prefix,
            )
        )
    return out
