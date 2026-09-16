"""Build and persist the BM25 index and chunk metadata."""

import json
import pickle
from pathlib import Path
from typing import Any

import bm25s

from .chunker import IndexedChunk
from .tokenize_utils import split_identifiers, _enrich_markdown

INDEX_DIR = Path("data/processed")
BM25_INDEX_FILE = INDEX_DIR / "bm25_index.pkl"
CHUNKS_FILE = INDEX_DIR / "chunks.jsonl"


def _save_chunks(chunks: list[IndexedChunk]) -> None:
    """Save chunks to a JSONL file
    Args:
    """

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        for chunk in chunks:
            record = {
                "file_path": chunk.file_path,
                "first_character_index": chunk.first_character_index,
                "last_character_index": chunk.last_character_index,
                "text": chunk.text,
            }
            f.write(json.dumps(record) + "\n")


def load_chunks() -> list[dict[str, Any]]:
    """Load the saved chunks from JSONL."""

    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {CHUNKS_FILE}. Run `index` first."
        )
    records = []
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def _build_bm25(chunks: list[IndexedChunk]) -> bm25s.BM25:
    """Build a BM25 index over the chunk texts."""

    # texts = [chunk.text for chunk in chunks]
    texts = []
    for chunk in chunks:
        path_tokens = chunk.file_path.replace("/", " ").replace("\\", " ")
        basename = Path(chunk.file_path).name
        header = f"{path_tokens} {basename}"
        body = (
            split_identifiers(chunk.text)
            if chunk.file_type == "code"
            else _enrich_markdown(chunk.text)
        )
        texts.append(f"{header} {body}")
        # if chunk.file_type == "code":
        #     texts.append(split_identifiers(chunk.text))
        # else:
        #     texts.append(_enrich_markdown(chunk.text))
    # tokenize: split on whitespace + punctuation,
    # keep identifier-like tokens (e.g. user_id)
    # stopword: filter out common low-value english word ("the", "and")
    # stemmer=None: words will not be chopped down to their base roots
    # NOTE: stemming is harmful for code. Test with or without stopwords
    corpus_tokens = bm25s.tokenize(texts, stopwords="en", stemmer=None)
    retriever = bm25s.BM25(k1=0.82, b=1.13)
    retriever.index(corpus_tokens)
    return retriever


def _save_bm25(retriever: bm25s.BM25) -> None:
    """Save the BM25 retriver to disk"""

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with open(BM25_INDEX_FILE, "wb") as f:
        pickle.dump(retriever, f)


def load_bm25() -> bm25s.BM25:
    """Load the saved BM25 retriever from disk."""

    if not BM25_INDEX_FILE.exists():
        raise FileNotFoundError(
            f"BM25 index not found: {BM25_INDEX_FILE}. Run `index` first."
        )
    with open(BM25_INDEX_FILE, "rb") as f:
        return pickle.load(f)


def persist_index(chunks: list[IndexedChunk]) -> None:
    """Persist both the chunk metadata and BM25 index."""

    _save_chunks(chunks)
    retriever = _build_bm25(chunks)
    _save_bm25(retriever)
