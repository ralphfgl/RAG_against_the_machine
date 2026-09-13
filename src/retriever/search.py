"""Search the persisted BM25 index."""

from typing import NamedTuple

import bm25s

from src.indexer.persist import load_bm25, load_chunks
from src.models import MinimalSource

_bm25: bm25s.BM25 | None = None
_chunks: list[dict] | None = None


def _ensure_loaded() -> tuple[bm25s.BM25, list[dict]]:
    """Load the BM25 index and chunk metadata (once)."""
    global _bm25, _chunks
    if _bm25 is None:
        _bm25 = load_bm25()
    if _chunks is None:
        _chunks = load_chunks()
    return _bm25, _chunks


def search(query: str, k: int = 10) -> list[MinimalSource]:
    """Return th top-k most relevant sources for a query.
    Args:
            query: The user's question.
            k: Number of results to return.
    Returns:
            A list of MinimalSource objects, ranked by relevance.
    """

    if not query or not query.strip():
        return []
    if k <= 0:
        return []

    retriever, chunks = _ensure_loaded()
    # tokenize the query the same way as the corpus
    query_tokens = bm25s.tokenize(query, stopwords="en", stemmer=None)
    # retrieve returns a 2-tuple: result-> the ranked document indices (which chunk matched, best first); _scores -> the corresponding BM25 relevance scores
    # result is a 2D array (shape(num_queries, k)) cause bm25s can handle batch queries in parrallel
    results, _scores = retriever.retrieve(
        query_tokens, k=k, show_progress=False
    )
    # ranked list of chunk for 1 query
    top_indices = results[0]
    sources: list[MinimalSource] = []
    # idx is a chunk index from BM25. index of BM25 correspond to record N in chunks.jsonl. That positional correspondance is the whole contract between the two file.
    for idx in top_indices:
        # bm25s returns indices as NumPy integer types rather than python int
        chunk = chunks[int(idx)]
        sources.append(
            MinimalSource(
                file_path=chunk["file_path"],
                first_character_index=chunk["first_character_index"],
                last_character_index=chunk["last_character_index"],
            )
        )
    return sources
