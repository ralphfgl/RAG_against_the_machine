"""Search the persisted BM25 index."""

import bm25s
import json
from tqdm import tqdm
from pathlib import Path

from src.indexer.persist import load_bm25, load_chunks
from src.models import (
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)

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
    # retrieve returns a 2-tuple:
    # result-> the ranked document indices (which chunk matched, best first);
    # _scores -> the corresponding BM25 relevance scores
    # result is a 2D array (shape(num_queries, k))
    # because bm25s can handle batch queries in parrallel
    results, _scores = retriever.retrieve(
        query_tokens, k=k, show_progress=False
    )
    # ranked list of chunk for 1 query
    top_indices = results[0]
    sources: list[MinimalSource] = []
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


def search_dataset(
    dataset_path: str,
    k: int = 10,
    save_directory: str = "data/output/search_results",
) -> str:
    """Run search over a whole dataset of questions.
    Args:
        dataset_path: Path to the input dataset JSON.
        k: number of sources per question.
        save_directory: Directory where the output JSON will be written.
    Returns:
        Path to the output file.
    """

    with open(dataset_path, encoding="utf-8") as f:
        raw = json.load(f)
    rag_dataset = RagDataset.model_validate(raw)
    search_results: list[MinimalSearchResults] = []
    for question in tqdm(
        rag_dataset.rag_questions, desc="Searching", unit="q"
    ):
        qid = question.question_id
        qtext = question.question
        sources = search(qtext, k=k)
        search_results.append(
            MinimalSearchResults(
                question_id=qid,
                question=qtext,
                retrieved_sources=sources,
            )
        )
    dataset_stem = Path(dataset_path).stem
    save_dir = Path(save_directory)
    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / f"{dataset_stem}.json"
    student_results = StudentSearchResults(search_results=search_results, k=k)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(student_results.model_dump_json(indent=2))
    print(f"Saved student_search_results to {output_path}")
    return str(output_path)


def load_chunk_text(source: MinimalSource) -> str:
    """Load the text of a source by reading the relevant slice of the file"""

    try:
        with open(source.file_path, encoding="utf-8", errors="replace") as f:
            content = f.read()
    except OSError:
        return ""
    start = max(0, source.first_character_index)
    end = min(len(content), source.last_character_index)
    return content[start:end]


def search_with_text(query: str, k: int = 10) -> list[dict]:
    """Like search(), but returns dicts that also include th chunk text.
    Returns a list of {"file_path", "first_character_index",
        "last_character_index", "text"}
    """

    sources = search(query, k=k)
    enriched = []
    for src in sources:
        enriched.append(
            {
                "file_path": src.file_path,
                "first_character_index": src.first_character_index,
                "last_character_index": src.last_character_index,
                "text": load_chunk_text(src),
            }
        )
    return enriched
