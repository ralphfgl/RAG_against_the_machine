"""Indexing pipeline: walk corpus, chunk and persist."""

from tqdm import tqdm

from .walker import walk_corpus
from .chunker import chunk_file, IndexedChunk
from .persist import persist_index, INDEX_DIR

# NOTE: need for a CLI arg or config file, instead of hardcoded path
DEFAULT_CORPUS_ROOT = "data/raw/vllm-0.10.1"


def run_index(
    max_chunk_size: int = 2000,
    corpus_root: str = DEFAULT_CORPUS_ROOT,
) -> int:
    """
    Run the full indexing pipeline.
    Args:
        max_chunk_size: Maximum number of characters per chunk.
        corpus_root: Path to the corpus directory.
    Returns:
        Number of chunks indexed.
    """

    files = list(walk_corpus(corpus_root))
    print(f"Found {len(files)} files to index.")

    all_chunks: list[IndexedChunk] = []

    for corpus_file in tqdm(files, desc="Chunking", unit="file"):
        try:
            chunks = chunk_file(corpus_file, max_chunk_size)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"[WARN] Failed to chunk {corpus_file.file_path}: {e}")
            continue

    print(f"Produced {len(all_chunks)} chunks. Building BM25 index...")

    persist_index(all_chunks)

    print(
        f"Indexing complete! Indexed {len(all_chunks)} chunks in {INDEX_DIR}/"
    )
    return len(all_chunks)
