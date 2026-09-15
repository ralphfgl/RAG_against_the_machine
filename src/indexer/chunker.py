"""Chunk code and markdown files using Chonkie."""

from typing import NamedTuple

from chonkie import CodeChunker, RecursiveChunker, Chunk

from .walker import CorpusFile


class IndexedChunk(NamedTuple):
    """A chunk ready to be indexed."""

    file_path: str
    first_character_index: int
    last_character_index: int
    text: str


_code_chunker: CodeChunker | None = None
_markdown_chunker: RecursiveChunker | None = None


def _get_code_chunker(max_chunk_size: int) -> CodeChunker:
    global _code_chunker
    if _code_chunker is None:
        _code_chunker = CodeChunker(
            language="python", chunk_size=max_chunk_size, tokenizer="character"
        )
    return _code_chunker


# tokenizer="character" -> count chunk size in character and not models token
def _get_markdown_chunker(max_chunk_size: int) -> RecursiveChunker:
    global _markdown_chunker
    if _markdown_chunker is None:
        _markdown_chunker = RecursiveChunker(
            chunk_size=max_chunk_size, tokenizer="character"
        )
    return _markdown_chunker


def _find_offset(
    content: str, chunk_text: str, search_start: int
) -> tuple[int, int]:
    """Find the offsets of chunk_text within content, starting at search_start.
    Args:
        content: the entire text file
        chunk_text: the text of one chunk returned by chonkie
        search_start: search from this character position
    Returns:
        (start, end). If not found, returns (-1, -1).
    """

    idx = content.find(chunk_text, search_start)
    if idx == -1:
        # fallback: search from the begining (ws normalization from chunker)
        idx = content.find(chunk_text)
        if idx == -1:
            return -1, -1
    return idx, idx + len(chunk_text)


def chunk_file(
    corpus_file: CorpusFile, max_chunk_size: int
) -> list[IndexedChunk]:
    """Split a single corpus file into indexed chunks.
    Ensure every chunk is at most max_chunk_size.
    Args:
    Returns:
    """

    if corpus_file.file_type == "code":
        chunker = _get_code_chunker(max_chunk_size)
    else:
        chunker = _get_markdown_chunker(max_chunk_size)
    try:
        chunks: list[Chunk] = chunker.chunk(corpus_file.content)
    except Exception:
        # if chonkie fail fallback to naive fixed sized chunking
        return _fallback_chunk(corpus_file, max_chunk_size)
    indexed: list[IndexedChunk] = []
    search_start = 0
    for chunk in chunks:
        text = chunk.text
        if not text.strip():
            continue
        if len(text) > max_chunk_size:
            text = text[:max_chunk_size]
        start, end = _find_offset(corpus_file.content, text, search_start)
        if start == -1:
            # skip chunk we can't locate
            continue
        search_start = start  # to avoid passing overlap
        indexed.append(
            IndexedChunk(
                file_path=corpus_file.file_path,
                first_character_index=start,
                last_character_index=end,
                text=text,
            )
        )
    if not indexed:
        # nothing valid came out of chonkie
        return _fallback_chunk(corpus_file, max_chunk_size)
    return indexed


def _fallback_chunk(
    corpus_file: CorpusFile, max_chunk_size: int
) -> list[IndexedChunk]:
    """Naive fixed-size chunking as a fallback.
    Args:
    Returns:
    """

    content = corpus_file.content
    indexed: list[IndexedChunk] = []
    for start in range(0, len(content), max_chunk_size):
        end = min(start + max_chunk_size, len(content))
        text = content[start:end]
        if not text.strip():
            continue
        indexed.append(
            IndexedChunk(
                file_path=corpus_file.file_path,
                first_character_index=start,
                last_character_index=end,
                text=text,
            )
        )
    return indexed
