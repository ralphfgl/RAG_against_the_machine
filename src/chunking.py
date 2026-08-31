from dataclasses import dataclass


# frozen=True make the instance immutable
@dataclass(frozen=True)
class Chunk:
    """
    text                    : text content of the chunk
    file_path               : where the original text came from
    first_character_index   : starting position in original text (inclusive)
    last_character_index    : ending position in original text (exclusive)
    """

    text: str
    file_path: str
    first_character_index: int
    last_character_index: int


def chunk_text(
    text: str,
    file_path: str,
    max_chunk_size: int = 2000,
) -> list[Chunk]:
    """Split text into chunks while preserving character offsets.

    The returned character indices refer to the original `text` string.
    `last_character_index` is exclusive, so:

        text[first_character_index:last_character_index] == chunk.text
    """
    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be greater than 0")

    chunks: list[Chunk] = []

    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + max_chunk_size, text_length)

        chunk = Chunk(
            text=text[start:end],
            file_path=file_path,
            first_character_index=start,
            last_character_index=end,
        )

        chunks.append(chunk)
        start = end

    return chunks
