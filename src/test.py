from src.chunking import chunk_text


def test_chunk_text_preserves_offsets() -> None:
    text = "abcdefghij"

    chunks = chunk_text(
        text=text,
        file_path="test.txt",
        max_chunk_size=4,
    )

    assert len(chunks) == 3

    assert chunks[0].text == "abcd"
    assert chunks[0].first_character_index == 0
    assert chunks[0].last_character_index == 4

    assert chunks[1].text == "efgh"
    assert chunks[1].first_character_index == 4
    assert chunks[1].last_character_index == 8

    assert chunks[2].text == "ij"
    assert chunks[2].first_character_index == 8
    assert chunks[2].last_character_index == 10

    for chunk in chunks:
        assert (
            text[chunk.first_character_index : chunk.last_character_index]
            == chunk.text
        )
