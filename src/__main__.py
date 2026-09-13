"""RAG against the machine — entry point."""

import fire

from indexer import run_index
from src.retriever.search import search
from src.models import (
    MinimalSource,
    StudentSearchResults,
    MinimalSearchResults,
)


class CLI:
    """RAG against the machine — command line interface."""

    def index(self, max_chunk_size: int = 2000) -> None:
        """Index the corpus into a searchable BM25 index.

        Args:
            max_chunk_size: Maximum characters per chunk.
        """
        run_index(max_chunk_size=max_chunk_size)

    def search(self, query: str, k: int = 10) -> None:
        """Return the top-k sources for a single query.
        Args:
            query: the question to search for.
            k: number of sources to return.
        """

        sources = search(query, k=k)
        for src in sources:
            print(
                f"{src.file_path}"
                f"[{src.first_character_index}:{src.last_character_index}]"
            )


def main() -> None:
    fire.Fire(CLI)


if __name__ == "__main__":
    main()
