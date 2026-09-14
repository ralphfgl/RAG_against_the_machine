"""RAG against the machine — entry point."""

import fire

from src.indexer import run_index
from src.retriever.search import search, search_dataset
from src.generator.generate import answer, answer_dataset
from src.evaluate.recall import evaluate
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

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 10,
        save_directory: str = "data/output/search_results",
    ) -> None:
        """Search a whole dataset of questions and save results.
        Args:
        """

        search_dataset(
            dataset_path=dataset_path, k=k, save_directory=save_directory
        )

    def answer(self, query: str, k: int = 10) -> None:
        """Answer a single query using retrieved context."""

        result = answer(query, k=k)
        print(f"Q: {result['question']}\nA: {result['answer']}\nSources:")
        for src in result["retrieved_sources"]:
            print(
                f"  {src.file_path} "
                f"[{src.first_character_index}:{src.last_character_index}]"
            )

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str = "data/output/search_results_and_answer",
    ) -> None:
        """Generate answer for a whole dataset."""

        answer_dataset(
            student_search_results_path=student_search_results_path,
            save_directory=save_directory,
        )

    def evaluate(
        self, student_search_results_path: str, dataset_path: str
    ) -> None:
        """Compute recall@k of student results against ground truth.

        Args:
            student_search_results_path: Output of search_dataset
            dataset_path: ground-truth AnsweredQuestions dataset.
        """

        result = evaluate(
            student_search_results_path=student_search_results_path,
            dataset_path=dataset_path,
        )
        print("Evaluation Results")
        print("==================")
        print(result.summary())


def main() -> None:
    fire.Fire(CLI)


if __name__ == "__main__":
    main()
