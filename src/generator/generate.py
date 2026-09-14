"""Answer generation: retrieve, prompt, generate."""

import json
from pathlib import Path

from tqdm import tqdm

from src.models import (
    MinimalAnswer,
    MinimalSource,
    StudentSearchResultsAndAnswer,
)
from src.retriever.search import search_with_text
from src.generator.prompt import build_prompt
from src.generator.model import generate


MAX_CONTEXT_CHARS = 6000  # about 3 chunks


def _trim_context(sources: list[dict], budget: int) -> list[dict]:
    """Trim the retrieved sources so their combined text fits in the budget."""
    kept = []
    used = 0
    for src in sources:
        text = src["text"]
        if used + len(text) > budget:
            # Take a partial chunk
            remaining = budget - used
            if remaining < 200:  # too little to be useful
                break
            src = {**src, "text": text[:remaining]}
        kept.append(src)
        used += len(src["text"])
        if used >= budget:
            break
    return kept


def answer(query: str, k: int = 10) -> dict:
    """
    Answer a single query.

    Returns a dict with:
        question, retrieved_sources (list of MinimalSource), answer (str)
    """
    retrieved = search_with_text(query, k=k)
    trimmed = _trim_context(retrieved, MAX_CONTEXT_CHARS)

    if not trimmed:
        return {
            "question": query,
            "retrieved_sources": [],
            "answer": "I cant answer from the retrieved sources.",
        }

    prompt = build_prompt(query, trimmed)
    raw_answer = generate(prompt, max_new_tokens=256, temperature=0.0)

    return {
        "question": query,
        "retrieved_sources": [
            MinimalSource(
                file_path=src["file_path"],
                first_character_index=src["first_character_index"],
                last_character_index=src["last_character_index"],
            )
            for src in trimmed
        ],
        "answer": raw_answer,
    }


def answer_dataset(
    student_search_results_path: str,
    save_directory: str = "data/output/search_results_and_answer",
) -> str:
    """
    Generate answers for a whole dataset.
    Args:
        student_search_results_path: Path to the JSON
            produced by search_dataset.
        save_directory: Where to write the output JSON.

    Returns:
        Path to the output file.
    """
    with open(student_search_results_path, encoding="utf-8") as f:
        raw = json.load(f)

    # The search dataset output uses StudentSearchResults, but the
    # MinimalSearchResults shape is identical for our purposes.
    from src.models import StudentSearchResults

    search_results = StudentSearchResults.model_validate(raw)

    answers: list[MinimalAnswer] = []
    for entry in tqdm(
        search_results.search_results, desc="Answering", unit="q"
    ):
        # Re-search with text so we get the actual snippets.
        retrieved = search_with_text(entry.question, k=search_results.k)
        trimmed = _trim_context(retrieved, MAX_CONTEXT_CHARS)

        if not trimmed:
            answer_text = (
                "I cannot answer this question from the retrieved sources."
            )
        else:
            prompt = build_prompt(entry.question, trimmed)
            answer_text = generate(prompt, max_new_tokens=256, temperature=0.0)

        answers.append(
            MinimalAnswer(
                question_id=entry.question_id,
                question=entry.question,
                retrieved_sources=[
                    MinimalSource(
                        file_path=src["file_path"],
                        first_character_index=src["first_character_index"],
                        last_character_index=src["last_character_index"],
                    )
                    for src in trimmed
                ],
                answer=answer_text,
            )
        )

    dataset_stem = Path(student_search_results_path).stem
    save_dir = Path(save_directory)
    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / f"{dataset_stem}.json"

    result = StudentSearchResultsAndAnswer(
        search_results=answers, k=search_results.k
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result.model_dump_json(indent=2))

    print(f"Saved search_results_and_answer to {output_path}")
    return str(output_path)
