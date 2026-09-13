"""Prompt template for grounded answer generation."""

from src.retriever.search import load_chunk_text

SYSTEM_INSTRUCTIONS = """You are a technical assistant answering questions about the vLLM codebase.

You will be given:
1. A set of retrieved source snippets from the vLLM repository.
2. A question.

You task: answer the question using ONLY the information in the snippets.

Rules:
- If the snippets do not contain the answer, reply exactly: "I cannot answer this question from the retrieved sources."
- Do not invent file paths, function names, or API details that are not in the snippets.
- Be concise. One short paragraph, or a short bullet list.
- If the snippets contain code, you may quote it verbatim to support your answer.
"""


def build_prompt(question: str, sources: list[dict]) -> str:
    """
    Build the full prompt for the model.
    Args:
            question: The user's question.
            sources: A list of dicts with keys "file_path" and "text"
    Returns:
            A single prompt ready to feed to the tokenizer.
    """

    snippets_parts = []
    for i, src in enumerate(sources, start=1):
        snippets_parts.append(
            f"[Source {i}] {src['file_path']}\n```\n{src['text']}\n```"
        )
    snippets = "\n\n".join(snippets_parts)
    return (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        f"=== Retrieved sources ===\n"
        f"{snippets}\n\n"
        f"=== Question ===\n"
        f"{question}\n\n"
        f"=== Answer ===\n"
    )
