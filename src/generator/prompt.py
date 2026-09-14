"""Prompt template for grounded answer generation."""

SYSTEM_INSTRUCTIONS = (
    "You are a technical assistant answering questions about the vLLM "
    "codebase.\n\n"
    "You will be given:\n"
    "1. A set of retrieved source snippets from the vLLM repository.\n"
    "2. A question.\n\n"
    "You task: answer the question using ONLY the information in the "
    "snippets.\n\n"
    "Rules:\n"
    '- If the snippets do not contain the answer, reply exactly: "I cannot '
    'answer this question from the retrieved sources."\n'
    "- Do not invent file paths, function names, or API details that are "
    "not in the snippets.\n"
    "- Be concise. One short paragraph, or a short bullet list.\n"
    "- If the snippets contain code, you may quote it verbatim to support "
    "your answer."
)


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
