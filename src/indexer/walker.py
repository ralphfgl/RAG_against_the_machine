"""Walk the corpus directory an yield files with their content."""

from pathlib import Path
from typing import Iterator, NamedTuple


class CorpusFile(NamedTuple):
    """A file from the corpus."""

    file_path: str
    absolute_path: Path
    file_type: str
    content: str


CODE_EXTENSIONS = {".py"}
MARKDOWN_EXTENSIONS = {".md", ".mdx", ".txt"}


def _categorize(path: Path) -> str | None:
    """Return 'code', 'markdown' or None if the file should be skipped."""

    suffix = path.suffix.lower()
    if suffix in CODE_EXTENSIONS:
        return "code"
    if suffix in MARKDOWN_EXTENSIONS:
        return "markdown"
    return None


def walk_corpus(corpus_root: str) -> Iterator[CorpusFile]:
    """Walk the corpus root and yield every code/markdown file.
    Args:
        corpus_root: Path to the corpus directory
    Yields:
        CorpusFile objects with file path, content and category.
    """

    root = Path(corpus_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Corpus root not found: {corpus_root}")
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        category = _categorize(path)
        if category is None:
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            # skip unreadable files silently
            continue
        if not content.strip():
            continue
        yield CorpusFile(
            file_path=str(path),
            absolute_path=path,
            file_type=category,
            content=content,
        )
