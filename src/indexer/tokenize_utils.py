import re

_LOWER_TO_UPPER = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_ACRONYM_TO_WORD = re.compile(r"(?<=[A-Z])(?=[A-Z][a-z])")


def split_identifiers(text: str) -> str:
    """Expand snake_case and camelCase/PascalCase identifiers into words."""

    expanded = text.replace("_", " ")
    expanded = _LOWER_TO_UPPER.sub(" ", expanded)
    expanded = _ACRONYM_TO_WORD.sub(" ", expanded)
    if expanded == text:
        return text
    return f"{text} {expanded}"


def _enrich_markdown(text: str) -> str:
    headings = re.findall(r"^#{1,6}\s+(.+)$", text, re.M)
    heading_text = " ".join(headings)
    return f"{heading_text} {text}" if heading_text else text
