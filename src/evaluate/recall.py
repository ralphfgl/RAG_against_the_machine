"""Recall@k evaluation against a ground-truth dataset."""

import json
from dataclasses import dataclass

from src.models import (
    AnsweredQuestion,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)

MIN_IOU = 0.05


def _iou(a: MinimalSource, b: MinimalSource) -> float:
    pass


def _source_found(
    target: MinimalSource, candidates: list[MinimalSource]
) -> bool:
    pass

@dataclass
class RecallResult:


def _recall_at_k():
    pass


def evalutate(
    student_search_results_path: str,
    dataset_path: str,
    show_failures: int = 10,
) -> RecallResult:
    """Compute recall@1/3/5/10 of student results vs. ground truth.
    Args:
        student_search_results_path: output of `search_dataset`.
        dataset_path: ground-truth dataset (AnsweredQuestions).
        show_failures: how many failing questions to print at the end.
    Returns:
        RecallResult with the aggregated metrics.
    """
