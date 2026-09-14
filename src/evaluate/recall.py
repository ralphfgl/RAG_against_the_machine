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
    """Intersection-over-Union between two character ranges.

    Returns:
        A float in [0, 1]. Assumes a.file_path == b.file_path; the caller must check that first.
    """

    a_start, a_end = a.first_character_index, a.last_character_index
    b_start, b_end = b.first_character_index, b.last_character_index
    inter = max(0, min(a_end, b_end) - max(a_start, b_start))
    if inter == 0:
        return 0.0
    union = max(a_end, b_end) - min(a_start, b_start)
    if union <= 0:
        return 0.0
    return inter / union


def _source_found(
    target: MinimalSource, candidates: list[MinimalSource]
) -> bool:
    for cand in candidates:
        if cand.file_path != target.file_path:
            continue
        if _iou(cand, target) >= MIN_IOU:
            return True
    return False


@dataclass
class RecallResult:
    k: int
    num_questions: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    per_question: list[dict]

    def summary(self) -> str:
        return (
            f"Questions evaluated: {self.num_questions}\n"
            f"Recall@1:   {self.recall_at_1:.3f} ({self.recall_at_1 * 100:.1f}%)\n"
            f"Recall@3:   {self.recall_at_3:.3f} ({self.recall_at_3 * 100:.1f}%)\n"
            f"Recall@5:   {self.recall_at_5:.3f} ({self.recall_at_5 * 100:.1f}%)\n"
            f"Recall@10:  {self.recall_at_10:.3f} ({self.recall_at_10 * 100:.1f}%)\n"
        )


def _recall_at_k(
    gt_sources: list[MinimalSource],
    retrieved: list[MinimalSource],
    k: int,
) -> float:
    """Per-questoin recall at rank k."""
    if not gt_sources:
        # sentinel value to exclude questions with no ground truth
        return -1.0
    top_k = retrieved[:k]
    found = sum(1 for gt in gt_sources if _source_found(gt, top_k))
    return found / len(gt_sources)


def evaluate(
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

    with open(student_search_results_path, encoding="utf-8") as f:
        student_raw = json.load(f)
    with open(dataset_path, encoding="utf-8") as f:
        gt_raw = json.load(f)
    student_results = StudentSearchResults.model_validate(student_raw)
    gt_dataset = RagDataset.model_validate(gt_raw)
    # index ground truth by question_id
    gt_by_id: dict[str, AnsweredQuestion] = {}
    for q in gt_dataset.rag_questions:
        if isinstance(q, AnsweredQuestion):
            gt_by_id[q.question_id] = q

    ks = [1, 3, 5, 10]
    recalls = {k: [] for k in ks}
    per_question: list[dict] = []
    for entry in student_results.search_results:
        gt = gt_by_id.get(entry.question_id)
        if gt is None:
            # Not in the ground-truth set; skip.
            continue
        row = {
            "question_id": entry.question_id,
            "question": entry.question,
            "num_gt_sources": len(gt.sources),
        }
        for k in ks:
            r = _recall_at_k(gt.sources, entry.retrieved_sources, k)
            if r >= 0:
                recalls[k].append(r)
                row[f"recall@{k}"] = r

        per_question.append(row)

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    result = RecallResult(
        k=student_results.k,
        num_questions=len(per_question),  # passed GT filter
        recall_at_1=mean(recalls[1]),
        recall_at_3=mean(recalls[3]),
        recall_at_5=mean(recalls[5]),
        recall_at_10=mean(recalls[10]),
        per_question=per_question,
    )

    if show_failures > 0:
        failures = [r for r in per_question if r.get("recall@5", 0.0) < 1.0]
        if failures:
            print(
                f"\n--- {len(failures)} failing questions (recall@5 < 1.0) ---"
            )
            for r in failures[:show_failures]:
                print(
                    f"[{r['recall@5']:.2f}] {r['question_id']}: {r['question']}"
                )

    return result
