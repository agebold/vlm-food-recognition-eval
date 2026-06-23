# PMC13092701 ingredient-level evaluation metric — Equations 1-6.
#
# Sim(a, b) = max(StrMatch(a, b), SemMatch(a, b))
#
# StrMatch uses normalized LCS length (Eq 5).
# SemMatch uses a synonym lookup (Eq 6).
#
# Precision and Recall are soft: every predicted/ground-truth ingredient
# gets a similarity score against its best match in the other list (Eq 1-2).
# This means partial name matches contribute fractionally rather than
# being binary hit/miss.

from __future__ import annotations
from synonyms import get_variants


def _lcs_length(a: str, b: str) -> int:
    m, n = len(a), len(b)
    # space-optimized DP — O(min(m,n)) memory
    if m < n:
        a, b, m, n = b, a, n, m
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[n]


def str_match(a: str, b: str) -> float:
    """Eq 5: 2 * |LCS(a,b)| / (|a| + |b|). Returns 0 if both empty."""
    total = len(a) + len(b)
    if total == 0:
        return 1.0
    return 2 * _lcs_length(a, b) / total


def sem_match(a: str, b: str) -> float:
    """Eq 6: 1.0 if b is a known synonym of a (or vice versa), else 0.0."""
    a, b = a.lower().strip(), b.lower().strip()
    if a == b:
        return 1.0
    variants_a = get_variants(a)
    variants_b = get_variants(b)
    if b in variants_a or a in variants_b:
        return 1.0
    return 0.0


def sim(a: str, b: str) -> float:
    """Eq 4: max(StrMatch, SemMatch)."""
    a_n, b_n = a.lower().strip(), b.lower().strip()
    if a_n == b_n:
        return 1.0
    return max(str_match(a_n, b_n), sem_match(a_n, b_n))


def precision(predicted: list[str], ground_truth: list[str]) -> float:
    """Eq 1: average best-match similarity of each prediction against GT."""
    if not predicted:
        return 0.0
    if not ground_truth:
        return 0.0
    return sum(max(sim(p, t) for t in ground_truth) for p in predicted) / len(predicted)


def recall(predicted: list[str], ground_truth: list[str]) -> float:
    """Eq 2: average best-match similarity of each GT item against predictions."""
    if not ground_truth:
        return 1.0  # nothing to recall
    if not predicted:
        return 0.0
    return sum(max(sim(t, p) for p in predicted) for t in ground_truth) / len(ground_truth)


def f1(p: float, r: float) -> float:
    """Eq 3."""
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def score(predicted: list[str], ground_truth: list[str]) -> dict[str, float]:
    """Compute precision, recall, and F1 for a single dish. Returns dict."""
    p = precision(predicted, ground_truth)
    r = recall(predicted, ground_truth)
    return {"precision": p, "recall": r, "f1": f1(p, r)}


def aggregate(scores: list[dict[str, float]]) -> dict[str, float]:
    """Macro-average P/R/F1 across a list of per-dish score dicts."""
    if not scores:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n": 0}
    n = len(scores)
    return {
        "precision": sum(s["precision"] for s in scores) / n,
        "recall": sum(s["recall"] for s in scores) / n,
        "f1": sum(s["f1"] for s in scores) / n,
        "n": n,
    }
