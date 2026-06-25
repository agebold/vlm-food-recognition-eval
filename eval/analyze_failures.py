"""
Failure analysis for VLM food recognition evaluation results.

Categories:
  MISIDENTIFICATION   — model predicts a completely wrong food (low sim to any GT item)
  HALLUCINATION       — model predicts ingredients not present in GT
  COVERAGE_FAILURE    — model under-predicts (misses many GT ingredients, low recall)
  SEMANTIC_NEAR_MISS  — model uses a near-synonym that LCS partially scores (0.4–0.8 sim)
  GOOD                — F1 >= 0.75

Usage:
  python3 analyze_failures.py results/sanity_check.json
  python3 analyze_failures.py results/gemma4_overhead_full.json --top 20
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from metric import sim

# Thresholds
GOOD_F1            = 0.75
MISID_SIM          = 0.25   # predicted item's best match to any GT item below this → misidentification
HALLUCINATION_SIM  = 0.30   # same threshold — item not matching anything in GT
NEAR_MISS_LOW      = 0.40
NEAR_MISS_HIGH     = 0.80
COVERAGE_RECALL    = 0.40   # recall below this when model predicts fewer than half the GT items


def classify_sample(sample: dict) -> dict:
    predicted = sample["predicted"]
    ground_truth = sample["ground_truth"]
    scores = sample["scores"]
    p, r, f1 = scores["precision"], scores["recall"], scores["f1"]

    issues: list[dict] = []

    # --- Misidentifications: predicted items that match nothing in GT ---
    for pred in predicted:
        best_sim = max((sim(pred, gt) for gt in ground_truth), default=0.0)
        if best_sim < MISID_SIM and ground_truth:
            issues.append({
                "type": "MISIDENTIFICATION",
                "predicted": pred,
                "best_gt_match": min(ground_truth, key=lambda gt: -sim(pred, gt)) if ground_truth else None,
                "best_sim": round(best_sim, 3),
            })

    # --- Missed GT items: GT ingredients the model didn't predict ---
    for gt in ground_truth:
        best_sim = max((sim(gt, pred) for pred in predicted), default=0.0)
        if best_sim < NEAR_MISS_LOW:
            issues.append({
                "type": "MISSED_INGREDIENT",
                "missed": gt,
                "best_predicted_match": min(predicted, key=lambda p: -sim(gt, p)) if predicted else None,
                "best_sim": round(best_sim, 3),
            })
        elif NEAR_MISS_LOW <= best_sim < NEAR_MISS_HIGH:
            best_pred = min(predicted, key=lambda p: -sim(gt, p)) if predicted else None
            issues.append({
                "type": "SEMANTIC_NEAR_MISS",
                "ground_truth": gt,
                "predicted_as": best_pred,
                "sim": round(best_sim, 3),
            })

    # --- Coverage failure: low recall with sparse prediction ---
    if r < COVERAGE_RECALL and len(predicted) < len(ground_truth) / 2:
        issues.append({
            "type": "COVERAGE_FAILURE",
            "gt_count": len(ground_truth),
            "predicted_count": len(predicted),
            "recall": round(r, 3),
        })

    # Determine primary failure category
    types = [i["type"] for i in issues]
    if f1 >= GOOD_F1:
        primary = "GOOD"
    elif "COVERAGE_FAILURE" in types:
        primary = "COVERAGE_FAILURE"
    elif types.count("MISIDENTIFICATION") >= len(predicted) / 2 and predicted:
        primary = "MISIDENTIFICATION"
    elif types.count("MISSED_INGREDIENT") > types.count("MISIDENTIFICATION"):
        primary = "MISSED_INGREDIENT"
    else:
        primary = "HALLUCINATION"

    return {
        "dish_id": sample["dish_id"],
        "camera": sample.get("camera", "unknown"),
        "primary_failure": primary,
        "f1": round(f1, 3),
        "precision": round(p, 3),
        "recall": round(r, 3),
        "gt_count": len(ground_truth),
        "pred_count": len(predicted),
        "ground_truth": ground_truth,
        "predicted": predicted,
        "issues": issues,
    }


def root_cause(analysis: dict) -> str:
    """One-line human-readable root cause explanation."""
    pf = analysis["primary_failure"]
    gt_n = analysis["gt_count"]
    pr_n = analysis["pred_count"]

    if pf == "GOOD":
        return "good match"

    if pf == "COVERAGE_FAILURE":
        return (f"model predicted only {pr_n} of {gt_n} ingredients "
                f"(recall={analysis['recall']}) — likely missed spices, condiments, or secondary items")

    misids = [i for i in analysis["issues"] if i["type"] == "MISIDENTIFICATION"]
    if pf == "MISIDENTIFICATION" and misids:
        examples = ", ".join(f"'{i['predicted']}'" for i in misids[:2])
        return f"model hallucinated wrong foods: {examples} (sim < {MISID_SIM} to any GT item)"

    missed = [i for i in analysis["issues"] if i["type"] == "MISSED_INGREDIENT"]
    if missed:
        examples = ", ".join(f"'{i['missed']}'" for i in missed[:3])
        suffix = f" (+{len(missed)-3} more)" if len(missed) > 3 else ""
        return f"missed GT ingredients: {examples}{suffix}"

    near = [i for i in analysis["issues"] if i["type"] == "SEMANTIC_NEAR_MISS"]
    if near:
        examples = ", ".join(f"'{i['ground_truth']}' predicted as '{i['predicted_as']}'" for i in near[:2])
        return f"near-miss naming: {examples} — synonym not in lookup table"

    return "mixed errors"


def analyze(results_path: str, top: int | None = None, min_f1: float | None = None) -> None:
    data = json.loads(Path(results_path).read_text())
    samples = data.get("samples", [])

    if not samples:
        print("No samples found in results file.")
        return

    analyses = [classify_sample(s) for s in samples]

    if min_f1 is not None:
        analyses = [a for a in analyses if a["f1"] < min_f1]

    # Sort worst first
    analyses.sort(key=lambda a: a["f1"])

    if top:
        analyses = analyses[:top]

    # --- Summary stats ---
    all_analyses = [classify_sample(s) for s in samples]
    total = len(all_analyses)
    category_counts: dict[str, int] = {}
    for a in all_analyses:
        category_counts[a["primary_failure"]] = category_counts.get(a["primary_failure"], 0) + 1

    avg_f1 = sum(a["f1"] for a in all_analyses) / total if total else 0
    avg_p  = sum(a["precision"] for a in all_analyses) / total if total else 0
    avg_r  = sum(a["recall"] for a in all_analyses) / total if total else 0

    print("=" * 70)
    print(f"FAILURE ANALYSIS — {Path(results_path).name}")
    print("=" * 70)
    print(f"\nModel:   {data.get('model', 'unknown')}")
    print(f"Mode:    {data.get('mode', 'unknown')}")
    print(f"Samples: {total}")
    print(f"\nAggregate:  P={avg_p:.4f}  R={avg_r:.4f}  F1={avg_f1:.4f}")
    print()
    print("Failure breakdown:")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        print(f"  {cat:<22} {count:>4}  ({pct:.1f}%)")

    # --- GT complexity vs F1 ---
    simple   = [a for a in all_analyses if a["gt_count"] <= 3]
    moderate = [a for a in all_analyses if 4 <= a["gt_count"] <= 8]
    complex_ = [a for a in all_analyses if a["gt_count"] > 8]

    print()
    print("F1 by dish complexity (# GT ingredients):")
    for label, group in [("1-3 ingredients", simple), ("4-8 ingredients", moderate), (">8 ingredients", complex_)]:
        if group:
            avg = sum(a["f1"] for a in group) / len(group)
            print(f"  {label:<22} n={len(group):>3}  avg F1={avg:.3f}")

    # --- Detailed worst cases ---
    print()
    print(f"{'=' * 70}")
    print(f"WORST CASES (showing {len(analyses)}, sorted by F1 ascending)")
    print(f"{'=' * 70}")

    for i, a in enumerate(analyses, 1):
        print(f"\n[{i}] {a['dish_id']}  (cam={a['camera']})")
        print(f"     F1={a['f1']}  P={a['precision']}  R={a['recall']}")
        print(f"     GT ({a['gt_count']}): {a['ground_truth']}")
        print(f"     Pred ({a['pred_count']}): {a['predicted']}")
        print(f"     Root cause: {root_cause(a)}")
        if a["issues"]:
            for issue in a["issues"][:5]:
                itype = issue["type"]
                if itype == "MISIDENTIFICATION":
                    print(f"       ✗ MISID: '{issue['predicted']}' → sim={issue['best_sim']} to best GT match '{issue['best_gt_match']}'")
                elif itype == "MISSED_INGREDIENT":
                    print(f"       ✗ MISSED: '{issue['missed']}' (best pred match sim={issue['best_sim']})")
                elif itype == "SEMANTIC_NEAR_MISS":
                    print(f"       ~ NEAR-MISS: GT='{issue['ground_truth']}' predicted as '{issue['predicted_as']}' (sim={issue['sim']})")
                elif itype == "COVERAGE_FAILURE":
                    print(f"       ✗ COVERAGE: {issue['predicted_count']}/{issue['gt_count']} ingredients found")

    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", help="Path to eval results JSON")
    parser.add_argument("--top", type=int, default=None, help="Show top N worst cases")
    parser.add_argument("--min-f1", type=float, default=None,
                        help="Only show cases below this F1 (default: all)")
    args = parser.parse_args()
    analyze(args.results, top=args.top, min_f1=args.min_f1)


if __name__ == "__main__":
    main()
