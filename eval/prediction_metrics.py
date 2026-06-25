#!/usr/bin/env python3
"""Prediction-quality metrics for the Bedrock evaluators.

A prediction is CORRECT when sim(pred, best GT) >= THRESH (0.4) — same as
plot_analysis.py. Reports the metrics that actually matter on this dataset
(F1/recall are misleading because GT includes invisible ingredients):

  Pred accuracy   = mean over dishes of (n_correct / n_pred)
  Correct/dish    = mean & median of n_correct
  Zero-detection  = % dishes with 0 correct
  All-correct     = % dishes where every prediction was correct (n_pred>0)
  Avg predictions = mean n_pred
"""
import json
import statistics as st
from pathlib import Path
from metric import sim

THRESH = 0.4

FILES = {
    "Kimi K2.5":           "results/bedrock_kimi25_overhead.json",
    "Llama4 Maverick 17B": "results/bedrock_llama4_maverick_full.json",
    "Llama4 Scout 17B":    "results/bedrock_llama4_scout_full.json",
    "Qwen3-VL 235B":       "results/bedrock_qwen3vl_overhead.json",
    "Claude Opus 4.8":     "results/claude_opus_overhead_full.json",
    "Nova Lite":           "results/bedrock_nova_lite_full.json",
    "Nova Pro":            "results/bedrock_nova_pro_full.json",
    "Sonnet 4.6":          "results/bedrock_sonnet46_overhead.json",
    "Nova 2 Lite":         "results/bedrock_nova2lite_overhead.json",
    "Haiku 4.5":           "results/bedrock_haiku45_overhead.json",
}


def per_dish(samples):
    rows = []
    for s in samples:
        gt, pred = s["ground_truth"], s["predicted"]
        n_correct = sum(1 for p in pred if max((sim(p, g) for g in gt), default=0) >= THRESH)
        rows.append({
            "n_pred": len(pred),
            "n_correct": n_correct,
            "pred_acc": n_correct / len(pred) if pred else 0.0,
            "all_correct": len(pred) > 0 and n_correct == len(pred),
            "zero": n_correct == 0,
        })
    return rows


def summarize(rows):
    n = len(rows)
    correct = [r["n_correct"] for r in rows]
    return {
        "pred_acc": 100 * st.mean(r["pred_acc"] for r in rows),
        "correct_mean": st.mean(correct),
        "correct_median": st.median(correct),
        "zero": 100 * sum(r["zero"] for r in rows) / n,
        "all_correct": 100 * sum(r["all_correct"] for r in rows) / n,
        "avg_pred": st.mean(r["n_pred"] for r in rows),
        "n": n,
    }


results = {}
for name, path in FILES.items():
    p = Path(path)
    if not p.exists():
        continue
    samples = json.loads(p.read_text())["samples"]
    results[name] = summarize(per_dish(samples))

order = sorted(results, key=lambda k: results[k]["pred_acc"], reverse=True)

hdr = f"{'Model':<13} {'PredAcc':>8} {'Correct/dish':>14} {'Zero-det':>9} {'All-correct':>12} {'AvgPred':>8} {'n':>5}"
print("\n" + hdr)
print("-" * len(hdr))
for name in order:
    m = results[name]
    cd = f"{m['correct_mean']:.1f} / {m['correct_median']:.0f}"
    print(f"{name:<13} {m['pred_acc']:>7.1f}% {cd:>14} {m['zero']:>8.1f}% "
          f"{m['all_correct']:>11.1f}% {m['avg_pred']:>8.1f} {m['n']:>5}")
print()
