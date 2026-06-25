"""
Generate analysis plots for the VLM food recognition evaluation.

Produces:
  plots/01_prediction_accuracy.png   — per-model prediction accuracy bar chart
  plots/02_correct_per_dish_dist.png — distribution of correct ingredients/dish
  plots/03_missed_ingredients.png    — top missed ingredients across all models
  plots/04_hallucinated.png          — top hallucinated ingredients per model
  plots/05_f1_by_complexity.png      — F1 vs dish complexity (GT ingredient count)
  plots/06_pred_accuracy_hist.png    — per-dish prediction accuracy histogram

Run: python3 plot_analysis.py
"""

import json, pathlib, sys, collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

sys.path.insert(0, ".")
from metric import sim

THRESH = 0.4
PLOT_DIR = pathlib.Path("plots")
PLOT_DIR.mkdir(exist_ok=True)

MODELS = {
    "Claude Opus 4.8":     "results/claude_opus_overhead_full.json",
    "Llama4 Scout 17B":    "results/bedrock_llama4_scout_full.json",
    "Llama4 Maverick 17B": "results/bedrock_llama4_maverick_full.json",
    "Nova Pro":            "results/bedrock_nova_pro_full.json",
    "Nova Lite":           "results/bedrock_nova_lite_full.json",
}
COLORS = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f"]


def load(path):
    return json.loads(pathlib.Path(path).read_text())["samples"]


def per_dish_stats(samples):
    stats = []
    for s in samples:
        gt, pred = s["ground_truth"], s["predicted"]
        correct = [p for p in pred if max((sim(p, g) for g in gt), default=0) >= THRESH]
        wrong   = [p for p in pred if max((sim(p, g) for g in gt), default=0) < THRESH]
        missed  = [g for g in gt  if max((sim(g, p) for p in pred), default=0) < THRESH]
        stats.append({
            "n_gt": len(gt), "n_pred": len(pred),
            "n_correct": len(correct), "n_wrong": len(wrong), "n_missed": len(missed),
            "pred_acc": len(correct) / len(pred) if pred else 0,
            "gt": gt, "pred": pred, "correct": correct, "missed": missed, "halluc": wrong,
        })
    return stats


# ── Load all data ────────────────────────────────────────────────────────────
data = {}
for name, path in MODELS.items():
    try:
        data[name] = per_dish_stats(load(path))
    except FileNotFoundError:
        print(f"SKIP {name}: file not found")

names   = list(data.keys())
colors  = COLORS[:len(names)]


# ── Plot 1: Prediction accuracy bar chart ────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
accs = [100 * np.mean([d["pred_acc"] for d in data[n]]) for n in names]
bars = ax.bar(names, accs, color=colors, width=0.55, edgecolor="white", linewidth=0.8)
ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=10, fontweight="bold")
ax.set_ylim(78, 87)
ax.set_ylabel("Prediction accuracy (%)", fontsize=11)
ax.set_title("When a model names an ingredient, how often is it actually in the dish?", fontsize=11, pad=12)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, fontsize=9)
ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
ax.grid(axis="y", alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(PLOT_DIR / "01_prediction_accuracy.png", dpi=150)
plt.close()
print("Saved 01_prediction_accuracy.png")


# ── Plot 2: Distribution of correct ingredients per dish ─────────────────────
fig, axes = plt.subplots(1, len(names), figsize=(16, 4), sharey=True)
for ax, (name, color) in zip(axes, zip(names, colors)):
    vals = [d["n_correct"] for d in data[name]]
    max_v = max(vals)
    bins = range(0, max_v + 2)
    counts, edges = np.histogram(vals, bins=bins)
    ax.bar(edges[:-1], counts, width=0.8, color=color, edgecolor="white", alpha=0.9)
    ax.set_title(name, fontsize=8, fontweight="bold")
    ax.set_xlabel("Correct ingredients/dish", fontsize=8)
    ax.axvline(np.median(vals), color="black", linestyle="--", linewidth=1, label=f"median={np.median(vals):.0f}")
    ax.legend(fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Number of dishes", fontsize=9)
fig.suptitle("Distribution of correctly identified ingredients per dish", fontsize=11, y=1.01)
fig.tight_layout()
fig.savefig(PLOT_DIR / "02_correct_per_dish_dist.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved 02_correct_per_dish_dist.png")


# ── Plot 3: Top missed ingredients (across ALL models combined) ──────────────
all_missed = collections.Counter()
per_model_missed = {n: collections.Counter() for n in names}
for name in names:
    for d in data[name]:
        for ing in d["missed"]:
            all_missed[ing] += 1
            per_model_missed[name][ing] += 1

top_missed = [ing for ing, _ in all_missed.most_common(25)]
top_counts = [all_missed[ing] for ing in top_missed]

fig, ax = plt.subplots(figsize=(11, 7))
y = np.arange(len(top_missed))
bars = ax.barh(y, top_counts, color="#4e79a7", edgecolor="white", alpha=0.85)
ax.set_yticks(y)
ax.set_yticklabels(top_missed, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("Times missed across all 5 models × 507 dishes", fontsize=10)
ax.set_title("Top 25 ingredients most often missed (GT present, model didn't name it)", fontsize=11, pad=12)
ax.bar_label(bars, padding=3, fontsize=8)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(PLOT_DIR / "03_missed_ingredients.png", dpi=150)
plt.close()
print("Saved 03_missed_ingredients.png")


# ── Plot 4: Top hallucinated per model (side by side) ───────────────────────
fig, axes = plt.subplots(1, len(names), figsize=(18, 6), sharey=False)
for ax, (name, color) in zip(axes, zip(names, colors)):
    halluc_ctr = collections.Counter()
    for d in data[name]:
        for ing in d["halluc"]:
            halluc_ctr[ing] += 1
    top = halluc_ctr.most_common(12)
    if not top:
        ax.set_visible(False)
        continue
    ings, cnts = zip(*top)
    y = np.arange(len(ings))
    ax.barh(y, cnts, color=color, edgecolor="white", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(ings, fontsize=8)
    ax.invert_yaxis()
    ax.set_title(name, fontsize=8, fontweight="bold")
    ax.set_xlabel("Times hallucinated", fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.3)
fig.suptitle("Top hallucinated ingredients per model (predicted but not in GT)", fontsize=11)
fig.tight_layout()
fig.savefig(PLOT_DIR / "04_hallucinated.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved 04_hallucinated.png")


# ── Plot 5: F1 by GT complexity buckets ─────────────────────────────────────
buckets     = [(1, 3, "1–3"), (4, 8, "4–8"), (9, 99, "9+")]
bucket_lbls = [b[2] for b in buckets]
x = np.arange(len(bucket_lbls))
width = 0.15

fig, ax = plt.subplots(figsize=(9, 5))
for i, (name, color) in enumerate(zip(names, colors)):
    avgs = []
    for lo, hi, _ in buckets:
        group = [s["scores"]["f1"] for s in load(MODELS[name])
                 if lo <= len(s["ground_truth"]) <= hi]
        avgs.append(np.mean(group) if group else 0)
    ax.bar(x + i * width, avgs, width=width, label=name, color=color, edgecolor="white")

ax.set_xticks(x + width * (len(names) - 1) / 2)
ax.set_xticklabels([f"{l} GT ingredients" for l in bucket_lbls], fontsize=10)
ax.set_ylabel("Average F1 (PMC13092701 metric)", fontsize=10)
ax.set_title("Model performance degrades sharply on complex dishes", fontsize=11, pad=12)
ax.legend(fontsize=8, loc="upper right")
ax.set_ylim(0.45, 0.85)
ax.grid(axis="y", alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(PLOT_DIR / "05_f1_by_complexity.png", dpi=150)
plt.close()
print("Saved 05_f1_by_complexity.png")


# ── Plot 6: Per-dish prediction accuracy histogram ───────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
bins = np.linspace(0, 1, 21)
for name, color in zip(names, colors):
    vals = [d["pred_acc"] for d in data[name]]
    ax.hist(vals, bins=bins, alpha=0.55, label=name, color=color, edgecolor="white")
ax.set_xlabel("Prediction accuracy per dish (correct / predicted)", fontsize=10)
ax.set_ylabel("Number of dishes", fontsize=10)
ax.set_title("Distribution of per-dish prediction accuracy across all models", fontsize=11, pad=12)
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(PLOT_DIR / "06_pred_accuracy_hist.png", dpi=150)
plt.close()
print("Saved 06_pred_accuracy_hist.png")

print("\nAll plots saved to eval/plots/")
