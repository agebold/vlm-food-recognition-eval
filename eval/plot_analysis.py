"""
Generate analysis plots for the VLM food recognition evaluation.

Produces:
  plots/01_prediction_accuracy.png   — per-model prediction accuracy bar chart
  plots/02_correct_per_dish_dist.png — 2x3 grid: correct ingredients/dish histogram per model
  plots/03_missed_ingredients.png    — top missed ingredients across all models
  plots/04_hallucinated.png          — 2x3 grid: top hallucinated ingredients per model
  plots/05_f1_by_complexity.png      — line chart: F1 vs dish complexity per model
  plots/06_pred_accuracy_dist.png    — horizontal boxplots: per-dish prediction accuracy

Run: python3 plot_analysis.py
"""

import json, pathlib, sys, collections, math
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

# Key models for plots 2 & 4 (6 models → clean 2×3 grid)
MODELS = {
    "Kimi K2.5":           "results/bedrock_kimi25_overhead.json",
    "Claude Opus 4.8":     "results/claude_opus_overhead_full.json",
    "Llama4 Maverick 17B": "results/bedrock_llama4_maverick_full.json",
    "Llama4 Scout 17B":    "results/bedrock_llama4_scout_full.json",
    "Nova Pro":            "results/bedrock_nova_pro_full.json",
    "Nova Lite":           "results/bedrock_nova_lite_full.json",
}
COLORS = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948"]


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

names  = list(data.keys())
colors = COLORS[:len(names)]


# ── Plot 1: Prediction accuracy bar chart ────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
accs = [100 * np.mean([d["pred_acc"] for d in data[n]]) for n in names]
# Sort descending
order = np.argsort(accs)[::-1]
sorted_names  = [names[i]  for i in order]
sorted_accs   = [accs[i]   for i in order]
sorted_colors = [colors[i] for i in order]
bars = ax.bar(sorted_names, sorted_accs, color=sorted_colors, width=0.55,
              edgecolor="white", linewidth=0.8)
ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=10, fontweight="bold")
ax.set_ylim(75, 90)
ax.set_ylabel("Prediction accuracy (%)", fontsize=11)
ax.set_title("When a model names an ingredient, how often is it actually in the dish?",
             fontsize=11, pad=12)
ax.set_xticklabels(sorted_names, fontsize=9, rotation=15, ha="right")
ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
ax.grid(axis="y", alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(PLOT_DIR / "01_prediction_accuracy.png", dpi=150)
plt.close()
print("Saved 01_prediction_accuracy.png")


# ── Plot 2: Correct ingredients/dish — 2×3 grid ──────────────────────────────
NCOLS, NROWS = 3, 2
fig, axes = plt.subplots(NROWS, NCOLS, figsize=(12, 8), sharey=False)
axes_flat = axes.flatten()

for idx, (name, color) in enumerate(zip(names, colors)):
    ax = axes_flat[idx]
    vals = [d["n_correct"] for d in data[name]]
    max_v = max(vals) if vals else 1
    bins = range(0, max_v + 2)
    counts, edges = np.histogram(vals, bins=bins)
    ax.bar(edges[:-1], counts, width=0.8, color=color, edgecolor="white", alpha=0.9)
    med = float(np.median(vals))
    mean = float(np.mean(vals))
    ax.axvline(med,  color="black",  linestyle="--", linewidth=1.2, label=f"median {med:.0f}")
    ax.axvline(mean, color="dimgray", linestyle=":",  linewidth=1.0, label=f"mean {mean:.1f}")
    ax.set_title(name, fontsize=10, fontweight="bold", pad=6)
    ax.set_xlabel("Correct ingredients / dish", fontsize=8)
    ax.set_ylabel("Dishes", fontsize=8)
    ax.legend(fontsize=7, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)

# Hide unused slots
for idx in range(len(names), NROWS * NCOLS):
    axes_flat[idx].set_visible(False)

fig.suptitle("Distribution of correctly identified ingredients per dish", fontsize=12, y=1.01)
fig.tight_layout()
fig.savefig(PLOT_DIR / "02_correct_per_dish_dist.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved 02_correct_per_dish_dist.png")


# ── Plot 3: Top missed ingredients (across ALL models combined) ──────────────
all_missed = collections.Counter()
for name in names:
    for d in data[name]:
        for ing in d["missed"]:
            all_missed[ing] += 1

top_missed = [ing for ing, _ in all_missed.most_common(25)]
top_counts = [all_missed[ing] for ing in top_missed]

fig, ax = plt.subplots(figsize=(11, 7))
y = np.arange(len(top_missed))
bars = ax.barh(y, top_counts, color="#4e79a7", edgecolor="white", alpha=0.85)
ax.set_yticks(y)
ax.set_yticklabels(top_missed, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel(f"Times missed across all {len(names)} models × ~507 dishes", fontsize=10)
ax.set_title("Top 25 ingredients most often missed (GT present, model didn't name it)",
             fontsize=11, pad=12)
ax.bar_label(bars, padding=3, fontsize=8)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(PLOT_DIR / "03_missed_ingredients.png", dpi=150)
plt.close()
print("Saved 03_missed_ingredients.png")


# ── Plot 4: Top hallucinated per model — 2×3 grid ───────────────────────────
fig, axes = plt.subplots(NROWS, NCOLS, figsize=(14, 10), sharey=False)
axes_flat = axes.flatten()

for idx, (name, color) in enumerate(zip(names, colors)):
    ax = axes_flat[idx]
    halluc_ctr = collections.Counter()
    for d in data[name]:
        for ing in d["halluc"]:
            halluc_ctr[ing] += 1
    top = halluc_ctr.most_common(10)
    if not top:
        ax.text(0.5, 0.5, "No hallucinations", ha="center", va="center",
                transform=ax.transAxes, fontsize=9)
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.set_visible(True)
        continue
    ings, cnts = zip(*top)
    y = np.arange(len(ings))
    ax.barh(y, cnts, color=color, edgecolor="white", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(ings, fontsize=8)
    ax.invert_yaxis()
    ax.set_title(name, fontsize=10, fontweight="bold", pad=6)
    ax.set_xlabel("Times hallucinated", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.3)
    ax.tick_params(labelsize=8)

for idx in range(len(names), NROWS * NCOLS):
    axes_flat[idx].set_visible(False)

fig.suptitle("Top hallucinated ingredients per model (predicted but not in GT)", fontsize=12)
fig.tight_layout()
fig.savefig(PLOT_DIR / "04_hallucinated.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved 04_hallucinated.png")


# ── Plot 5: F1 by GT complexity — line chart ─────────────────────────────────
buckets = [(1, 3, "1–3\n(simple)"), (4, 8, "4–8\n(moderate)"), (9, 99, "9+\n(complex)")]

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(buckets))

for name, color in zip(names, colors):
    samples = load(MODELS[name])
    avgs = []
    for lo, hi, _ in buckets:
        group = [s["scores"]["f1"] for s in samples if lo <= len(s["ground_truth"]) <= hi]
        avgs.append(float(np.mean(group)) if group else float("nan"))
    ax.plot(x, avgs, marker="o", markersize=7, linewidth=2, label=name, color=color)
    # Label endpoint
    if not math.isnan(avgs[-1]):
        ax.annotate(f"{avgs[-1]:.2f}", xy=(x[-1], avgs[-1]),
                    xytext=(4, 0), textcoords="offset points",
                    fontsize=7.5, color=color, va="center")

ax.set_xticks(x)
ax.set_xticklabels([b[2] for b in buckets], fontsize=10)
ax.set_ylabel("Average F1  (PMC13092701 metric)", fontsize=10)
ax.set_xlabel("Number of GT ingredients in dish", fontsize=10)
ax.set_title("All models degrade on complex dishes — F1 by dish complexity", fontsize=11, pad=12)
ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
ax.set_ylim(0.3, 0.95)
ax.grid(axis="y", alpha=0.3)
ax.grid(axis="x", alpha=0.15)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(PLOT_DIR / "05_f1_by_complexity.png", dpi=150)
plt.close()
print("Saved 05_f1_by_complexity.png")


# ── Plot 6: Per-dish prediction accuracy — horizontal boxplots ───────────────
fig, ax = plt.subplots(figsize=(9, 5))

# Reverse so highest model is at top
plot_names  = list(reversed(names))
plot_colors = list(reversed(colors))
plot_data   = [[d["pred_acc"] * 100 for d in data[n]] for n in plot_names]

bp = ax.boxplot(plot_data, vert=False, patch_artist=True,
                medianprops=dict(color="black", linewidth=2),
                whiskerprops=dict(linewidth=1.2),
                flierprops=dict(marker=".", markersize=3, alpha=0.4))

for patch, color in zip(bp["boxes"], plot_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)

# Annotate median values
for i, vals in enumerate(plot_data, start=1):
    med = float(np.median(vals))
    ax.text(med + 0.5, i, f"{med:.0f}%", va="center", fontsize=8, fontweight="bold")

ax.set_yticks(range(1, len(plot_names) + 1))
ax.set_yticklabels(plot_names, fontsize=9)
ax.set_xlabel("Per-dish prediction accuracy — correct / predicted  (%)", fontsize=10)
ax.set_title("Prediction accuracy distribution per model\n(box = IQR, line = median, dots = outliers)",
             fontsize=11, pad=10)
ax.set_xlim(0, 110)
ax.grid(axis="x", alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(PLOT_DIR / "06_pred_accuracy_dist.png", dpi=150)
plt.close()
print("Saved 06_pred_accuracy_dist.png")

print(f"\nAll plots saved to eval/plots/  ({len(names)} models)")
