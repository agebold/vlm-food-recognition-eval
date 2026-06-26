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
# Shared x cap at 99th percentile across ALL models (outlier-robust)
_all_correct = [d["n_correct"] for name in names for d in data[name]]
X_CAP = int(np.percentile(_all_correct, 99)) + 2   # e.g. 9 for most runs

NCOLS, NROWS = 3, 2
fig, axes = plt.subplots(NROWS, NCOLS, figsize=(13, 8), sharey=False)
axes_flat = axes.flatten()

for idx, (name, color) in enumerate(zip(names, colors)):
    ax = axes_flat[idx]
    vals = [d["n_correct"] for d in data[name]]
    # Clip outliers for display — note in title if any are clipped
    clipped = sum(1 for v in vals if v > X_CAP)
    vals_clipped = [min(v, X_CAP) for v in vals]
    bins = range(0, X_CAP + 2)
    counts, edges = np.histogram(vals_clipped, bins=bins)
    ax.bar(edges[:-1], counts, width=0.8, color=color, edgecolor="white")
    med  = float(np.median(vals))
    mean = float(np.mean(vals))
    ax.axvline(med,  color="#111827", linestyle="--", linewidth=1.8, label=f"median {med:.0f}")
    ax.axvline(mean, color="#D97706", linestyle=":",  linewidth=1.5, label=f"mean {mean:.1f}")
    title = name + (f"  [{clipped} outlier>cap]" if clipped else "")
    ax.set_title(title, fontsize=10, fontweight="bold", color=color, pad=6)
    ax.set_xlabel("Correct ingredients per dish", fontsize=9)
    ax.set_ylabel("Number of dishes", fontsize=9)
    ax.set_xlim(0, X_CAP + 1)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.85)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)

for idx in range(len(names), NROWS * NCOLS):
    axes_flat[idx].set_visible(False)

fig.suptitle("Distribution of correctly identified ingredients per dish  (shared x-axis, outliers capped)",
             fontsize=12, fontweight="bold", y=1.01)
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


# ── Plot 5: F1 by GT complexity — 3-panel grouped bar chart ─────────────────
BUCK5 = [(1, 3, "Simple\n(1–3 ingredients)"),
         (4, 8, "Moderate\n(4–8 ingredients)"),
         (9, 99, "Complex\n(9+ ingredients)")]

# Compute F1 per model per bucket
f1_data = {}
for name in names:
    samples = load(MODELS[name])
    f1_data[name] = []
    for lo, hi, _ in BUCK5:
        grp = [s["scores"]["f1"] for s in samples if lo <= len(s["ground_truth"]) <= hi]
        f1_data[name].append((float(np.mean(grp)) if grp else float("nan"), len(grp)))

# Sort models by Simple F1 descending for consistent ordering
sort_idx = sorted(range(len(names)), key=lambda i: f1_data[names[i]][0][0], reverse=True)
snames = [names[i]  for i in sort_idx]
scolors = [colors[i] for i in sort_idx]

# Global tight x range (same across all 3 panels)
all_f1_vals = [v for n in names for v, _ in f1_data[n] if not math.isnan(v)]
x_lo = max(0.0, min(all_f1_vals) - 0.03)
x_hi = max(all_f1_vals) + 0.04

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
y_pos = np.arange(len(snames))

for bi, (ax, (lo, hi, label)) in enumerate(zip(axes, BUCK5)):
    f1_vals = [f1_data[n][bi][0] for n in snames]
    n_dishes = f1_data[snames[0]][bi][1]

    bars = ax.barh(y_pos, f1_vals, color=scolors, height=0.65, edgecolor="white", linewidth=0.5)

    # Value labels at bar end, bold, model color
    for bar, val, col in zip(bars, f1_vals, scolors):
        if not math.isnan(val):
            ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=10, fontweight="bold", color=col)

    ax.set_yticks(y_pos)
    if bi == 0:
        ax.set_yticklabels(snames, fontsize=10, fontweight="bold")
        for tick, col in zip(ax.get_yticklabels(), scolors):
            tick.set_color(col)
    else:
        ax.set_yticklabels([])
    ax.set_xlim(x_lo, x_hi)
    ax.set_xlabel("Average F1", fontsize=10)
    ax.set_title(f"{label}\n(n={n_dishes} dishes)", fontsize=11, fontweight="bold", pad=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25)
    ax.tick_params(axis="x", labelsize=9)

fig.suptitle("F1 degrades with dish complexity — same x-scale across all panels", fontsize=12,
             fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(PLOT_DIR / "05_f1_by_complexity.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved 05_f1_by_complexity.png")


# ── Plot 6: Prediction accuracy buckets — stacked horizontal bars ─────────────
#  Matches D3 fig06: Perfect / Good / Fair / Poor per model
from matplotlib.patches import Patch

BUCK6 = [
    ("Perfect (100%)",  100, 100, "#059669"),
    ("Good (75–99%)",    75,  99, "#1D4ED8"),
    ("Fair (50–74%)",    50,  74, "#D97706"),
    ("Poor (<50%)",       0,  49, "#DC2626"),
]

model_rows = []
for name in names:
    vals = [round(d["pred_acc"] * 100) for d in data[name]]
    total = len(vals)
    segs = []
    for label, lo, hi, col in BUCK6:
        count = sum(1 for v in vals if lo <= v <= hi)
        segs.append({"label": label, "pct": 100 * count / total, "color": col})
    model_rows.append({"name": name, "segs": segs, "perfect": segs[0]["pct"]})

# Sort by Perfect descending
model_rows.sort(key=lambda r: r["perfect"], reverse=True)

fig, ax = plt.subplots(figsize=(10, 5))
y_pos = np.arange(len(model_rows))

for i, row in enumerate(model_rows):
    left = 0.0
    for seg in row["segs"]:
        ax.barh(i, seg["pct"], left=left, color=seg["color"], height=0.65,
                edgecolor="white", linewidth=0.5)
        if seg["pct"] >= 5:
            ax.text(left + seg["pct"] / 2, i, f"{seg['pct']:.0f}%",
                    ha="center", va="center", fontsize=10, fontweight="bold", color="white")
        left += seg["pct"]

ax.set_yticks(y_pos)
ax.set_yticklabels([r["name"] for r in model_rows], fontsize=10, fontweight="bold")
ax.set_xlabel("Share of dishes (%)", fontsize=10)
ax.set_xlim(0, 100)
ax.set_title("Per-dish prediction accuracy — what fraction of dishes did each model nail?",
             fontsize=11, fontweight="bold", pad=12)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", alpha=0.2)

legend_els = [Patch(facecolor=col, label=lbl) for lbl, _, _, col in BUCK6]
ax.legend(handles=legend_els, loc="lower right", fontsize=9, framealpha=0.9)

fig.tight_layout()
fig.savefig(PLOT_DIR / "06_pred_accuracy_dist.png", dpi=150)
plt.close()
print("Saved 06_pred_accuracy_dist.png")

print(f"\nAll plots saved to eval/plots/  ({len(names)} models)")
