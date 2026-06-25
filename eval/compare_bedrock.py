#!/usr/bin/env python3
"""Print a P/R/F1 comparison table across the 5 Bedrock evaluators."""
import json
from pathlib import Path

FILES = {
    "Haiku 4.5":   "results/bedrock_haiku45_overhead.json",
    "Sonnet 4.6":  "results/bedrock_sonnet46_overhead.json",
    "Nova 2 Lite": "results/bedrock_nova2lite_overhead.json",
    "Qwen3-VL":    "results/bedrock_qwen3vl_overhead.json",
    "Kimi K2.5":   "results/bedrock_kimi25_overhead.json",
}

rows = []
for name, path in FILES.items():
    p = Path(path)
    if not p.exists():
        rows.append((name, None))
        continue
    d = json.loads(p.read_text())
    rows.append((name, d))

rows.sort(key=lambda r: (r[1]["aggregate"]["f1"] if r[1] else -1), reverse=True)

print(f"\n{'Model':<13} {'P':>7} {'R':>7} {'F1':>7} {'n':>5} {'skip':>5}")
print("-" * 50)
for name, d in rows:
    if d is None:
        print(f"{name:<13} {'(no results yet)':>37}")
        continue
    a = d["aggregate"]
    print(f"{name:<13} {a['precision']:>7.3f} {a['recall']:>7.3f} "
          f"{a['f1']:>7.3f} {d['n_samples']:>5} {d['n_skipped']:>5}")
print()
