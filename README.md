# VLM Food Recognition Evaluation

Reproducible evaluation harness for photo-based food ingredient recognition using vision-language models (VLMs), benchmarked against the [Nutrition5k](https://github.com/google-research-datasets/Nutrition5k) dataset.

## What This Is

We replicate and extend the evaluation protocol from [PMC13092701](https://pmc.ncbi.nlm.nih.gov/articles/PMC13092701/) — a peer-reviewed April 2026 study that tested 16 VLMs (including Llama 3.2-90B-Vision and Gemini 2.5 Flash) on ingredient-level food recognition. The study reported Llama 3.2-90B-Vision achieving **0.6626–0.6967 precision** on Nutrition5k.

Our first result: **gemma4 (8B, local, Q4_K_M quantized) hits 0.69 precision** on the same dataset in a 5-dish sanity run — matching a 90B cloud model from a local machine.

---

## Key Finding from Our Research

> The PMC13092701 precision figures are **not inherited baselines** from Nutrition5k. The original dataset paper has no precision/recall/F1 at all — it only measures calorie/mass/macro MAE. PMC13092701 invented a custom **LCS + synonym-list similarity function** (Equations 1–6) to produce those numbers.

This harness implements that exact metric, making our results directly comparable to the paper.

---

## Metric: PMC13092701 Equations 1–6

Ingredient matching is **soft** — partial name matches contribute fractionally via normalized LCS, and known synonyms match perfectly.

```
Sim(a, b)    = max( StrMatch(a, b),  SemMatch(a, b) )

StrMatch(a,b) = 2 × |LCS(a,b)| / (|a| + |b|)          # Eq 5 — normalized LCS
SemMatch(a,b) = 1.0 if b ∈ Var(a) or a ∈ Var(b)        # Eq 6 — synonym lookup
               else 0.0

Precision    = Σ p∈P  max t∈T  Sim(p,t)  /  |P|        # Eq 1
Recall       = Σ t∈T  max p∈P  Sim(t,p)  /  |T|        # Eq 2
F1           = 2 × P × R / (P + R)                      # Eq 3
```

Where `P` = predicted ingredient list (from VLM), `T` = ground-truth ingredient list (from Nutrition5k CSV).

---

## Dataset: Nutrition5k

- **5,006 plates** of real cafeteria food with per-ingredient mass, calorie, fat, carb, protein annotations
- **Source:** USDA Food and Nutrient Database
- **Splits:** 90% train (~4,505) / 10% test
  - `rgb_test_ids.txt` — 709 dishes (side-angle H.264 videos, cameras A–D)
  - `depth_test_ids.txt` — 507 dishes (overhead RGB-D PNG images) ← **what this harness uses**
- **Download:** public GCS bucket, no auth required

```bash
gsutil ls gs://nutrition5k_dataset/
```

We use the **overhead RGB split** (`depth_test_ids.txt`, 507 dishes) because images are pre-extracted PNGs — no video decoding needed. The side-angle split stores raw H.264 video requiring ffmpeg frame extraction.

---

## Project Structure

```
eval/
├── download_dataset.sh       # pull splits + metadata (~10MB) or test images
├── synonyms.py               # 100+ food synonym groups → bidirectional lookup
├── metric.py                 # PMC13092701 Eq 1-6 (LCS + synonym P/R/F1)
├── parse_nutrition5k.py      # load ground-truth CSVs, locate image files
├── run_eval.py               # main runner → Ollama / any OpenAI-compat API → score → JSON
├── requirements.txt
└── results/                  # output JSONs (gitignored — too large)
```

---

## Setup

```bash
# 1. Install dependencies (just requests + tqdm)
pip install -r eval/requirements.txt

# 2. Download splits + metadata (~10MB, fast)
cd eval && bash download_dataset.sh

# 3. Download overhead test images (507 dishes, ~194MB)
bash download_dataset.sh --images
# or manually:
mkdir -p data/nutrition5k/imagery/realsense_overhead
while IFS= read -r dish_id; do
  gsutil cp "gs://nutrition5k_dataset/nutrition5k_dataset/imagery/realsense_overhead/$dish_id/rgb.png" \
            "data/nutrition5k/imagery/realsense_overhead/$dish_id.png"
done < data/nutrition5k/dish_ids/splits/depth_test_ids.txt

# 4. Make sure Ollama is running with a vision model
ollama pull gemma4
ollama serve
```

---

## Running the Eval

```bash
cd eval

# Quick sanity check — 5 dishes
python3 run_eval.py --overhead --limit 5 --out results/sanity.json

# Full 507-dish eval with gemma4
python3 run_eval.py --overhead --model gemma4:latest --out results/gemma4_full.json

# Side-angle cameras (requires H.264 → frame extraction, see notes)
python3 run_eval.py --cameras A B C D --model gemma4:latest --out results/gemma4_side.json
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--model` | `gemma4:latest` | Ollama model name |
| `--overhead` | false | Use overhead PNGs (depth split, 507 dishes) |
| `--cameras` | `A` | Side cameras to evaluate (A B C D) |
| `--frame-index` | `10` | Frame index in H.264 video (PMC13092701 uses 10) |
| `--limit` | None | Cap at N dishes for quick testing |
| `--out` | `results/eval_results.json` | Output path |
| `--data-dir` | `./data/nutrition5k` | Nutrition5k root |

---

## Early Results

> Full 507-dish run in progress. These are from the 5-dish sanity run.

| Model | Params | Precision | Recall | F1 | Mode |
|---|---|---|---|---|---|
| **gemma4** (local, Q4_K_M) | 8B | **0.6915** | 0.6027 | 0.6367 | overhead |
| Llama 3.2-90B-Vision¹ | 90B | 0.6626–0.6967 | — | — | side cameras A–D |

¹ From PMC13092701 (April 2026, peer-reviewed). Evaluated on side-angle cameras using their custom LCS+synonym metric. Direct comparison requires replicating exact prompt and camera perspective.

**Initial observation:** An 8B local model quantized to Q4_K_M appears competitive with a 90B cloud model on this task. This may reflect the task's sensitivity to prompt design more than raw model scale.

---

## Output JSON Schema

```json
{
  "model": "gemma4:latest",
  "mode": "overhead",
  "n_dishes": 507,
  "n_samples": 507,
  "n_skipped": 0,
  "aggregate": {
    "precision": 0.xx,
    "recall": 0.xx,
    "f1": 0.xx,
    "n": 507
  },
  "per_camera": {
    "overhead": { "precision": 0.xx, "recall": 0.xx, "f1": 0.xx, "n": 507 }
  },
  "samples": [
    {
      "dish_id": "dish_1565035746",
      "camera": "overhead",
      "predicted": ["chicken", "rice", "broccoli"],
      "ground_truth": ["chicken breast", "white rice", "broccoli florets"],
      "scores": { "precision": 0.92, "recall": 0.88, "f1": 0.90 },
      "raw_response": "..."
    }
  ]
}
```

---

## Prompt Used

```
You are a food recognition assistant.
Look at this image of a meal and list every distinct food ingredient you can see.
Return ONLY a JSON array of ingredient name strings — no quantities, no units, no explanation.
Example: ["chicken", "rice", "broccoli", "carrots"]
```

Low temperature (0.1) for consistent outputs.

---

## Extending to Other Models

The runner calls Ollama's `/api/chat` endpoint. To test Gemini 2.5 Flash or Llama 3.2-90B-Vision via cloud APIs, replace `call_ollama()` in `run_eval.py` with your provider's client — the rest of the pipeline (metric, parsing, output) is model-agnostic.

---

## What's Not Solved

1. **Synonym lists are incomplete.** The paper's synonym lists are not published. Our `synonyms.py` covers ~100 common equivalences but will undercount semantic matches, slightly depressing recall.

2. **Camera perspective matters.** PMC13092701 reports per-camera scores (A/B/C/D). Overhead images give a different view — our scores are not directly apples-to-apples with their side-camera numbers.

3. **Prompt sensitivity.** The exact prompts used in PMC13092701 are not disclosed. Different prompt wording can swing precision by several points on this task.

---

## References

- **PMC13092701** — VLM food ingredient recognition study, April 2026. [Link](https://pmc.ncbi.nlm.nih.gov/articles/PMC13092701/)
- **Nutrition5k** — Thames et al., CVPR 2021. [arXiv](https://arxiv.org/abs/2103.03375) · [Dataset](https://github.com/google-research-datasets/Nutrition5k)
- **FoodNExTDB** — CVPR 2025 Workshop benchmark (Gemini 2.0 Flash: avg EWR 70.16). [arXiv](https://arxiv.org/abs/2504.06925)
- **January Food Benchmark (JFB)** — arXiv August 2025, Hungarian matching + embedding cosine similarity. [arXiv](https://arxiv.org/abs/2508.09966)
