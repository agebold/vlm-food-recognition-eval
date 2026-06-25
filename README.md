# VLM Food Recognition Evaluation on Nutrition5k

Reproducible evaluation harness for photo-based food ingredient recognition using vision-language models (VLMs), benchmarked against the [Nutrition5k](https://github.com/google-research-datasets/Nutrition5k) dataset using the custom metric from [PMC13092701](https://pmc.ncbi.nlm.nih.gov/articles/PMC13092701/).

---

## Results

Full 507-dish evaluation on the Nutrition5k overhead test split. All models use the same prompt, same test images, same metric.

### Prediction Accuracy (honest metric)

**Prediction accuracy** = fraction of named ingredients that were actually in the dish. This is the only fair metric when you cannot see invisible ingredients (olive oil, salt, pepper) in a photo.

| Model | Provider | F1 | Precision | Pred Accuracy | Correct/Dish | Avg Predictions | Zero-Correct | All-Correct |
|---|---|---|---|---|---|---|---|---|
| **Claude Opus 4.8** | Anthropic API | **0.677** | 0.714 | 82.7% | 3.30 | 4.1 | 4.7% | 37.1% |
| Llama4 Scout 17B | AWS Bedrock | 0.655 | 0.711 | 83.8% | 2.71 | 3.2 | 5.1% | 37.7% |
| Llama4 Maverick 17B | AWS Bedrock | 0.646 | 0.703 | **84.2%** | 2.61 | 3.1 | 5.3% | 39.5% |
| Nova Lite | AWS Bedrock | 0.649 | 0.712 | 82.5% | 2.57 | 3.1 | 6.7% | **39.9%** |
| Nova Pro | AWS Bedrock | 0.644 | 0.700 | 81.0% | 2.66 | 3.3 | 7.1% | 37.1% |

### What these numbers actually mean

**Claude's F1 lead is an artifact of verbosity, not accuracy.** Claude names 4.1 ingredients per dish; Maverick names 3.1. Claude gets more right in absolute terms (3.3 vs 2.6) but its *rate* of being correct (82.7%) is lower than Maverick's (84.2%). When you ask "for each thing the model named, was it actually there?" — Maverick wins.

The F1 metric rewards predicting more things (higher recall numerator) even when those things aren't visible in the photo. This inflates Claude's score but doesn't reflect real-world usefulness.

**Cost comparison (per 507 dishes):**
- Claude Opus 4.8 (vision): ~$0.80–1.20
- Llama4 Maverick (Bedrock): ~$0.02–0.05
- Nova Lite (Bedrock): ~$0.003–0.005

At 10–40x cheaper with *higher* prediction accuracy, Maverick and Nova Lite are the practical choices for this task.

---

## Why "Recall" Is a Misleading Metric Here

The PMC13092701 paper reports recall, and we implement it faithfully — but treating it as a real performance signal is wrong.

Nutrition5k's ground truth includes ingredients that are **physically invisible in overhead photos**:

| Category | Top examples | Why a model can't see them |
|---|---|---|
| Oils / fats | olive oil, butter | Absorbed into food, no visual presence |
| Dissolved seasonings | salt, pepper | Dissolved or tiny specks below image resolution |
| Aromatics cooked in | garlic, shallots, onions | Minced/diced and hidden inside other foods |
| Acid/liquid components | vinegar, lemon juice | Clear liquids poured on and invisible |

The top missed ingredients across all 5 models confirm this — the leaderboard of "misses" is dominated by invisible ingredients:

| Ingredient | Times missed (across 5 models × 507 dishes) |
|---|---|
| olive oil | 877 |
| salt | 595 |
| pepper | 363 |
| garlic | 344 |
| mustard | 312 |
| vinegar | 277 |
| lemon juice | 206 |

A model that correctly answers "I see chicken, rice, broccoli, and cherry tomatoes" will get low recall because GT also includes olive oil, salt, and garlic that are invisible. Recall punishes perfect visual predictions for not hallucinating invisible ingredients. **Don't use recall as a standalone metric on this dataset.**

---

## Analysis Plots

All plots are in [eval/plots/](eval/plots/).

### Plot 1 — Prediction accuracy per model
![Prediction accuracy](eval/plots/01_prediction_accuracy.png)

When a model names an ingredient, how often is it actually in the dish? Maverick leads at 84.2%.

### Plot 2 — Correct ingredients per dish (distribution)
![Correct per dish](eval/plots/02_correct_per_dish_dist.png)

Distribution of correct ingredient counts per dish for each model. Claude's higher median reflects more predictions, not higher accuracy.

### Plot 3 — Top missed ingredients across all models
![Missed ingredients](eval/plots/03_missed_ingredients.png)

The top missed ingredients are dominated by invisible or hard-to-see items. These are a fundamental benchmark limitation — no VLM can see dissolved salt or absorbed olive oil.

### Plot 4 — Hallucinated ingredients per model
![Hallucinated](eval/plots/04_hallucinated.png)

Ingredients each model predicted that weren't in ground truth. Common patterns: generic terms ("seasoning", "sauce") and confident wrong identifications.

### Plot 5 — F1 by dish complexity
![F1 by complexity](eval/plots/05_f1_by_complexity.png)

All models degrade sharply on complex dishes (9+ GT ingredients). Simple dishes (1–3 GT ingredients) get high F1 because naming one or two visible items matches most of GT.

### Plot 6 — Per-dish prediction accuracy histogram
![Pred accuracy hist](eval/plots/06_pred_accuracy_hist.png)

Distribution of per-dish prediction accuracy (correct/predicted) across all 507 dishes. Most dishes cluster above 80%, but there's a long tail of difficult cases near 0%.

---

## Dataset: Nutrition5k

- **5,006 plates** of real cafeteria food with per-ingredient mass, calorie, fat, carb, protein annotations
- **507 overhead RGB test images** from the `depth_test_ids.txt` split — pre-extracted PNGs, no video decoding needed
- **Average GT ingredients per dish:** 7.1 — but many are invisible condiments and seasonings
- **Download:** public GCS bucket, no authentication required

```bash
# Download overhead test images (~194 MB)
mkdir -p eval/data/nutrition5k/imagery/realsense_overhead
while IFS= read -r dish_id; do
  gsutil cp "gs://nutrition5k_dataset/nutrition5k_dataset/imagery/realsense_overhead/$dish_id/rgb.png" \
            "eval/data/nutrition5k/imagery/realsense_overhead/$dish_id.png"
done < eval/data/nutrition5k/dish_ids/splits/depth_test_ids.txt
```

---

## Metric: PMC13092701 Equations 1–6

Ingredient matching is **soft** — partial name matches contribute fractionally via normalized LCS, and known synonyms match perfectly.

```
Sim(a, b)     = max( StrMatch(a, b),  SemMatch(a, b) )
StrMatch(a,b) = 2 × |LCS(a,b)| / (|a| + |b|)           # Eq 5 — normalized LCS
SemMatch(a,b) = 1.0 if b ∈ Var(a) or a ∈ Var(b) else 0 # Eq 6 — synonym lookup

Precision     = Σ p∈P  max t∈T  Sim(p,t)  /  |P|        # Eq 1
Recall        = Σ t∈T  max p∈P  Sim(t,p)  /  |T|        # Eq 2
F1            = 2 × P × R / (P + R)                      # Eq 3
```

Where `P` = predicted ingredient list (from VLM), `T` = ground-truth ingredient list.

**Implementation:** [eval/metric.py](eval/metric.py) — macro-averaged across dishes.

**Synonym coverage:** [eval/synonyms.py](eval/synonyms.py) — 100+ bidirectional equivalences (e.g. "capsicum" ↔ "bell pepper", "courgette" ↔ "zucchini"). The paper's synonym list is not published; ours covers common equivalences but may undercount some matches.

---

## Prompt Used

All models receive the same prompt at temperature 0.1:

```
You are a food recognition assistant.
Look at this image of a meal and list every distinct food ingredient you can see.
Return ONLY a JSON array of ingredient name strings — no quantities, no units, no explanation.
Example: ["chicken", "rice", "broccoli", "carrots"]
```

---

## Project Structure

```
eval/
├── metric.py                # PMC13092701 Eq 1-6 (LCS + synonym P/R/F1)
├── synonyms.py              # 100+ food synonym groups → bidirectional lookup
├── parse_nutrition5k.py     # load GT CSVs, locate image files
├── run_eval.py              # Ollama / OpenAI-compat API runner
├── run_eval_mlx.py          # Apple Silicon MLX runner (Qwen3-VL-8B bf16)
├── run_eval_bedrock.py      # AWS Bedrock runner (Llama4, Nova, etc.)
├── plot_analysis.py         # generate all 6 analysis plots
├── analyze_failures.py      # per-dish failure categorization
├── plots/                   # generated PNG plots
├── results/                 # eval result JSONs (per-dish scores)
└── data/nutrition5k/        # dataset (gitignored)
```

---

## Running the Evals

### AWS Bedrock (fastest — 507 dishes in ~20 min)

```bash
cd eval
pip install boto3 tqdm
aws sso login --profile agebold-ds

# Llama4 Maverick
python3 run_eval_bedrock.py --model us.meta.llama4-maverick-17b-instruct-v1:0 --delay 0.5

# Nova Lite
python3 run_eval_bedrock.py --model us.amazon.nova-lite-v1:0 --delay 0.3

# Any Bedrock model with vision support
python3 run_eval_bedrock.py --model <model-id> --out results/mymodel.json
```

Runs resume automatically if interrupted (incremental save after each dish).

### Apple Silicon / MLX (local, unquantized)

```bash
cd eval
pip install mlx-vlm tqdm

# Convert bf16 model from HuggingFace (one-time, ~16 GB)
mlx_vlm.convert --hf-path Qwen/Qwen3-VL-8B-Instruct \
                --mlx-path ./models/Qwen3-VL-8B-Instruct-bf16 \
                --dtype bfloat16

# Run (reloads model every 100 dishes to flush MLX memory state)
python3.12 run_eval_mlx.py \
    --model ./models/Qwen3-VL-8B-Instruct-bf16 \
    --out results/qwen3vl_8b_bf16_full.json \
    --reload-every 100
```

**Note on MLX memory:** After ~100 consecutive inference calls, the Metal allocator accumulates state and model outputs degrade to empty responses. The `--reload-every` flag deletes the old model before reloading to avoid double-loading 16 GB on 24 GB unified memory.

### Generate Plots

```bash
cd eval
python3 plot_analysis.py
# Saves 6 plots to eval/plots/
```

### Failure Analysis

```bash
cd eval
python3 analyze_failures.py results/bedrock_llama4_maverick_full.json --top 20
```

---

## Benchmark Limitations

**1. Ground truth includes invisible ingredients.** Nutrition5k was designed for nutritional estimation, not visual recognition. Many GT ingredients (olive oil, salt, vinegar, garlic) are physically impossible to identify from photos. The top missed ingredients across all models are exactly these invisible items. Any eval on this dataset that doesn't account for this will understate model performance.

**2. Recall is not a reliable metric here.** A model that correctly names every visible ingredient will still get low recall because GT is padded with invisible seasonings. Use prediction accuracy (correct/predicted) as the primary metric.

**3. Synonym lists are incomplete.** The PMC13092701 paper's synonym lists are unpublished. Our lists cover common equivalences but will miss some soft matches, slightly deflating scores uniformly across models.

**4. Camera perspective.** PMC13092701 evaluated side-angle cameras (A–D). This eval uses overhead PNGs only — per-camera comparison with paper results is not direct.

**5. PMC13092701 Llama 3.2-90B claim (F1=0.6626–0.6967).** This model is available in Bedrock but must be re-enabled in the AWS console (marked Legacy after 30 days inactivity). Direct comparison with the paper's reported best model is possible but not yet completed.

---

## References

- **PMC13092701** — VLM food ingredient recognition study, April 2026. [Link](https://pmc.ncbi.nlm.nih.gov/articles/PMC13092701/)
- **Nutrition5k** — Thames et al., CVPR 2021. [arXiv](https://arxiv.org/abs/2103.03375) · [Dataset](https://github.com/google-research-datasets/Nutrition5k)
- **Llama 4 Scout/Maverick** — Meta AI, April 2025. [AWS Bedrock](https://aws.amazon.com/bedrock/)
- **Amazon Nova** — Amazon, 2024. [AWS Bedrock](https://aws.amazon.com/bedrock/nova/)
