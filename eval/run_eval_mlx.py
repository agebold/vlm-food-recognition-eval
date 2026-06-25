"""
Evaluation runner — Nutrition5k, PMC13092701 metric.
Uses mlx-vlm for fast Apple Silicon inference (no Ollama, no PyTorch MPS hacks).

Features:
  - Native MLX inference — loads and runs in seconds, not minutes
  - Incremental save after every dish — kill and resume safely
  - Configurable delay for thermal headroom

Usage:
  python3 run_eval_mlx.py --limit 5                              # sanity check
  python3 run_eval_mlx.py                                        # full 507-dish run (8B 4bit)
  python3 run_eval_mlx.py --model mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit
"""

from __future__ import annotations
import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

from tqdm import tqdm
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

from metric import score, aggregate
from parse_nutrition5k import load_dishes, load_split_ids, find_overhead_rgb

PROMPT = (
    "You are a food recognition assistant. "
    "Look at this image of a meal and list every distinct food ingredient you can see. "
    "Return ONLY a JSON array of ingredient name strings — no quantities, no units, no explanation. "
    'Example: ["chicken", "rice", "broccoli", "carrots"]'
)


def setup_logger(log_path: Path) -> logging.Logger:
    log = logging.getLogger("eval_mlx")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)
    return log


def call_model(model, processor, config, image_path: Path) -> str:
    formatted = apply_chat_template(processor, config, PROMPT, num_images=1)
    output = generate(
        model,
        processor,
        image=str(image_path),
        prompt=formatted,
        max_tokens=256,
        temperature=0.1,
        repetition_penalty=1.2,
        verbose=False,
    )
    text = output.text if hasattr(output, "text") else str(output)
    # Detect repetition garbage (>20 same chars in a row)
    import re as _re
    if _re.search(r'(.)\1{20,}', text):
        return "[]"
    return text


def parse_ingredients(raw: str) -> list[str]:
    match = re.search(r"\[.*?\]", raw, re.DOTALL)
    if match:
        try:
            items = json.loads(match.group())
            if isinstance(items, list):
                return [str(i).lower().strip() for i in items if str(i).strip()]
        except json.JSONDecodeError:
            pass
    lines = re.split(r"[\n,]", raw)
    cleaned = []
    for line in lines:
        line = re.sub(r"^[\s\-\*\d\.\)]+", "", line).strip().lower()
        if line and len(line) < 60:
            cleaned.append(line)
    return cleaned


def load_model_fresh(model_id: str, log: logging.Logger, old_model=None):
    import gc
    import mlx.core as mx
    # Explicitly release the old model BEFORE loading the new one to avoid
    # holding two 16GB copies in unified memory simultaneously (would OOM on 24GB)
    if old_model is not None:
        del old_model
        gc.collect()
        mx.clear_cache()
    log.info(f"(Re)loading {model_id} ...")
    t0 = time.time()
    model, processor = load(model_id)
    config = load_config(model_id)
    mx.clear_cache()
    gc.collect()
    log.info(f"Loaded in {time.time()-t0:.1f}s")
    return model, processor, config


def run(data_dir: str, model_id: str, limit: int | None, out_path: str,
        id_file: str | None, delay: float, reload_every: int = 100) -> None:

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    log = setup_logger(out.with_suffix(".log"))

    # Resume from partial results
    done_ids: set[str] = set()
    existing_samples: list[dict] = []
    if out.exists():
        try:
            prev = json.loads(out.read_text())
            existing_samples = prev.get("samples", [])
            done_ids = {s["dish_id"] for s in existing_samples}
            log.info(f"Resuming — {len(done_ids)} dishes already done")
        except Exception:
            log.warning("Could not parse existing results, starting fresh")

    model, processor, config = load_model_fresh(model_id, log)

    data_dir = Path(data_dir)
    dishes = load_dishes(data_dir)

    if id_file:
        test_ids = Path(id_file).read_text().splitlines()
        test_ids = [i.strip() for i in test_ids if i.strip()]
    else:
        test_ids = load_split_ids(data_dir, split="test", prefix="depth")
    if limit:
        test_ids = test_ids[:limit]

    pending = [tid for tid in test_ids if tid not in done_ids]
    log.info(f"Total: {len(test_ids)} | Pending: {len(pending)}")

    samples = list(existing_samples)
    skipped = 0
    dishes_since_reload = 0

    for dish_id in tqdm(pending, desc="Evaluating"):
        # Periodically reload model to flush accumulated MLX state.
        # Pass old model so it's deleted BEFORE the new one is loaded (avoids 2x RAM usage).
        if dishes_since_reload >= reload_every:
            log.info(f"Reloading model after {reload_every} dishes to flush MLX state ...")
            model, processor, config = load_model_fresh(model_id, log, old_model=model)
            dishes_since_reload = 0

        dish = dishes.get(dish_id)
        if not dish or not dish.ingredients:
            skipped += 1
            continue

        frame = find_overhead_rgb(data_dir, dish_id)
        if frame is None:
            skipped += 1
            log.warning(f"SKIP {dish_id}: image not found")
            continue

        t0 = time.time()
        try:
            raw = call_model(model, processor, config, frame)
        except Exception as e:
            skipped += 1
            log.error(f"SKIP {dish_id}: {e}")
            continue

        elapsed = time.time() - t0
        predicted = parse_ingredients(raw)

        # Skip empty outputs — treat as inference failure, don't pollute results
        if not predicted:
            skipped += 1
            log.warning(f"SKIP {dish_id}: empty prediction (raw={repr(raw[:80])})")
            dishes_since_reload += 1
            continue

        s = score(predicted, dish.ingredients)

        samples.append({
            "dish_id": dish_id,
            "camera": "overhead",
            "predicted": predicted,
            "ground_truth": dish.ingredients,
            "scores": s,
            "raw_response": raw,
        })
        log.info(f"{dish_id} | P={s['precision']:.3f} R={s['recall']:.3f} F1={s['f1']:.3f} | {elapsed:.1f}s")
        dishes_since_reload += 1

        _save(out, model_id, test_ids, samples, skipped)

        if delay > 0:
            time.sleep(delay)

    _save(out, model_id, test_ids, samples, skipped)
    agg = aggregate([s["scores"] for s in samples])
    log.info(f"DONE | P={agg['precision']:.4f} R={agg['recall']:.4f} F1={agg['f1']:.4f} | n={len(samples)}")


def _save(out: Path, model_id: str, test_ids: list, samples: list, skipped: int):
    agg = aggregate([s["scores"] for s in samples]) if samples else {"precision": 0, "recall": 0, "f1": 0, "n": 0}
    out.write_text(json.dumps({
        "model": model_id,
        "provider": "mlx",
        "mode": "overhead",
        "cameras": ["overhead"],
        "n_dishes": len(test_ids),
        "n_samples": len(samples),
        "n_skipped": skipped,
        "aggregate": agg,
        "per_camera": {"overhead": agg},
        "samples": samples,
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/Qwen3-VL-8B-Instruct-4bit")
    parser.add_argument("--data-dir", default="./data/nutrition5k")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--id-file", default=None)
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--reload-every", type=int, default=100,
                        help="Reload model every N dishes to flush MLX state accumulation")
    args = parser.parse_args()

    slug = args.model.replace("/", "_").replace("-", "_")
    out_path = args.out or f"results/{slug}_overhead_full.json"

    run(
        data_dir=args.data_dir,
        model_id=args.model,
        limit=args.limit,
        out_path=out_path,
        id_file=args.id_file,
        delay=args.delay,
        reload_every=args.reload_every,
    )


if __name__ == "__main__":
    main()
