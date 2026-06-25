"""
Evaluation runner — Nutrition5k, PMC13092701 metric.
Uses HuggingFace transformers + PyTorch MPS directly (no Ollama).

Features:
  - Loads model on CPU (unified memory on Apple Silicon) then runs inference on MPS
  - Incremental save after every dish — safe to kill and resume
  - Configurable delay between calls for thermal/memory headroom
  - Auto-resumes from partial results file

Usage:
  python3 run_eval_hf.py --limit 5                                   # sanity check
  python3 run_eval_hf.py --model Qwen/Qwen3-VL-8B-Instruct          # full run
  python3 run_eval_hf.py --model Qwen/Qwen3-VL-30B-A3B-Instruct --delay 2
"""

from __future__ import annotations
import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from tqdm import tqdm

from metric import score, aggregate
from parse_nutrition5k import load_dishes, load_split_ids, find_overhead_rgb

PROMPT = (
    "You are a food recognition assistant. "
    "Look at this image of a meal and list every distinct food ingredient you can see. "
    "Return ONLY a JSON array of ingredient name strings — no quantities, no units, no explanation. "
    'Example: ["chicken", "rice", "broccoli", "carrots"]'
)

COMPUTE_DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def setup_logger(log_path: Path) -> logging.Logger:
    log = logging.getLogger("eval")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)
    return log


def load_model(model_id: str, log: logging.Logger):
    log.info(f"Loading {model_id} ...")
    t0 = time.time()

    # Step 1: load on CPU — avoids caching_allocator_warmup trying to
    # pre-allocate the full model as a single 15GB Metal buffer (Metal rejects it)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    # Step 2: move to MPS one tensor at a time — each layer is small enough for Metal
    if COMPUTE_DEVICE == "mps":
        model = model.to("mps")
    model.eval()

    processor = AutoProcessor.from_pretrained(model_id)
    log.info(f"Loaded in {time.time()-t0:.1f}s | inference device: {COMPUTE_DEVICE}")
    return model, processor


def call_model(model, processor, image_path: Path) -> str:
    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
    ).to(COMPUTE_DEVICE)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )

    input_len = inputs["input_ids"].shape[1]
    generated = output_ids[0][input_len:]
    return processor.decode(generated, skip_special_tokens=True)


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


def run(data_dir: str, model_id: str, limit: int | None, out_path: str,
        id_file: str | None, delay: float) -> None:

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    log = setup_logger(out.with_suffix(".log"))

    # --- Resume from partial results ---
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

    model, processor = load_model(model_id, log)

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
    log.info(f"Total: {len(test_ids)} | Pending: {len(pending)} | Device: {COMPUTE_DEVICE}")

    samples = list(existing_samples)
    skipped = 0

    for dish_id in tqdm(pending, desc="Evaluating"):
        dish = dishes.get(dish_id)
        if not dish or not dish.ingredients:
            skipped += 1
            log.warning(f"SKIP {dish_id}: no dish data")
            continue

        frame = find_overhead_rgb(data_dir, dish_id)
        if frame is None:
            skipped += 1
            log.warning(f"SKIP {dish_id}: image not found")
            continue

        t0 = time.time()
        try:
            raw = call_model(model, processor, frame)
        except Exception as e:
            skipped += 1
            log.error(f"SKIP {dish_id}: {e}")
            continue

        elapsed = time.time() - t0
        predicted = parse_ingredients(raw)
        s = score(predicted, dish.ingredients)

        sample = {
            "dish_id": dish_id,
            "camera": "overhead",
            "predicted": predicted,
            "ground_truth": dish.ingredients,
            "scores": s,
            "raw_response": raw,
        }
        samples.append(sample)
        log.info(f"{dish_id} | P={s['precision']:.3f} R={s['recall']:.3f} F1={s['f1']:.3f} | {elapsed:.1f}s")

        # Incremental save after every dish
        _save(out, model_id, test_ids, samples, skipped)

        if delay > 0:
            time.sleep(delay)

    _save(out, model_id, test_ids, samples, skipped)
    agg = aggregate([s["scores"] for s in samples])
    log.info(f"\nDONE | P={agg['precision']:.4f} R={agg['recall']:.4f} F1={agg['f1']:.4f} | n={len(samples)}")


def _save(out: Path, model_id: str, test_ids: list, samples: list, skipped: int):
    agg = aggregate([s["scores"] for s in samples]) if samples else {"precision": 0, "recall": 0, "f1": 0, "n": 0}
    result = {
        "model": model_id,
        "provider": "hf_mps",
        "mode": "overhead",
        "cameras": ["overhead"],
        "n_dishes": len(test_ids),
        "n_samples": len(samples),
        "n_skipped": skipped,
        "aggregate": agg,
        "per_camera": {"overhead": agg},
        "samples": samples,
    }
    out.write_text(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--data-dir", default="./data/nutrition5k")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--overhead", action="store_true", default=True)
    parser.add_argument("--id-file", default=None)
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Seconds to wait between inferences (thermal headroom)")
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
    )


if __name__ == "__main__":
    main()
