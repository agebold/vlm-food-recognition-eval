"""
Evaluation runner — Nutrition5k, PMC13092701 metric.
Uses AWS Bedrock converse API (supports Llama 4, Nova, Qwen, etc.)

Usage:
  python3 run_eval_bedrock.py --model us.meta.llama4-maverick-17b-instruct-v1:0
  python3 run_eval_bedrock.py --model us.amazon.nova-pro-v1:0
  python3 run_eval_bedrock.py --model us.meta.llama3-2-90b-instruct-v1:0
  python3 run_eval_bedrock.py --limit 20   # quick sanity check

Requirements:
  pip3 install boto3
  aws sso login --profile agebold-ds
"""

from __future__ import annotations
import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from tqdm import tqdm

from metric import score, aggregate
from parse_nutrition5k import load_dishes, load_split_ids, find_overhead_rgb

PROMPT = (
    "You are a food recognition assistant. "
    "Look at this image of a meal and list every distinct food ingredient you can see. "
    "Return ONLY a JSON array of ingredient name strings — no quantities, no units, no explanation. "
    'Example: ["chicken", "rice", "broccoli", "carrots"]'
)

AWS_PROFILE = "agebold-ds"
AWS_REGION = "us-east-1"


def setup_logger(log_path: Path) -> logging.Logger:
    log = logging.getLogger("eval_bedrock")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)
    return log


def call_bedrock(client, model_id: str, image_bytes: bytes, max_retries: int = 3) -> str:
    body = {
        "role": "user",
        "content": [
            {"image": {"format": "png", "source": {"bytes": image_bytes}}},
            {"text": PROMPT},
        ],
    }
    for attempt in range(max_retries):
        try:
            resp = client.converse(
                modelId=model_id,
                messages=[body],
                inferenceConfig={"maxTokens": 256, "temperature": 0.1},
            )
            return resp["output"]["message"]["content"][0]["text"]
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if code == "ThrottlingException":
                wait = 2 ** attempt * 5
                time.sleep(wait)
                continue
            raise
    raise RuntimeError(f"Max retries exceeded for {model_id}")


def parse_ingredients(raw: str) -> list[str]:
    # Strip markdown code fences if present
    raw = re.sub(r"```json\s*|\s*```", "", raw).strip()
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
        id_file: str | None, delay: float, profile: str = AWS_PROFILE) -> None:

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

    session = boto3.Session(profile_name=profile)
    client = session.client("bedrock-runtime", region_name=AWS_REGION)
    log.info(f"Bedrock client ready | model={model_id}")

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

    for dish_id in tqdm(pending, desc="Evaluating"):
        dish = dishes.get(dish_id)
        if not dish or not dish.ingredients:
            skipped += 1
            continue

        frame = find_overhead_rgb(data_dir, dish_id)
        if frame is None:
            skipped += 1
            log.warning(f"SKIP {dish_id}: image not found")
            continue

        image_bytes = frame.read_bytes()
        t0 = time.time()
        try:
            raw = call_bedrock(client, model_id, image_bytes)
        except Exception as e:
            skipped += 1
            log.error(f"SKIP {dish_id}: {e}")
            if delay > 0:
                time.sleep(delay)
            continue

        elapsed = time.time() - t0
        predicted = parse_ingredients(raw)

        if not predicted:
            skipped += 1
            log.warning(f"SKIP {dish_id}: empty prediction (raw={repr(raw[:80])})")
            if delay > 0:
                time.sleep(delay)
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
        "provider": "bedrock",
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
    parser.add_argument("--model", default="us.meta.llama4-maverick-17b-instruct-v1:0")
    parser.add_argument("--data-dir", default="./data/nutrition5k")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--id-file", default=None)
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Seconds between calls (rate limit headroom)")
    parser.add_argument("--profile", default=AWS_PROFILE)
    args = parser.parse_args()

    slug = args.model.replace("/", "_").replace("-", "_").replace(":", "_").replace(".", "_")
    out_path = args.out or f"results/bedrock_{slug}_overhead_full.json"

    run(
        data_dir=args.data_dir,
        model_id=args.model,
        limit=args.limit,
        out_path=out_path,
        id_file=args.id_file,
        delay=args.delay,
        profile=args.profile,
    )


if __name__ == "__main__":
    main()
