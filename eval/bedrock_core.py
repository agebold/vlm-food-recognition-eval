"""
Shared core for Bedrock food-recognition evaluators (Nutrition5k, PMC13092701 metric).

Each per-model evaluator (eval_haiku45.py, eval_sonnet46.py, ...) just imports
run_eval() and calls it with its own model_id / region / rate-limit settings.
Every model gets its OWN process, OWN output file, and OWN rate-limit bucket —
see the rate-limit notes below.

RATE LIMITS (how Bedrock enforces them):
  Throttling is a Service Quota scoped to  account x region x model.
  Two limits per model: requests/minute (RPM) and tokens/minute (TPM).
  - Each model has an INDEPENDENT bucket -> running 5 different models in
    parallel never contend with each other.
  - The SAME model called concurrently DOES share one bucket.
  So we run one process per model, each self-throttling with `delay` between
  calls plus exponential backoff on ThrottlingException.

Requirements:
  pip3 install boto3 tqdm
  aws sso login --sso-session agebold-sso
"""

from __future__ import annotations
import json
import logging
import re
import sys
import time
from pathlib import Path

import boto3
from botocore.config import Config
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

DEFAULT_PROFILE = "agebold-ds"


def setup_logger(name: str, log_path: Path) -> logging.Logger:
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)
    return log


def call_bedrock(client, model_id: str, image_bytes: bytes, max_tokens: int,
                 temperature: float, log: logging.Logger, max_retries: int = 6) -> str:
    """Single Converse call with exponential backoff on throttling/transient errors."""
    message = {
        "role": "user",
        "content": [
            {"image": {"format": "png", "source": {"bytes": image_bytes}}},
            {"text": PROMPT},
        ],
    }
    inference = {"maxTokens": max_tokens, "temperature": temperature}
    for attempt in range(max_retries):
        try:
            resp = client.converse(
                modelId=model_id,
                messages=[message],
                inferenceConfig=inference,
            )
            return resp["output"]["message"]["content"][0]["text"]
        except ClientError as e:
            code = e.response["Error"]["Code"]
            # Throttling / transient server errors -> back off and retry.
            if code in ("ThrottlingException", "TooManyRequestsException",
                        "ServiceUnavailableException", "ModelTimeoutException",
                        "InternalServerException"):
                if attempt == max_retries - 1:
                    raise
                # 5, 10, 20, 40, 80... seconds (capped), honoring the per-model bucket.
                wait = min(2 ** attempt * 5, 90)
                log.warning(f"{code} on {model_id} — backoff {wait}s (attempt {attempt + 1})")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError(f"Max retries exceeded for {model_id}")


def parse_ingredients(raw: str) -> list[str]:
    """Pull a list of ingredient strings out of the model's raw text."""
    raw = re.sub(r"```json\s*|\s*```", "", raw).strip()
    match = re.search(r"\[.*?\]", raw, re.DOTALL)
    if match:
        try:
            items = json.loads(match.group())
            if isinstance(items, list):
                return [str(i).lower().strip() for i in items if str(i).strip()]
        except json.JSONDecodeError:
            pass
    # Fallback: line/comma split, strip list markers.
    cleaned = []
    for line in re.split(r"[\n,]", raw):
        line = re.sub(r"^[\s\-\*\d\.\)]+", "", line).strip().lower()
        if line and len(line) < 60:
            cleaned.append(line)
    return cleaned


def _save(out: Path, model_id: str, region: str, test_ids: list, samples: list, skipped: int):
    agg = aggregate([s["scores"] for s in samples]) if samples else {
        "precision": 0, "recall": 0, "f1": 0, "n": 0}
    out.write_text(json.dumps({
        "model": model_id,
        "provider": "bedrock",
        "region": region,
        "mode": "overhead",
        "cameras": ["overhead"],
        "n_dishes": len(test_ids),
        "n_samples": len(samples),
        "n_skipped": skipped,
        "aggregate": agg,
        "per_camera": {"overhead": agg},
        "samples": samples,
    }, indent=2))


def run_eval(
    model_id: str,
    region: str,
    out_path: str,
    delay: float = 0.5,
    max_tokens: int = 256,
    temperature: float = 0.1,
    limit: int | None = None,
    id_file: str | None = None,
    data_dir: str = "./data/nutrition5k",
    profile: str = DEFAULT_PROFILE,
    label: str | None = None,
) -> dict:
    """
    Evaluate one Bedrock model on the Nutrition5k overhead test split.
    Resumes from a partial out_path if present. Returns the aggregate dict.
    """
    label = label or model_id.split(".")[-1]
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    log = setup_logger(f"eval_{label}", out.with_suffix(".log"))

    # Resume from partial results.
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

    # Generous client-side timeout + a couple of low-level retries on top of ours.
    cfg = Config(read_timeout=120, connect_timeout=15,
                 retries={"max_attempts": 2, "mode": "standard"})
    session = boto3.Session(profile_name=profile)
    client = session.client("bedrock-runtime", region_name=region, config=cfg)
    log.info(f"Bedrock ready | model={model_id} | region={region} | delay={delay}s | max_tokens={max_tokens}")

    data_path = Path(data_dir)
    dishes = load_dishes(data_path)

    if id_file:
        test_ids = [i.strip() for i in Path(id_file).read_text().splitlines() if i.strip()]
    else:
        test_ids = load_split_ids(data_path, split="test", prefix="depth")
    if limit:
        test_ids = test_ids[:limit]

    pending = [tid for tid in test_ids if tid not in done_ids]
    log.info(f"Total: {len(test_ids)} | Pending: {len(pending)}")

    samples = list(existing_samples)
    skipped = 0

    for dish_id in tqdm(pending, desc=label):
        dish = dishes.get(dish_id)
        if not dish or not dish.ingredients:
            skipped += 1
            continue

        frame = find_overhead_rgb(data_path, dish_id)
        if frame is None:
            skipped += 1
            log.warning(f"SKIP {dish_id}: image not found")
            continue

        image_bytes = frame.read_bytes()
        t0 = time.time()
        try:
            raw = call_bedrock(client, model_id, image_bytes, max_tokens, temperature, log)
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
        _save(out, model_id, region, test_ids, samples, skipped)

        if delay > 0:
            time.sleep(delay)

    _save(out, model_id, region, test_ids, samples, skipped)
    agg = aggregate([s["scores"] for s in samples])
    log.info(f"DONE {label} | P={agg['precision']:.4f} R={agg['recall']:.4f} "
             f"F1={agg['f1']:.4f} | n={len(samples)} skipped={skipped}")
    return agg
