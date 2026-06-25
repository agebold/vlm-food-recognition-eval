"""
Evaluation runner — Nutrition5k depth-split (overhead), PMC13092701 metric.

Uses the Anthropic Claude API instead of local Ollama.

Usage:
  python3 run_eval_claude.py --overhead --limit 5      # quick sanity check
  python3 run_eval_claude.py --overhead                 # full 507-dish eval
  python3 run_eval_claude.py --overhead --model claude-haiku-4-5  # cheaper/faster

Environment:
  ANTHROPIC_API_KEY  — required

Output JSON schema matches run_eval.py exactly, with extra "provider": "claude" field.
"""

from __future__ import annotations
import argparse
import base64
import json
import re
import sys
import time
from pathlib import Path

import anthropic
from tqdm import tqdm

from metric import score, aggregate
from parse_nutrition5k import load_dishes, load_split_ids, find_overhead_rgb

PROMPT = (
    "You are a food recognition assistant. "
    "Look at this image of a meal and list every distinct food ingredient you can see. "
    "Return ONLY a JSON array of ingredient name strings — no quantities, no units, no explanation. "
    'Example: ["chicken", "rice", "broccoli", "carrots"]'
)


def call_claude(
    client: anthropic.Anthropic,
    image_path: Path,
    model: str,
) -> str:
    image_b64 = base64.b64encode(image_path.read_bytes()).decode()
    response = client.messages.create(
        model=model,
        max_tokens=256,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_b64,
                        },
                    },
                    {"type": "text", "text": PROMPT},
                ],
            }
        ],
    )
    return response.content[0].text


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


def run(
    data_dir: str,
    model: str,
    limit: int | None,
    out_path: str,
    delay: float,
) -> None:
    client = anthropic.Anthropic()
    data_dir = Path(data_dir)

    print(f"Loading dishes from {data_dir}...")
    dishes = load_dishes(data_dir)
    test_ids = load_split_ids(data_dir, split="test", prefix="depth")
    if limit:
        test_ids = test_ids[:limit]

    print(f"Test dishes: {len(test_ids)} | Model: {model}")

    samples = []
    overhead_scores: list[dict] = []
    skipped = 0
    errors = 0

    for dish_id in tqdm(test_ids, desc="Evaluating"):
        dish = dishes.get(dish_id)
        if not dish or not dish.ingredients:
            skipped += 1
            continue

        frame = find_overhead_rgb(data_dir, dish_id)
        if frame is None:
            skipped += 1
            continue

        for attempt in range(3):
            try:
                raw = call_claude(client, frame, model)
                break
            except anthropic.RateLimitError:
                wait = 30 * (attempt + 1)
                tqdm.write(f"  [RATE LIMIT] {dish_id} — waiting {wait}s")
                time.sleep(wait)
            except anthropic.APIError as e:
                tqdm.write(f"  [API ERROR] {dish_id}: {e}")
                errors += 1
                skipped += 1
                raw = None
                break
        else:
            tqdm.write(f"  [SKIP] {dish_id}: exceeded retry limit")
            skipped += 1
            raw = None

        if raw is None:
            continue

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
        overhead_scores.append(s)

        if delay > 0:
            time.sleep(delay)

    agg = aggregate([s["scores"] for s in samples])
    per_camera_agg = {"overhead": aggregate(overhead_scores)}

    result = {
        "model": model,
        "provider": "claude",
        "mode": "overhead",
        "cameras": ["overhead"],
        "frame_index": None,
        "n_dishes": len(test_ids),
        "n_samples": len(samples),
        "n_skipped": skipped,
        "n_errors": errors,
        "aggregate": agg,
        "per_camera": per_camera_agg,
        "samples": samples,
    }

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    print(f"\n{'='*50}")
    print(f"Model:     {model}")
    print(f"Samples:   {len(samples)}  (skipped: {skipped}, errors: {errors})")
    print(f"Precision: {agg['precision']:.4f}")
    print(f"Recall:    {agg['recall']:.4f}")
    print(f"F1:        {agg['f1']:.4f}")
    print(f"{'='*50}")
    print(f"Results saved to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Nutrition5k Claude API evaluation")
    parser.add_argument("--data-dir", default="./data/nutrition5k",
                        help="Path to Nutrition5k dataset root")
    parser.add_argument("--model", default="claude-opus-4-8",
                        help="Claude model ID (default: claude-opus-4-8)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap at N dishes for quick testing")
    parser.add_argument("--out", default=None,
                        help="Output JSON path (default: results/{model}_overhead_full.json)")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Seconds between API calls for rate limiting (default: 0.5)")
    parser.add_argument("--overhead", action="store_true", default=True,
                        help="Use overhead rgb.png images (always True for this runner)")
    args = parser.parse_args()

    model_slug = args.model.replace("/", "_").replace(":", "_")
    out_path = args.out or f"results/{model_slug}_overhead_full.json"

    run(
        data_dir=args.data_dir,
        model=args.model,
        limit=args.limit,
        out_path=out_path,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
