"""
Evaluation runner — Nutrition5k test split, PMC13092701 metric.

Sends each dish frame to a local Ollama model (default: gemma4:latest),
parses the predicted ingredient list, and scores with the LCS + synonym metric.

Usage:
  python run_eval.py                                # all default settings
  python run_eval.py --model gemma4:latest          # model override
  python run_eval.py --cameras A                    # single camera only
  python run_eval.py --limit 50                     # first N dishes (quick sanity check)
  python run_eval.py --data-dir /path/to/nutrition5k
  python run_eval.py --out results/gemma4_cam_A.json

Results JSON:
  {
    "model": "gemma4:latest",
    "cameras": ["A"],
    "n_dishes": 501,
    "n_samples": 501,
    "aggregate": {"precision": 0.xx, "recall": 0.xx, "f1": 0.xx, "n": 501},
    "per_camera": {"A": {...}},
    "samples": [{"dish_id": ..., "camera": ..., "predicted": [...], "ground_truth": [...], "scores": {...}}, ...]
  }
"""

from __future__ import annotations
import argparse
import base64
import json
import re
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm

from metric import score, aggregate
from parse_nutrition5k import load_dishes, load_split_ids, find_frame, find_overhead_rgb

OLLAMA_URL = "http://localhost:11434/api/chat"

PROMPT = (
    "You are a food recognition assistant. "
    "Look at this image of a meal and list every distinct food ingredient you can see. "
    "Return ONLY a JSON array of ingredient name strings — no quantities, no units, no explanation. "
    'Example: ["chicken", "rice", "broccoli", "carrots"]'
)


def call_ollama(image_path: Path, model: str, timeout: int = 60) -> str:
    image_b64 = base64.b64encode(image_path.read_bytes()).decode()
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": PROMPT,
                "images": [image_b64],
            }
        ],
        "stream": False,
        "options": {"temperature": 0.1},  # low temp for consistent ingredient lists
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def parse_ingredients(raw: str) -> list[str]:
    """Extract a list of ingredient strings from model output.

    Handles:
      - Clean JSON arrays: ["a", "b"]
      - JSON embedded in prose: 'Here are the ingredients: ["a", "b"] ...'
      - Newline-separated lists (fallback)
    """
    # Try to find a JSON array anywhere in the response
    match = re.search(r"\[.*?\]", raw, re.DOTALL)
    if match:
        try:
            items = json.loads(match.group())
            if isinstance(items, list):
                return [str(i).lower().strip() for i in items if str(i).strip()]
        except json.JSONDecodeError:
            pass

    # Fallback: split on newlines / commas and strip bullets/numbers
    lines = re.split(r"[\n,]", raw)
    cleaned = []
    for line in lines:
        line = re.sub(r"^[\s\-\*\d\.\)]+", "", line).strip().lower()
        if line and len(line) < 60:  # ignore long sentences
            cleaned.append(line)
    return cleaned


def run(
    data_dir: str,
    model: str,
    cameras: list[str],
    frame_index: int,
    overhead: bool,
    limit: int | None,
    out_path: str,
    delay: float,
) -> None:
    data_dir = Path(data_dir)

    print(f"Loading dishes from {data_dir}...")
    dishes = load_dishes(data_dir)
    split = "test"
    split_prefix = "depth" if overhead else "rgb"
    test_ids = load_split_ids(data_dir, split=split, prefix=split_prefix)
    if limit:
        test_ids = test_ids[:limit]

    mode = "overhead rgb.png" if overhead else f"side cameras {cameras}"
    print(f"Test dishes: {len(test_ids)} | Mode: {mode} | Model: {model}")

    samples = []
    per_camera_scores: dict[str, list[dict]] = {"overhead": []} if overhead else {c: [] for c in cameras}
    skipped = 0

    for dish_id in tqdm(test_ids, desc="Evaluating"):
        dish = dishes.get(dish_id)
        if not dish or not dish.ingredients:
            skipped += 1
            continue

        if overhead:
            frame = find_overhead_rgb(data_dir, dish_id)
            cam_label = "overhead"
            frame_list = [(cam_label, frame)]
        else:
            frame_list = [
                (cam, find_frame(data_dir, dish_id, camera=cam, frame_index=frame_index))
                for cam in cameras
            ]

        for cam_label, frame in frame_list:
            if frame is None:
                skipped += 1
                continue

            try:
                raw = call_ollama(frame, model)
            except Exception as e:
                tqdm.write(f"  [SKIP] {dish_id} cam={cam_label}: {e}")
                skipped += 1
                continue

            predicted = parse_ingredients(raw)
            s = score(predicted, dish.ingredients)

            sample = {
                "dish_id": dish_id,
                "camera": cam_label,
                "predicted": predicted,
                "ground_truth": dish.ingredients,
                "scores": s,
                "raw_response": raw,
            }
            samples.append(sample)
            per_camera_scores[cam_label].append(s)

            if delay > 0:
                time.sleep(delay)

    agg = aggregate([s["scores"] for s in samples])
    per_camera_agg = {cam: aggregate(scores) for cam, scores in per_camera_scores.items()}

    result = {
        "model": model,
        "mode": "overhead" if overhead else "side_angles",
        "cameras": ["overhead"] if overhead else cameras,
        "frame_index": frame_index if not overhead else None,
        "n_dishes": len(test_ids),
        "n_samples": len(samples),
        "n_skipped": skipped,
        "aggregate": agg,
        "per_camera": per_camera_agg,
        "samples": samples,
    }

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    print(f"\n{'='*50}")
    print(f"Model:     {model}")
    print(f"Cameras:   {cameras}")
    print(f"Samples:   {len(samples)}  (skipped: {skipped})")
    print(f"Precision: {agg['precision']:.4f}")
    print(f"Recall:    {agg['recall']:.4f}")
    print(f"F1:        {agg['f1']:.4f}")
    print(f"{'='*50}")
    print(f"Results saved to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Nutrition5k VLM evaluation")
    parser.add_argument("--data-dir", default="./data/nutrition5k",
                        help="Path to Nutrition5k dataset root")
    parser.add_argument("--model", default="gemma4:latest",
                        help="Ollama model name")
    parser.add_argument("--cameras", nargs="+", default=["A"],
                        choices=["A", "B", "C", "D"],
                        help="Camera perspectives to evaluate")
    parser.add_argument("--frame-index", type=int, default=10,
                        help="Frame index within each camera video (PMC13092701 uses 10)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to first N dishes (for quick testing)")
    parser.add_argument("--out", default="results/eval_results.json",
                        help="Output JSON file path")
    parser.add_argument("--delay", type=float, default=0.0,
                        help="Seconds to wait between API calls (for rate limiting)")
    parser.add_argument("--overhead", action="store_true",
                        help="Use overhead rgb.png images (depth split, 507 dishes) instead of side H.264 videos")
    args = parser.parse_args()

    # Sanity-check Ollama is running
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        resp.raise_for_status()
    except Exception:
        print("ERROR: Ollama not reachable at localhost:11434. Is it running?", file=sys.stderr)
        sys.exit(1)

    run(
        data_dir=args.data_dir,
        model=args.model,
        cameras=args.cameras,
        frame_index=args.frame_index,
        overhead=args.overhead,
        limit=args.limit,
        out_path=args.out,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
