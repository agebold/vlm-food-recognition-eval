#!/usr/bin/env python3
"""Evaluator: Anthropic Claude Sonnet 4.6 (vision) on Nutrition5k.

  python3 eval_sonnet46.py            # full 507-dish test split
  python3 eval_sonnet46.py --limit 3  # smoke test
"""
import argparse
from bedrock_core import run_eval

MODEL_ID = "us.anthropic.claude-sonnet-4-6"
REGION = "us-east-2"
DELAY = 0.5          # Sonnet is heavier; slightly more headroom
MAX_TOKENS = 256

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="results/bedrock_sonnet46_overhead.json")
    args = ap.parse_args()
    run_eval(MODEL_ID, REGION, args.out, delay=DELAY, max_tokens=MAX_TOKENS,
             limit=args.limit, label="sonnet46")
