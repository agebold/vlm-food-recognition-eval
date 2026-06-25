#!/usr/bin/env python3
"""Evaluator: Anthropic Claude Haiku 4.5 (vision) on Nutrition5k.

  python3 eval_haiku45.py            # full 507-dish test split
  python3 eval_haiku45.py --limit 3  # smoke test
"""
import argparse
from bedrock_core import run_eval

MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
REGION = "us-east-2"
DELAY = 0.4          # ~2.5 req/s; Anthropic on-demand quotas are comfortable
MAX_TOKENS = 256

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="results/bedrock_haiku45_overhead.json")
    args = ap.parse_args()
    run_eval(MODEL_ID, REGION, args.out, delay=DELAY, max_tokens=MAX_TOKENS,
             limit=args.limit, label="haiku45")
