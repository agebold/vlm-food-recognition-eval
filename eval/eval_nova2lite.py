#!/usr/bin/env python3
"""Evaluator: Amazon Nova 2 Lite (vision) on Nutrition5k.

  python3 eval_nova2lite.py            # full 507-dish test split
  python3 eval_nova2lite.py --limit 3  # smoke test
"""
import argparse
from bedrock_core import run_eval

MODEL_ID = "us.amazon.nova-2-lite-v1:0"
REGION = "us-east-2"
DELAY = 0.3          # Nova Lite is fast and has generous quotas
MAX_TOKENS = 256

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="results/bedrock_nova2lite_overhead.json")
    args = ap.parse_args()
    run_eval(MODEL_ID, REGION, args.out, delay=DELAY, max_tokens=MAX_TOKENS,
             limit=args.limit, label="nova2lite")
