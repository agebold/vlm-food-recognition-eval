#!/usr/bin/env python3
"""Evaluator: Moonshot Kimi K2.5 (vision) on Nutrition5k.

Invoked by bare model id (on-demand; no inference profile exists for it).
NOTE: kimi-k2-thinking is text-only — the vision model is kimi-k2.5.
Third-party models often have LOWER rpm quotas, so we throttle harder.

  python3 eval_kimi25.py            # full 507-dish test split
  python3 eval_kimi25.py --limit 3  # smoke test
"""
import argparse
from bedrock_core import run_eval

MODEL_ID = "moonshotai.kimi-k2.5"
REGION = "us-east-2"
DELAY = 1.0          # conservative — newer 3rd-party model, unknown quota
MAX_TOKENS = 512

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="results/bedrock_kimi25_overhead.json")
    args = ap.parse_args()
    run_eval(MODEL_ID, REGION, args.out, delay=DELAY, max_tokens=MAX_TOKENS,
             limit=args.limit, label="kimi25")
