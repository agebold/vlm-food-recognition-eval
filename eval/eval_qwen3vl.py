#!/usr/bin/env python3
"""Evaluator: Qwen3-VL 235B (vision) on Nutrition5k.

Invoked by bare model id (on-demand; no inference profile exists for it).
Third-party models often have LOWER rpm quotas, so we throttle harder.

  python3 eval_qwen3vl.py            # full 507-dish test split
  python3 eval_qwen3vl.py --limit 3  # smoke test
"""
import argparse
from bedrock_core import run_eval

MODEL_ID = "qwen.qwen3-vl-235b-a22b"
REGION = "us-east-2"
DELAY = 1.0          # conservative — newer 3rd-party model, unknown quota
MAX_TOKENS = 512     # VL model may emit a little reasoning before the array

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="results/bedrock_qwen3vl_overhead.json")
    args = ap.parse_args()
    run_eval(MODEL_ID, REGION, args.out, delay=DELAY, max_tokens=MAX_TOKENS,
             limit=args.limit, label="qwen3vl")
