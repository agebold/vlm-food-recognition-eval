#!/usr/bin/env bash
# Launch all 5 Bedrock evaluators in PARALLEL — one process per model.
# Each model has its own Bedrock rate-limit bucket, so they don't contend.
# Each writes results/bedrock_<model>_overhead.json (+ .log) and resumes if re-run.
#
# Usage:
#   ./run_all_bedrock.sh            # full 507-dish run
#   ./run_all_bedrock.sh --limit 3  # smoke test all 5
#
# Logs stream to results/run_<model>.out. Ctrl-C kills the whole group.

set -uo pipefail
cd "$(dirname "$0")"
export AWS_PROFILE=agebold-ds

ARGS="$*"
mkdir -p results
EVALS=(eval_haiku45 eval_sonnet46 eval_nova2lite eval_qwen3vl eval_kimi25)

pids=()
for e in "${EVALS[@]}"; do
  echo "launching $e $ARGS"
  python3 "$e.py" $ARGS > "results/run_${e}.out" 2>&1 &
  pids+=($!)
done

# Kill all children if this script is interrupted.
trap 'echo "interrupt — killing all"; kill "${pids[@]}" 2>/dev/null' INT TERM

echo "running ${#pids[@]} evaluators in parallel (pids: ${pids[*]})"
echo "tail a model with:  tail -f results/run_eval_haiku45.out"

fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done

echo "all done (fail=$fail). compare with:  python3 compare_bedrock.py"
exit $fail
