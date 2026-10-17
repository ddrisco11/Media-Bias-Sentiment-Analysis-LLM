#!/bin/sh
# Run the pre-registered RL conditions as two sequential queues (two processes share the GPU).
# Each run resumes nothing: a finished run has results/v2/rl/<run>/finished.json and is skipped.
cd "$(dirname "$0")/.." || exit 1
MODEL=Qwen/Qwen2.5-0.5B-Instruct
run() {
  if [ -f "results/v2/rl/$1-s$2/finished.json" ]; then echo "skip $1-s$2"; return; fi
  rm -rf "results/v2/rl/$1-s$2"
  .venv/bin/python -m study.rl --condition "$1" --seed "$2" --model "$MODEL" > "results/v2/rl/logs/$1-s$2.log" 2>&1 || echo "FAILED $1-s$2"
}
mkdir -p results/v2/rl/logs
( run calibrated 1; run calibrated 2; run shuffled 1; run format 2 ) &
( run format 1; run calibrated 3; run pooled 1 ) &
wait
echo "all RL runs finished"
