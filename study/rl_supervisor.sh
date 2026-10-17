#!/bin/sh
# RL schedule after the v2.6 deviations: final checkpoint at step 200, one heavy GPU job at a time.
# (Two fp32 runs in parallel exhausted the M2 Max's unified memory and stalled on swap.)
# Order of scientific priority, with an interim report (evaluate, score, judge, summarise, push) between runs:
#   format-s1 (already running, launched with --steps 300; stopped after its step-200 checkpoint)
#   calibrated-s1 -> [interim: first calibrated-vs-control pair]
#   calibrated-s2 -> shuffled-s1 -> [interim] -> format-s2 -> pooled-s1 -> [interim] -> calibrated-s3 -> [interim]
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
MODEL=Qwen/Qwen2.5-0.5B-Instruct
mkdir -p results/v2/rl/logs

wait_for() { while [ ! -f "artifacts/rl/$1/step-$2/adapter_model.safetensors" ]; do sleep 60; done; }
stop_run() {
  pkill -f "study.rl --condition $1 --seed $2 " ; sleep 10
  printf '{"finished_at": "%s", "steps": 200, "note": "launched with --steps 300; stopped after the step-200 checkpoint (v2.6 feasibility deviation)"}\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "results/v2/rl/$1-s$2/finished.json"
}
run() {
  if [ -f "results/v2/rl/$1-s$2/finished.json" ]; then echo "skip $1-s$2"; return; fi
  rm -rf "results/v2/rl/$1-s$2" "artifacts/rl/$1-s$2"
  echo "start $1-s$2 $(date -u)"
  $PY -m study.rl --condition "$1" --seed "$2" --model "$MODEL" --steps 200 > "results/v2/rl/logs/$1-s$2.log" 2>&1 || echo "FAILED $1-s$2"
  echo "end $1-s$2 $(date -u)"
}
interim() { $PY -u -m study.rl_interim --wait-for "$@" >> results/v2/rl_interim.log 2>&1 || echo "interim failed"; }

wait_for format-s1 200; stop_run format 1
run calibrated 1
interim calibrated-s1:200 format-s1:200
run calibrated 2
run shuffled 1
interim calibrated-s2:200 shuffled-s1:200
run format 2
run pooled 1
interim format-s2:200 pooled-s1:200
run calibrated 3
interim calibrated-s3:200
echo "RL schedule complete $(date -u)"
