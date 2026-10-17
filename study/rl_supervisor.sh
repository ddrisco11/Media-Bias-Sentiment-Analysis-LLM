#!/bin/sh
# RL schedule after the v2.6 feasibility deviation: final checkpoint at step 200, runs in priority-ordered pairs,
# with an interim report (evaluate, score, judge, summarise, push) after each milestone.
#   pair 1: calibrated-s1 + format-s1 (launched with --steps 300; stopped here once their step-200 adapter exists)
#   pair 2: calibrated-s2 + shuffled-s1
#   pair 3: format-s2 + pooled-s1
#   pair 4: calibrated-s3
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
MODEL=Qwen/Qwen2.5-0.5B-Instruct
mkdir -p results/v2/rl/logs

wait_for() { while [ ! -f "artifacts/rl/$1/step-$2/adapter_model.safetensors" ]; do sleep 60; done; }
stop_run() {  # stop a pair-1 run after its step-200 adapter is saved, and mark it finished at 200
  pkill -f "study.rl --condition $1 --seed $2 " ; sleep 5
  printf '{"finished_at": "%s", "steps": 200, "note": "launched with --steps 300; stopped after the step-200 checkpoint (v2.6 feasibility deviation)"}\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "results/v2/rl/$1-s$2/finished.json"
}
run() {
  if [ -f "results/v2/rl/$1-s$2/finished.json" ]; then echo "skip $1-s$2"; return; fi
  rm -rf "results/v2/rl/$1-s$2"
  $PY -m study.rl --condition "$1" --seed "$2" --model "$MODEL" --steps 200 > "results/v2/rl/logs/$1-s$2.log" 2>&1 || echo "FAILED $1-s$2"
}
interim() { $PY -u -m study.rl_interim --wait-for "$@" >> results/v2/rl_interim.log 2>&1 || echo "interim failed"; }

wait_for calibrated-s1 100; wait_for format-s1 100; interim calibrated-s1:100 format-s1:100
wait_for calibrated-s1 200; stop_run calibrated 1
wait_for format-s1 200; stop_run format 1
( run calibrated 2 ) & ( run shuffled 1 ) &
interim calibrated-s1:200 format-s1:200
wait
interim calibrated-s2:200 shuffled-s1:200
( run format 2 ) & ( run pooled 1 ) & wait
interim format-s2:200 pooled-s1:200
run calibrated 3
interim calibrated-s3:200
echo "RL schedule complete $(date -u)"
