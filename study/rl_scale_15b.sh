#!/bin/sh
# Exploratory scale check (added 2026-10-17, v2.7): Qwen2.5-1.5B-Instruct, calibrated vs format-only, 200 steps each.
# Starts after the pre-registered 0.5B schedule finishes; one GPU job at a time.
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
MODEL=Qwen/Qwen2.5-1.5B-Instruct
until grep -q "RL schedule complete" results/v2/rl_supervisor.log 2>/dev/null; do sleep 120; done
echo "timing pilot $(date -u)"
$PY -m study.rl --condition calibrated --seed 0 --steps 10 --timing --tag timing-1.5b --model "$MODEL" > results/v2/rl/logs/timing-1.5b.log 2>&1
mv results/v2/rl/calibrated-s0-timing-1.5b results/v2/rl/timing/ 2>/dev/null
$PY -m study.rl_eval --base --base-name base-1.5b >> results/v2/rl_interim.log 2>&1
for c in calibrated format; do
  echo "start $c-s1-1.5b $(date -u)"
  $PY -m study.rl --condition "$c" --seed 1 --steps 200 --tag 1.5b --model "$MODEL" > "results/v2/rl/logs/$c-s1-1.5b.log" 2>&1 || echo "FAILED $c-s1-1.5b"
done
$PY -u -m study.rl_interim --wait-for calibrated-s1-1.5b:200 format-s1-1.5b:200 >> results/v2/rl_interim.log 2>&1
echo "1.5B scale check complete $(date -u)"
