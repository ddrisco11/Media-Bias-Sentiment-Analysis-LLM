"""Interim RL reports: wait for checkpoints, evaluate them, score, judge, summarise, commit and push.

For every run directory with saved adapters (artifacts/rl/<run>/step-<n>) that has not been evaluated yet, this:
evaluates the base model once (full sample), evaluates each new checkpoint (reduced sample, or full at step 300),
scores all samples with the classifiers, rates them with the blind judge (Batch API), recomputes
results/v2/analysis/rl_results.json, writes docs/paper/rl_interim.md and figure 7, and commits and pushes.
Usage: python -m study.rl_interim --wait-for calibrated-s1:100 format-s1:100
"""
import argparse
import json
import subprocess
import time

from study.common import RESULTS, ROOT, now

PY = str(ROOT / '.venv' / 'bin' / 'python')


def sh(*args):
    print('$', ' '.join(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def available():
    out = {}
    for d in (ROOT / 'artifacts' / 'rl').glob('*/step-*'):
        if d.parent.name.startswith('calibrated-s0'):
            continue
        out.setdefault(d.parent.name, set()).add(int(d.name.split('-')[1]))
    return out


def evaluated(run, step):
    return (RESULTS / 'rl_eval' / run / f'step-{step}.jsonl').exists()


def summary_md(res):
    lines = [f'# RL interim report ({now()[:16]} UTC)', '',
             'Interim, not final. Intermediate checkpoints use small samples (5 per outlet x training topic, 10 per outlet x',
             'held-out topic); step 300 uses the full pre-registered sample. Lower TV = closer to the real outlet\'s label',
             'distribution under that measure. M1 is the reward model; M3 and the judge never enter training.', '',
             '| Run | Step | Split | Format fail | TV M1 (reward) | TV M1, names masked | TV M3 (held-out) | TV judge | Party-mention overshoot | Distinct-2 | Duplicates |',
             '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    f = lambda x, d=3: '' if x is None else f'{x:.{d}f}'
    for run in sorted(res['evals']):
        for step in sorted(res['evals'][run], key=int):
            for split, v in res['evals'][run][step].items():
                lines.append(f"| {run} | {step} | {split} | {f(v['format_fail_rate'], 2)} | {f(v['tv_m1'])} | {f(v['tv_m1_names_masked'])} | "
                             f"{f(v['tv_m3'])} | {f(v['tv_judge'])} | {f(v['party_overshoot'], 2)} | {f(v['distinct2'], 2)} | {f(v['duplicate_rate'], 2)} |")
    nf = res.get('noise_floor', {})
    lines += ['', 'Noise floor (TV between two real subsamples of the same size, mean over outlets): ' +
              ', '.join(f"{k} {v['mean']:.3f}" for k, v in nf.items()), '',
              '![RL curves](figures/fig7_rl_curves.png)', '']
    if res.get('h4_contrasts'):
        lines += ['## Calibrated versus format-only at step 300', '', '```', json.dumps(res['h4_contrasts'], indent=1)[:4000], '```']
    return '\n'.join(lines) + '\n'


def cycle():
    avail = available()
    todo = [(r, s) for r, steps in avail.items() for s in sorted(steps) if not evaluated(r, s)]
    if not (RESULTS / 'rl_eval' / 'base' / 'step-0.jsonl').exists():
        sh(PY, '-m', 'study.rl_eval', '--base')
    for run, step in todo:
        sh(PY, '-m', 'study.rl_eval', '--run', run, '--steps', str(step))
    sh(PY, '-m', 'study.rl_analyze', 'score')
    files = [str(p.relative_to(ROOT)) for p in sorted((RESULTS / 'rl_eval').glob('*/step-*.jsonl'))]
    sh(PY, '-m', 'study.judge', 'rate', '--set', 'rl-eval', '--input', *files)
    sh(PY, '-m', 'study.rl_analyze', 'report')
    sh(PY, '-m', 'study.figures')
    res = json.loads((RESULTS / 'analysis' / 'rl_results.json').read_text())
    (ROOT / 'docs' / 'paper' / 'rl_interim.md').write_text(summary_md(res), encoding='utf-8')
    sh('git', 'add', 'docs/paper/rl_interim.md', 'docs/paper/figures', 'results/v2/rl_eval', 'results/v2/analysis/rl_results.json',
       'results/v2/judge/claude-sonnet-5-5/rl-eval', 'results/v2/rl')
    subprocess.run(['git', 'commit', '-q', '-m', f'RL interim report {now()[:16]}\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>'], cwd=ROOT)
    subprocess.run(['git', 'push', '-q'], cwd=ROOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--wait-for', nargs='+', required=True, help='run:step pairs that must have checkpoints first')
    args = parser.parse_args()
    need = [(w.split(':')[0], int(w.split(':')[1])) for w in args.wait_for]
    while not all(s in available().get(r, set()) for r, s in need):
        time.sleep(120)
    cycle()
    print('interim report done', now(), flush=True)


if __name__ == '__main__':
    main()
