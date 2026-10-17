"""Revisiting v1: start from the published v1 real-vs-expected comparison and apply one correction at a time.

v1 (README, V1Results.md) reported, per outlet, real score minus expected score, where score = 100*mean[P(right)-P(left)]
under M1, real = all 2,500 tweets scored with their t.co links, expected = 525 Haiku tweets per voice (main-v1) scored as
written. Each step below changes exactly one thing relative to the previous step:
  S0 v1 as published (v1 score files)
  S1 same texts, links removed from real tweets and wrapping quotes from generated ones (v2 scoring text)
  S2 real tweets restricted to the cleaned set (pre-registered exclusions)
  S3 generated tweets restricted to those passing the source dataset's election-keyword filter (symmetric selection)
  S4 both sides restricted to tweets that name both parties (party-matched)
  S5 same comparison as S2 but measured by the blind judge's slant instead of M1 (different scale: -3..+3)
Each step reports real level, expected level, real - expected (v1's sign), M1 center shares, and bootstrap 95% CIs.
Usage: python -m study.v1_revisited   (writes results/v2/analysis/v1_revisited.json)
"""
import numpy as np

from study.common import OUTLETS, RESULTS, ROOT, read_jsonl, scoring_text, write_json, now
from study.measures import load_cache, text_hash

rng = np.random.default_rng(5)


def net(p):
    return 100 * (p['right'] - p['left'])


def gap_ci(real, gen, B=4000):
    real, gen = np.array(real, float), np.array(gen, float)
    if len(real) < 5 or len(gen) < 5:
        return {'real': float(real.mean()) if len(real) else None, 'expected': float(gen.mean()) if len(gen) else None,
                'gap': None, 'n_real': int(len(real)), 'n_expected': int(len(gen))}
    draws = [real[rng.integers(0, len(real), len(real))].mean() - gen[rng.integers(0, len(gen), len(gen))].mean() for _ in range(B)]
    return {'real': float(real.mean()), 'expected': float(gen.mean()), 'gap': float(real.mean() - gen.mean()),
            'ci95': [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))], 'n_real': int(len(real)), 'n_expected': int(len(gen))}


def main():
    # S0: v1 score files exactly as published
    v1_real = {r['tweet_id']: r for r in read_jsonl(ROOT / 'dataset' / 'political_scores.jsonl')}
    v1_gen = {r['key']: r for r in read_jsonl(ROOT / 'reports' / 'v1' / 'political_scores.jsonl')}
    units = read_jsonl(RESULTS / 'analysis' / 'units_j5.jsonl')
    real_units = [u for u in units if u['source'] == 'real']
    gen_units = [u for u in units if u['condition'] == 'main-v1' and u['group'] in OUTLETS]
    m1 = load_cache('leaning')
    out = {'created_at': now(), 'steps': {}}

    def center(ps):
        return float(np.mean([max(p, key=p.get) == 'center' for p in ps])) if ps else None

    s0 = {}
    for o in OUTLETS:
        rp = [r['probabilities'] for r in v1_real.values() if r['account'] == o]
        gp = [r['probabilities'] for k, r in v1_gen.items() if k.startswith('main-v1/') and f'-{o}-' in k]
        s0[o] = {**gap_ci([net(p) for p in rp], [net(p) for p in gp]), 'center_real': center(rp), 'center_expected': center(gp)}
    out['steps']['S0_v1_as_published'] = s0

    def step(real_sel, gen_sel, measure='m1'):
        res = {}
        for o in OUTLETS:
            rs = [u for u in real_units if u['group'] == o and real_sel(u)]
            gs = [u for u in gen_units if u['group'] == o and gen_sel(u)]
            if measure == 'm1':
                rp, gp = [m1[text_hash(u['text'])]['probabilities'] for u in rs], [m1[text_hash(u['text'])]['probabilities'] for u in gs]
                res[o] = {**gap_ci([net(p) for p in rp], [net(p) for p in gp]), 'center_real': center(rp), 'center_expected': center(gp)}
            else:
                res[o] = gap_ci([u['judge']['slant'] for u in rs if u['judge']], [u['judge']['slant'] for u in gs if u['judge']])
        return res

    out['steps']['S1_links_removed'] = step(lambda u: True, lambda u: True)
    out['steps']['S2_cleaned_real'] = step(lambda u: u['included'], lambda u: True)
    out['steps']['S3_symmetric_keyword_selection'] = step(lambda u: u['included'], lambda u: bool(u['keywords']))
    out['steps']['S4_both_parties_named'] = step(lambda u: u['included'] and u['party'] == 'both', lambda u: u['party'] == 'both')
    out['steps']['S5_judge_instead_of_M1'] = step(lambda u: u['included'], lambda u: True, measure='judge')
    out['steps']['S5b_judge_both_parties_named'] = step(lambda u: u['included'] and u['party'] == 'both', lambda u: u['party'] == 'both', measure='judge')
    write_json(RESULTS / 'analysis' / 'v1_revisited.json', out)
    for name, res in out['steps'].items():
        print(name)
        for o in OUTLETS:
            r = res[o]
            ci = f"[{r['ci95'][0]:+.1f},{r['ci95'][1]:+.1f}]" if r.get('ci95') else ''
            c = f" center {r['center_real']:.2f}/{r['center_expected']:.2f}" if r.get('center_real') is not None else ''
            print(f"   {o:8} real {r['real']:+7.2f} exp {r['expected']:+7.2f} gap {r['gap'] if r['gap'] is None else round(r['gap'], 2)} {ci} n {r['n_real']}/{r['n_expected']}{c}")


if __name__ == '__main__':
    main()
