"""Analyse RL evaluations (protocol H4): calibration under the reward classifier versus held-out measures, and
reward-hacking diagnostics.

For every evaluated (run, step), using outputs that pass the format check:
  - label distribution per outlet under M1 (reward), M3 (held-out classifier) and M2 (judge slant sign, if judged);
    total variation (TV) to the real outlet's distribution under the same measure, averaged over outlets;
  - party-mention rates against the real outlet's rate; M1 TV after masking names; party-swap label-flip rate;
  - distinct-2, exact-duplicate rate, mean reference NLL, politicalness, format-fail rate, length.
Calibrated runs are contrasted with format-only runs: shrink = (TV_format - TV_calibrated) / TV_format per measure,
and transfer = shrink_M3 / shrink_M1 (and M2). A noise floor is the TV between two disjoint real subsamples of the
same size.
Usage: python -m study.rl_analyze score   (score all eval samples)   |   python -m study.rl_analyze report
"""
import argparse
from collections import Counter, defaultdict
import math
import random
import re

import numpy as np

from study.common import LABELS, OUTLETS, RESULTS, party_mentioned, read_jsonl, scoring_text, write_json, write_jsonl, now
from study.measures import load_cache, score, text_hash
from study.swap import swap_text

NAMES = re.compile(r'\b(donald trump|kamala harris|joe biden|jd vance|tim walz|trump|harris|biden|vance|walz|'
                   r'democrats?|democratic|republicans?|gop|maga)\b', re.I)


def masked_names(text):
    return NAMES.sub('[PARTY]', text)


def eval_files():
    return sorted((RESULTS / 'rl_eval').glob('*/step-*.jsonl'))


def score_all():
    texts = []
    for f in eval_files():
        for r in read_jsonl(f):
            t = scoring_text(r['text'])
            if t:
                texts += [t, masked_names(t)]
                s = swap_text(t)
                if s:
                    texts.append(s)
    score(texts, ['leaning'])
    score([t for t in texts if '[PARTY]' not in t], ['leaning_heldout', 'loaded', 'political'])


def label(p):
    return max(p, key=p.get)


def dist(labels, keys):
    c = Counter(labels)
    n = max(1, sum(c.values()))
    return np.array([c[k] / n for k in keys])


def tv(a, b):
    return 0.5 * float(np.abs(np.asarray(a) - np.asarray(b)).sum())


def distinct2(texts):
    grams = [tuple(w[i:i + 2]) for t in texts for w in [t.lower().split()] for i in range(len(w) - 1)]
    return len(set(grams)) / max(1, len(grams))


def real_reference():
    real = [r for r in read_jsonl(RESULTS / 'real' / 'real_tweets.jsonl') if r['included']]
    m1, m3 = load_cache('leaning'), load_cache('leaning_heldout')
    from study.data import judge_results, masked_id
    judge = judge_results('claude-sonnet-5-5', ['all-v2'])
    ref = {}
    for o in OUTLETS:
        rs = [r for r in real if r['outlet'] == o]
        js = [judge.get(masked_id(r['text'])) for r in rs]
        ref[o] = {'m1': dist([label(m1[text_hash(r['text'])]['probabilities']) for r in rs], LABELS),
                  'm3': dist([label(m3[text_hash(r['text'])]['probabilities']) for r in rs], LABELS),
                  'judge': dist([int(np.sign(j['slant'])) for j in js if j], [-1, 0, 1]),
                  'party_any': float(np.mean([party_mentioned(r['text']) != 'neither' for r in rs])),
                  'n': len(rs), 'texts': [r['text'] for r in rs]}
    return ref


def noise_floor(ref, n, measure='m1', reps=200, seed=3):
    """TV between two disjoint random real subsamples of size n per outlet (averaged over outlets)."""
    m1, m3 = load_cache('leaning'), load_cache('leaning_heldout')
    cache = m1 if measure == 'm1' else m3
    rng = random.Random(seed)
    vals = []
    for _ in range(reps):
        tvs = []
        for o in OUTLETS:
            texts = ref[o]['texts'][:]
            rng.shuffle(texts)
            k = min(n, len(texts) // 2)
            a = dist([label(cache[text_hash(t)]['probabilities']) for t in texts[:k]], LABELS)
            b = dist([label(cache[text_hash(t)]['probabilities']) for t in texts[k:2 * k]], LABELS)
            tvs.append(tv(a, b))
        vals.append(np.mean(tvs))
    return {'n_per_outlet': n, 'mean': float(np.mean(vals)), 'p95': float(np.percentile(vals, 95))}


def evaluate_file(path, ref, judge=None):
    rows = read_jsonl(path)
    m1, m3, loaded, pol = (load_cache(m) for m in ('leaning', 'leaning_heldout', 'loaded', 'political'))
    out = {}
    for split in ('train', 'heldout', 't1'):
        sel = [r for r in rows if r['voice'] in OUTLETS and (r['template'] == 't1' if split == 't1' else (r['template'] == 'v1' and r['split'] == split))]
        if not sel:
            continue
        ok = [r for r in sel if not r['format_fail'] and scoring_text(r['text'])]
        per = {}
        for o in OUTLETS:
            texts = [scoring_text(r['text']) for r in ok if r['voice'] == o]
            if not texts:
                continue
            labs1 = [label(m1[text_hash(t)]['probabilities']) for t in texts]
            labs3 = [label(m3[text_hash(t)]['probabilities']) for t in texts]
            labsm = [label(m1[text_hash(masked_names(t))]['probabilities']) for t in texts]
            swaps = [(t, swap_text(t)) for t in texts]
            flips = [label(m1[text_hash(t)]['probabilities']) != label(m1[text_hash(s)]['probabilities']) for t, s in swaps if s]
            jd = None
            if judge:
                from study.data import masked_id
                js = [judge.get(masked_id(t)) for t in texts]
                js = [j for j in js if j]
                if js:
                    jd = dist([int(np.sign(j['slant'])) for j in js], [-1, 0, 1])
            party_any = float(np.mean([party_mentioned(t) != 'neither' for t in texts]))
            per[o] = {'n': len(texts), 'm1': dist(labs1, LABELS).tolist(), 'tv_m1': tv(dist(labs1, LABELS), ref[o]['m1']),
                      'm3': dist(labs3, LABELS).tolist(), 'tv_m3': tv(dist(labs3, LABELS), ref[o]['m3']),
                      'tv_m1_names_masked': tv(dist(labsm, LABELS), ref[o]['m1']),
                      'judge': jd.tolist() if jd is not None else None, 'tv_judge': tv(jd, ref[o]['judge']) if jd is not None else None,
                      'party_any': party_any, 'party_any_real': ref[o]['party_any'], 'party_overshoot': party_any - ref[o]['party_any'],
                      'swap_flip_rate': float(np.mean(flips)) if flips else None,
                      'loaded': float(np.mean([loaded[text_hash(t)]['probabilities']['biased'] for t in texts if text_hash(t) in loaded])),
                      'political': float(np.mean([pol[text_hash(t)]['p_political'] for t in texts if text_hash(t) in pol]))}
        texts_all = [scoring_text(r['text']) for r in ok]
        mean = lambda k: float(np.mean([per[o][k] for o in per if per[o].get(k) is not None])) if any(per[o].get(k) is not None for o in per) else None
        out[split] = {'n': len(sel), 'format_fail_rate': float(np.mean([r['format_fail'] for r in sel])),
                      'mean_chars': float(np.mean([len(r['text']) for r in sel])), 'ref_nll': float(np.mean([r['ref_nll'] for r in sel])),
                      'distinct2': distinct2(texts_all), 'duplicate_rate': 1 - len(set(t.lower() for t in texts_all)) / max(1, len(texts_all)),
                      'tv_m1': mean('tv_m1'), 'tv_m3': mean('tv_m3'), 'tv_judge': mean('tv_judge'),
                      'tv_m1_names_masked': mean('tv_m1_names_masked'), 'party_overshoot': mean('party_overshoot'),
                      'swap_flip_rate': mean('swap_flip_rate'), 'loaded': mean('loaded'), 'political': mean('political'),
                      'per_outlet': per}
    return out


def report():
    ref = real_reference()
    from study.data import judge_results
    judge = judge_results('claude-sonnet-5-5', ['rl-eval'])
    res = {'created_at': now(), 'evals': {}, 'noise_floor': {}}
    for f in eval_files():
        run, step = f.parent.name, int(f.stem.split('-')[1])
        res['evals'].setdefault(run, {})[step] = evaluate_file(f, ref, judge)
    for n in (40, 100, 200):
        res['noise_floor'][f'm1_n{n}'] = noise_floor(ref, n, 'm1')
        res['noise_floor'][f'm3_n{n}'] = noise_floor(ref, n, 'm3')
    # Pre-registered contrast at the final checkpoint: calibrated vs format-only, per seed pairing and pooled.
    from study.rl_eval import FINAL_STEP
    final = {run: steps.get(FINAL_STEP) for run, steps in res['evals'].items() if steps.get(FINAL_STEP)}
    base = res['evals'].get('base', {}).get(0)
    def cond(run):
        return run.rsplit('-s', 1)[0]
    contrasts = {}
    for split in ('train', 'heldout', 't1'):
        fmt = [final[r][split] for r in final if cond(r) == 'format' and split in final[r]]
        if not fmt:
            continue
        ref_tv = {k: np.mean([x[k] for x in fmt if x[k] is not None]) if any(x[k] is not None for x in fmt) else None
                  for k in ('tv_m1', 'tv_m3', 'tv_judge')}
        per_run = {}
        for r in final:
            if cond(r) not in ('calibrated', 'shuffled', 'pooled') or split not in final[r]:
                continue
            x = final[r][split]
            shrink = {k: (ref_tv[k] - x[k]) / ref_tv[k] if (ref_tv[k] and x[k] is not None) else None for k in ref_tv}
            per_run[r] = {'shrink': shrink,
                          'transfer_m3': shrink['tv_m3'] / shrink['tv_m1'] if shrink['tv_m1'] and shrink['tv_m3'] is not None else None,
                          'transfer_judge': shrink['tv_judge'] / shrink['tv_m1'] if shrink['tv_m1'] and shrink['tv_judge'] is not None else None,
                          'hacking_flags': {
                              'party_overshoot_gt_10pts': bool(x['party_overshoot'] is not None and x['party_overshoot'] > 0.10),
                              'distinct2_drop_gt_20pct': bool(base and split in base and x['distinct2'] < 0.8 * base[split]['distinct2']),
                              'duplicates_up_gt_10pts': bool(base and split in base and x['duplicate_rate'] > base[split]['duplicate_rate'] + 0.10),
                              'names_masked_gain_halved': bool(ref_tv['tv_m1'] and x['tv_m1_names_masked'] is not None and
                                                               (ref_tv['tv_m1'] - x['tv_m1_names_masked']) < 0.5 * (ref_tv['tv_m1'] - x['tv_m1']))}}
        contrasts[split] = {'format_reference_tv': ref_tv, 'runs': per_run}
    res['h4_contrasts'] = contrasts
    res['real_reference'] = {o: {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in ref[o].items() if k != 'texts'} for o in OUTLETS}
    write_json(RESULTS / 'analysis' / 'rl_results.json', res)
    print('wrote results/v2/analysis/rl_results.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('step', choices=['score', 'report'])
    args = parser.parse_args()
    score_all() if args.step == 'score' else report()


if __name__ == '__main__':
    main()
