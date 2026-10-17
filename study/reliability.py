"""Judge reliability (protocol sections 5-6): inter-judge agreement, test-retest, and, once human labels exist,
human-human agreement, judge/classifier validity against humans, and prediction-powered estimates.

Usage: python -m study.reliability            (judges only; writes results/v2/analysis/reliability.json)
       python -m study.reliability --human    (also reads results/v2/human/coded_A.csv and coded_B.csv)
Coded sheets are the coder sheets with the rating columns filled in; rows are matched through key.jsonl.
"""
import argparse
import csv

import numpy as np

from study.common import RESULTS, read_jsonl, write_json, now
from study.data import judge_results

STANCE = {'-2': -2, '-1': -1, '0': 0, '1': 1, '2': 2}


def alpha(rows, level):
    """Krippendorff's alpha; rows = list of per-coder lists aligned by item (None = missing)."""
    import krippendorff
    data = np.array([[np.nan if v is None else v for v in r] for r in rows], dtype=float)
    if np.sum(~np.isnan(data).all(0)) < 5:
        return None
    try:
        return float(krippendorff.alpha(reliability_data=data, level_of_measurement=level))
    except (ValueError, ZeroDivisionError):
        return None


def fields(j):
    if not j:
        return {'slant': None, 'stance_dem': None, 'stance_rep': None, 'genre': None, 'dem_na': None}
    return {'slant': j['slant'], 'stance_dem': STANCE.get(j['stance_democrats']), 'stance_rep': STANCE.get(j['stance_republicans']),
            'genre': j['genre'], 'dem_na': int(j['stance_democrats'] == 'NA')}


def compare(a, b, ids):
    out = {}
    for f, level in [('slant', 'ordinal'), ('stance_dem', 'ordinal'), ('stance_rep', 'ordinal'), ('genre', 'nominal'), ('dem_na', 'nominal')]:
        xa, xb = [fields(a.get(i))[f] for i in ids], [fields(b.get(i))[f] for i in ids]
        if f == 'genre':
            cats = sorted({v for v in xa + xb if v is not None})
            xa = [cats.index(v) if v is not None else None for v in xa]
            xb = [cats.index(v) if v is not None else None for v in xb]
        both = [(x, y) for x, y in zip(xa, xb) if x is not None and y is not None]
        out[f] = {'alpha': alpha([xa, xb], level), 'n_both': len(both),
                  'exact_agreement': float(np.mean([x == y for x, y in both])) if both else None,
                  'mean_difference': float(np.mean([x - y for x, y in both])) if both and f != 'genre' else None}
    return out


def judges():
    human_items = [r['item'] for r in read_jsonl(RESULTS / 'human' / 'human_set.jsonl')]
    sonnet = {**judge_results('claude-sonnet-5-5', ['all-v2', 'swap-v2'])}
    opus = judge_results('claude-opus-5-5', ['human-set'])
    qwen = judge_results('qwen2.5-7b', ['human-set'])
    retest = judge_results('claude-sonnet-5-5', ['retest-retest'])
    out = {'human_set_items': len(human_items),
           'coverage': {'sonnet': sum(i in sonnet for i in human_items), 'opus': sum(i in opus for i in human_items),
                        'qwen': sum(i in qwen for i in human_items)},
           'sonnet_vs_opus': compare(sonnet, opus, human_items),
           'sonnet_vs_qwen': compare(sonnet, qwen, human_items),
           'opus_vs_qwen': compare(opus, qwen, human_items),
           'sonnet_test_retest': compare(sonnet, retest, list(retest))}
    return out


def read_coded(path):
    with open(path, encoding='utf-8') as handle:
        return {int(r['item_no']): r for r in csv.DictReader(handle)}


def human(out):
    """Human-human agreement, judge/classifier vs human, and a prediction-powered (PPI) estimate of mean slant."""
    key = read_jsonl(RESULTS / 'human' / 'key.jsonl')
    coded = {c: read_coded(RESULTS / 'human' / f'coded_{c}.csv') for c in ('A', 'B')}
    by_item = {}
    for k in key:
        r = coded[k['coder']].get(k['item_no'])
        if r and r.get('slant', '').strip() not in ('', None):
            by_item.setdefault(k['item'], {})[k['coder']] = {'slant': int(r['slant']), 'stance_dem': STANCE.get(r['stance_democrats'].strip()),
                                                           'stance_rep': STANCE.get(r['stance_republicans'].strip()), 'genre': r['genre'].strip()}
    ids = sorted(by_item)
    hh = {}
    for f, level in [('slant', 'ordinal'), ('stance_dem', 'ordinal'), ('stance_rep', 'ordinal')]:
        hh[f] = alpha([[by_item[i].get('A', {}).get(f) for i in ids], [by_item[i].get('B', {}).get(f) for i in ids]], level)
    out['human_human_alpha'] = hh
    out['human_gold_status'] = {f: ('gold' if a and a >= 0.8 else 'tentative' if a and a >= 0.667 else 'not usable') for f, a in hh.items()}
    # consensus = mean of coders (rounded) where both coded
    consensus = {i: {f: (np.mean([by_item[i][c][f] for c in by_item[i] if by_item[i][c][f] is not None]))
                     for f in ('slant', 'stance_dem', 'stance_rep')} for i in ids}
    sonnet = judge_results('claude-sonnet-5-5', ['all-v2', 'swap-v2'])
    out['judge_vs_human'] = {f: alpha([[round(consensus[i][f]) if not np.isnan(consensus[i][f]) else None for i in ids],
                                       [fields(sonnet.get(i))[f] for i in ids]], 'ordinal') for f in ('slant', 'stance_dem', 'stance_rep')}
    # PPI for the mean slant of the sampled population: judge mean on all units + human-corrected rectifier on the
    # labelled sample, weighted by inverse inclusion probabilities (Angelopoulos et al. 2023; Egami et al. 2023).
    sample = [k for k in key if k['coder'] == 'A' and k['kind'] == 'sample' and k['item'] in consensus]
    w = np.array([1 / k['inclusion_prob'] for k in sample])
    rect = np.array([consensus[k['item']]['slant'] - (fields(sonnet.get(k['item']))['slant'] or 0) for k in sample])
    out['ppi_slant_rectifier'] = {'weighted_mean': float(np.sum(w * rect) / np.sum(w)), 'n': len(sample),
                                  'note': 'add to the judge-based population mean to obtain the PPI estimate'}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--human', action='store_true')
    args = parser.parse_args()
    out = {'created_at': now(), **judges()}
    if args.human:
        out = human(out)
    write_json(RESULTS / 'analysis' / 'reliability.json', out)
    import json
    print(json.dumps({k: v for k, v in out.items() if k != 'created_at'}, indent=1, default=str)[:3000])


if __name__ == '__main__':
    main()
