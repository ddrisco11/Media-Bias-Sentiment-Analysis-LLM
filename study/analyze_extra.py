"""Additional analyses requested by the independent peer review (logged in protocol change log v2.5).

All use existing data. Each is labelled in the paper as pre-registered sensitivity (Jaccard, full set, Holm) or as
post-review exploratory.
  A. judge slant levels, spreads and gaps within the party-named stratum (rules out a "no party named -> slant 0" floor)
  B. Fox within genre (real vs generated, per judge genre)
  C. event arm: TOST at the data-driven and pre-registered margins, lexical overlap, event-matched individuation,
     all-measure gaps
  D. period-matched comparison: real P2 (21 Jul - 5 Nov) vs main-v1 and vs sens-month prompts dated Aug-Oct
  E. Fox slant conditional on outlet misidentification (M6)
  F. pre-registered sensitivities: Jaccard 0.3/0.7 clusters, full (uncleaned) real set
  G. bootstrap CIs for external-benchmark rank correlations
  H. judge name-sensitivity detail: change rate on real neutral items; sign-reversal rate on partisan items
  I. judge slant variance by condition (range restriction)
  J. equivalence at a data-driven margin (half the real between-outlet spread)
Usage: python -m study.analyze_extra   (writes results/v2/analysis/extra.json)
"""
import json
import random
import re
from collections import Counter, defaultdict

import numpy as np

from study.analyze import IDX, OUTLETS, PRIMARY, TOPICS, real_topic_weights, values, voice_contrasts
from study.common import RESULTS, read_jsonl, write_json, now
from study.stats import Bootstrap, tost_equivalent

rng = np.random.default_rng(11)


def boot_mean(xs, B=5000):
    xs = np.array([x for x in xs if x is not None and not np.isnan(x)])
    if len(xs) < 2:
        return {'n': int(len(xs)), 'mean': float(xs.mean()) if len(xs) else None}
    draws = np.array([xs[rng.integers(0, len(xs), len(xs))].mean() for _ in range(B)])
    return {'n': int(len(xs)), 'mean': float(xs.mean()), 'ci95': [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
            'sd': float(xs.std())}


def slant(u):
    return u['judge']['slant'] if u.get('judge') else None


def real(units, o=None):
    return [u for u in units if u['source'] == 'real' and u['included'] and (o is None or u['group'] == o)]


def gen(units, cond, o=None):
    return [u for u in units if u['condition'] == cond and (o is None or u['group'] == o)]


def party_stratum(units):
    out = {}
    for o in OUTLETS:
        r = [slant(u) for u in real(units, o) if u['party'] != 'neither']
        g = [slant(u) for u in gen(units, 'main-v1', o) if u['party'] != 'neither']
        e = [slant(u) for u in gen(units, 'event-v1', o) if u['party'] != 'neither']
        br, bg, be = boot_mean(r), boot_mean(g), boot_mean(e)
        out[o] = {'real': br, 'generated_main': bg, 'generated_event': be,
                  'sd_ratio_main': (bg.get('sd') or 0) / br['sd'] if br.get('sd') else None,
                  'sd_ratio_event': (be.get('sd') or 0) / br['sd'] if br.get('sd') else None}
    return out


def fox_within_genre(units):
    out = {}
    for g in ['news_report', 'breaking_or_results', 'explainer_or_roundup', 'opinion_or_advocacy', 'live_or_promo', 'civic_reminder', 'quote_led']:
        r = [slant(u) for u in real(units, 'foxnews') if u['judge'] and u['judge']['genre'] == g]
        m = [slant(u) for u in gen(units, 'main-v1', 'foxnews') if u['judge'] and u['judge']['genre'] == g]
        e = [slant(u) for u in gen(units, 'event-v1', 'foxnews') if u['judge'] and u['judge']['genre'] == g]
        out[g] = {'real': boot_mean(r), 'generated_main': boot_mean(m), 'generated_event': boot_mean(e)}
    return out


def words(t):
    return set(re.findall(r"[a-z][a-z'\-]+", t.lower())) - {'the', 'a', 'an', 'of', 'to', 'in', 'and', 'on', 'for', 'is', 'at', 'with', 'by', 'as', 'that', 'his', 'her', 'its', 'was', 'be', 'from', 'has', 'are', 'it'}


def jaccard(a, b):
    a, b = words(a), words(b)
    return len(a & b) / len(a | b) if a | b else 0.0


def event_arm(units, margin_data):
    events = {e['event']: e for e in read_jsonl(RESULTS / 'events' / 'events.jsonl')}
    real_by_key = {u['id']: u for u in units if u['source'] == 'real'}
    gen_by = defaultdict(list)
    for u in gen(units, 'event-v1'):
        gen_by[(u['event'], u['group'])].append(u)
    # lexical overlap: generated vs same-outlet real member tweets, compared with description vs those tweets
    ov_gen, ov_desc, ov_gen_desc = [], [], []
    diffs = defaultdict(list)
    for e, ev in events.items():
        for o in OUTLETS:
            members = [real_by_key[m['key']] for m in ev['members'] if m['outlet'] == o and m['key'] in real_by_key]
            gs = gen_by.get((e, o), [])
            if not members or not gs:
                continue
            for g in gs:
                ov_gen.append(np.mean([jaccard(g['text'], m['text']) for m in members]))
                ov_gen_desc.append(jaccard(g['text'], ev['description']))
            ov_desc.append(np.mean([jaccard(ev['description'], m['text']) for m in members]))
            gv, rv = np.nanmean([values(x) for x in gs], 0), np.nanmean([values(x) for x in members], 0)
            diffs[o].append(gv - rv)
    out = {'lexical_overlap': {'generated_vs_same_outlet_real': float(np.mean(ov_gen)), 'description_vs_same_outlet_real': float(np.mean(ov_desc)),
                               'generated_vs_description': float(np.mean(ov_gen_desc))}, 'gaps': {}, 'tost': {}}
    for o in OUTLETS:
        d = np.array(diffs[o])
        draws = np.array([np.nanmean(d[rng.integers(0, len(d), len(d))], 0) for _ in range(5000)])
        out['gaps'][o] = {m: {'estimate': float(np.nanmean(d[:, IDX[m]])), 'ci95': [float(np.nanpercentile(draws[:, IDX[m]], 2.5)), float(np.nanpercentile(draws[:, IDX[m]], 97.5))],
                              'ci90': [float(np.nanpercentile(draws[:, IDX[m]], 5)), float(np.nanpercentile(draws[:, IDX[m]], 95))]}
                          for m in ['m1_net', 'slant', 'm3_net', 'stance_dem', 'stance_rep', 'loaded', 'party_any', 'chars']}
        out['tost'][o] = {f'slant_margin_{mg}': tost_equivalent(out['gaps'][o]['slant']['ci90'], mg) for mg in (margin_data, 0.4)}
        out['gaps'][o]['n_events'] = int(len(d))
    return out


def event_matched_auc(units):
    """Real vs generated discrimination using only real tweets from the events and their generated counterparts."""
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from study.mask import mask_outlets
    events = read_jsonl(RESULTS / 'events' / 'events.jsonl')
    real_by_key = {u['id']: u for u in units if u['source'] == 'real'}
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device='cpu')
    out = {}
    for o in OUTLETS:
        rs, gs, groups = [], [], []
        for ev in events:
            mem = [real_by_key[m['key']] for m in ev['members'] if m['outlet'] == o and m['key'] in real_by_key]
            g = [u for u in gen(units, 'event-v1', o) if u['event'] == ev['event']]
            if mem and g:
                rs += mem
                gs += g[:max(1, len(mem))]
                groups += [ev['event']] * (len(mem) + len(g[:max(1, len(mem))]))
        if len(rs) < 20:
            continue
        X = model.encode([mask_outlets(u['text']) for u in rs + gs], normalize_embeddings=True, show_progress_bar=False)
        y = np.array([0] * len(rs) + [1] * len(gs))
        scores = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
            scores[te] = LogisticRegression(max_iter=2000).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        out[o] = {'auc': float(roc_auc_score(y, scores)), 'n_real': len(rs), 'n_generated': len(gs)}
    return out


def period_matched(units):
    out = {}
    months_p2 = {'August', 'September', 'October'}
    for o in OUTLETS:
        r = [slant(u) for u in real(units, o) if u['period'] == 'P2']
        rp = [u['party'] != 'neither' for u in real(units, o) if u['period'] == 'P2']
        m = [slant(u) for u in gen(units, 'sens-month', o) if u.get('month') in months_p2]
        out[o] = {'real_P2': boot_mean(r), 'real_P2_party_share': float(np.mean(rp)),
                  'sens_month_Aug_Oct': boot_mean(m), 'main_v1': boot_mean([slant(u) for u in gen(units, 'main-v1', o)])}
    names = Counter()
    for u in gen(units, 'sens-month'):
        if u.get('month') in months_p2:
            for n in ('Harris', 'Biden', 'Trump', 'Walz', 'Vance'):
                names[n] += bool(re.search(rf'\b{n}\b', u['text']))
    out['sens_month_Aug_Oct_name_counts'] = dict(names)
    out['sens_month_Aug_Oct_n'] = sum(1 for u in gen(units, 'sens-month') if u.get('month') in months_p2)
    return out


def misidentified(units):
    out = {}
    for name, sel in [('real', lambda u: u['source'] == 'real' and u['included']), ('main-v1', lambda u: u['condition'] == 'main-v1')]:
        us = [u for u in units if sel(u) and u['group'] == 'foxnews' and u.get('identify') and u.get('judge')]
        out[name] = {'identified_as_fox': boot_mean([slant(u) for u in us if u['identify']['answer'] == 'foxnews']),
                     'not_identified_as_fox': boot_mean([slant(u) for u in us if u['identify']['answer'] != 'foxnews'])}
    return out


def cluster_sensitivity(B=2000):
    out = {}
    for j in (3, 7):
        path = RESULTS / 'analysis' / f'units_j{j}.jsonl'
        if not path.exists():
            continue
        units = read_jsonl(path)
        res = voice_contrasts(units, 'main-v1', real_topic_weights(units), B)
        out[f'jaccard_0.{j}'] = {o: {m: res['minus_neutral'][o][m] for m in PRIMARY} for o in OUTLETS}
        sizes = Counter(Counter(u['cluster'] for u in units if u['condition'] == 'main-v1').values())
        out[f'jaccard_0.{j}']['cluster_size_distribution_main_v1'] = dict(sorted(sizes.items()))
    return out


def full_set(units):
    out = {}
    for o in OUTLETS:
        allr = [u for u in units if u['source'] == 'real' and u['group'] == o]
        out[o] = {m: boot_mean([values(u)[IDX[m]] for u in allr]) for m in PRIMARY}
    return out


def benchmark_ci(units, B=2000):
    from scipy.stats import spearmanr
    ext = json.loads((RESULTS / 'external' / 'ratings.json').read_text())['ratings']
    by = {o: np.array([slant(u) for u in real(units, o) if slant(u) is not None], float) for o in OUTLETS}
    m1 = {o: np.array([values(u)[IDX['m1_net']] for u in real(units, o)], float) for o in OUTLETS}
    out = {}
    for name, r in ext.items():
        ref = [r['values'][o] for o in OUTLETS]
        res = {}
        for label, data in [('slant', by), ('m1_net', m1)]:
            draws = [spearmanr(ref, [data[o][rng.integers(0, len(data[o]), len(data[o]))].mean() for o in OUTLETS])[0] for _ in range(B)]
            res[label] = {'ci95': [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))], 'share_rho_eq_1': float(np.mean(np.isclose(draws, 1)))}
        out[name] = res
    return out


def judge_swap_detail():
    from study.data import judge_results, masked_id
    items = read_jsonl(RESULTS / 'swap' / 'items.jsonl')
    judge = {**judge_results('claude-sonnet-5-5', ['all-v2']), **judge_results('claude-sonnet-5-5', ['swap-v2'])}
    neutral_real, partisan = [], []
    for it in items:
        ja, jb = judge.get(masked_id(it['original'])), judge.get(masked_id(it['swapped']))
        if not ja or not jb:
            continue
        neutral = ja['slant'] == 0 and ja['stance_democrats'] in ('0', 'NA') and ja['stance_republicans'] in ('0', 'NA')
        if neutral and it['source'] == 'real':
            neutral_real.append(jb['slant'] != 0)
        if not neutral:
            partisan.append((ja['slant'], jb['slant']))
    flips = [np.sign(a) == -np.sign(b) and a != 0 for a, b in partisan]
    same = [np.sign(a) == np.sign(b) and a != 0 for a, b in partisan]
    return {'real_neutral_change_rate': float(np.mean(neutral_real)), 'n_real_neutral': len(neutral_real),
            'partisan_sign_reversal_rate': float(np.mean(flips)), 'partisan_same_sign_rate': float(np.mean(same)),
            'partisan_to_zero_rate': float(np.mean([b == 0 and a != 0 for a, b in partisan])), 'n_partisan': len(partisan)}


def judge_variance(units):
    out = {}
    for cond in ['real', 'main-v1', 'event-v1', 'sens-ref']:
        us = real(units) if cond == 'real' else gen(units, cond)
        s = [slant(u) for u in us if slant(u) is not None]
        out[cond] = {'sd': float(np.std(s)), 'share_zero': float(np.mean([x == 0 for x in s])), 'n': len(s)}
    return out


def main():
    units = read_jsonl(RESULTS / 'analysis' / 'units_j5.jsonl')
    spread = [np.nanmean([slant(u) for u in real(units, o) if slant(u) is not None]) for o in OUTLETS]
    nonfox = [s for o, s in zip(OUTLETS, spread) if o != 'foxnews']
    margin = round(float((max(nonfox) - min(nonfox)) / 2), 3)
    out = {'created_at': now(), 'data_driven_margin': {'value': margin, 'definition': 'half the range of real outlet mean judge slant, excluding Fox'}}
    print('A'); out['party_stratum'] = party_stratum(units)
    print('B'); out['fox_within_genre'] = fox_within_genre(units)
    print('C'); out['event_arm'] = event_arm(units, margin)
    out['event_matched_auc'] = event_matched_auc(units)
    print('D'); out['period_matched'] = period_matched(units)
    print('E'); out['fox_by_identification'] = misidentified(units)
    print('F'); out['cluster_sensitivity'] = cluster_sensitivity()
    out['full_real_set'] = full_set(units)
    print('G'); out['benchmark_ci'] = benchmark_ci(units)
    print('H'); out['judge_swap_detail'] = judge_swap_detail()
    print('I'); out['judge_variance'] = judge_variance(units)
    main_res = json.loads((RESULTS / 'analysis' / 'results.json').read_text())
    out['equivalence_data_margin'] = {f'{o}_vs_neutral': tost_equivalent(main_res['rq1_main']['minus_neutral'][o]['slant']['ci90'], margin) for o in OUTLETS}
    out['holm_adjusted'] = {k: v.get('pooled_p_holm') for k, v in main_res['h1']['robust'].items()}
    write_json(RESULTS / 'analysis' / 'extra.json', out)
    print('wrote results/v2/analysis/extra.json')


if __name__ == '__main__':
    main()
