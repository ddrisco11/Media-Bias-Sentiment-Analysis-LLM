"""Pre-registered analyses for RQ1-RQ4 and validity checks (protocol sections 3, 5, 6, 8).

Reads results/v2/analysis/units_j5.jsonl (built by study.data) and writes results/v2/analysis/results.json.
Usage: python -m study.analyze [--B 10000]
"""
import argparse
from collections import Counter, defaultdict
import math
import random

import numpy as np

from study.common import OUTLETS, RESULTS, read_jsonl, write_json, now
from study.judge import GENRES
from study.stats import Bootstrap, holm, random_effects, tost_equivalent

TOPICS = ['president', 'elections', 'crime and justice', 'gun issues', 'foreign affairs', 'immigration', 'economy',
          'race and civil rights', 'healthcare', 'gender rights', 'reproductive rights', 'education', 'technology', 'environment']
PARTIES = ['democrat only', 'republican only', 'both', 'neither']
STANCE = {'-2': -2, '-1': -1, '0': 0, '1': 1, '2': 2}
METRICS = (['m1_left', 'm1_center', 'm1_right', 'm1_net', 'm3_net', 'loaded', 'political', 'slant', 'stance_dem', 'stance_rep',
            'dem_na', 'rep_na', 'plausibility', 'party_any', 'party_dem_only', 'party_rep_only', 'party_both', 'kw_pass', 'chars']
           + [f'genre_{g}' for g in GENRES])
IDX = {m: i for i, m in enumerate(METRICS)}
PRIMARY = ['m1_net', 'slant']
MARGIN = {'m1_net': 15.0, 'slant': 0.4}          # H1 directional / equivalence margins
ROBUST = {'m1_net': 7.5, 'slant': 0.2}           # H1 robustness: sign match and |delta| >= half the margin
CONFIRMATORY_VARIANTS = ['sens-t1', 'sens-t2', 'sens-t3', 'sens-month', 'model-sonnet']


def values(u):
    nan = float('nan')
    v = [nan] * len(METRICS)
    p = u.get('leaning')
    if p:
        lab = max(p, key=p.get)
        v[IDX['m1_left']], v[IDX['m1_center']], v[IDX['m1_right']] = (lab == 'left', lab == 'center', lab == 'right')
        v[IDX['m1_net']] = 100 * (p['right'] - p['left'])
    h = u.get('leaning_heldout')
    if h:
        v[IDX['m3_net']] = 100 * (h['right'] - h['left'])
    if u.get('loaded_biased') is not None:
        v[IDX['loaded']] = u['loaded_biased']
    if u.get('p_political') is not None:
        v[IDX['political']] = u['p_political']
    j = u.get('judge')
    if j:
        v[IDX['slant']] = j['slant']
        sd, sr = j['stance_democrats'], j['stance_republicans']
        v[IDX['stance_dem']] = STANCE.get(sd, nan)
        v[IDX['stance_rep']] = STANCE.get(sr, nan)
        v[IDX['dem_na']], v[IDX['rep_na']] = float(sd == 'NA'), float(sr == 'NA')
        v[IDX['plausibility']] = j['plausibility']
        for g in GENRES:
            v[IDX[f'genre_{g}']] = float(j['genre'] == g)
    party = u['party']
    v[IDX['party_any']] = float(party != 'neither')
    v[IDX['party_dem_only']], v[IDX['party_rep_only']], v[IDX['party_both']] = (party == 'democrat only', party == 'republican only', party == 'both')
    v[IDX['kw_pass']] = float(bool(u['keywords']))
    v[IDX['chars']] = len(u['text'])
    return np.array(v, dtype=float)


def wmean(means, keys, weights):
    """Weighted mean of stratum mean vectors, renormalising weights per metric over strata with data."""
    acc, wsum = np.zeros(len(METRICS)), np.zeros(len(METRICS))
    for k, w in zip(keys, weights):
        m = means.get(k)
        if m is None:
            continue
        ok = ~np.isnan(m)
        acc[ok] += w * m[ok]
        wsum[ok] += w
    with np.errstate(invalid='ignore', divide='ignore'):
        return acc / wsum


def fmt(res, i, j=None):
    sel = (lambda a: a[i]) if j is None else (lambda a: a[i][j])
    est, ci, ci90 = sel(res['estimate']), sel(res['ci95']), sel(res['ci90'])
    return {'estimate': float(est), 'ci95': [float(ci[0]), float(ci[1])], 'ci90': [float(ci90[0]), float(ci90[1])],
            'p_boot': float(sel(res['p_boot'])), 'se': float(sel(res['se']))}


# ---------------------------------------------------------------- RQ1: voices vs neutral, within a condition
def voice_contrasts(units, condition, weights, B, voices=OUTLETS):
    us = [u for u in units if u['condition'] == condition and u['group'] in voices + ['neutral']]
    bs = Bootstrap(us, values, lambda u: (u['group'], u['requested_topic']))
    keys = lambda v: [(v, t) for t in TOPICS]
    w = [weights[t] for t in TOPICS]
    def est(means):
        neutral = wmean(means, keys('neutral'), w)
        return np.stack([wmean(means, keys(v), w) for v in voices] + [neutral] +
                        [wmean(means, keys(v), w) - neutral for v in voices] +
                        [wmean(means, keys('reuters'), w) - wmean(means, keys('ap'), w)])
    res = bs.run(est, B)
    out = {'n': {v: sum(1 for u in us if u['group'] == v) for v in voices + ['neutral']}, 'levels': {}, 'minus_neutral': {}}
    for i, v in enumerate(voices + ['neutral']):
        out['levels'][v] = {m: fmt(res, i, IDX[m]) for m in METRICS}
    for i, v in enumerate(voices):
        out['minus_neutral'][v] = {m: fmt(res, len(voices) + 1 + i, IDX[m]) for m in METRICS}
    out['reuters_minus_ap'] = {m: fmt(res, 2 * len(voices) + 1, IDX[m]) for m in METRICS}
    return out


def h1_rules(main, variants):
    """Directional tests in main-v1 (exploratory replication), robustness across confirmatory variants, TOST, Holm."""
    out = {'directional_main_v1': {}, 'robust': {}, 'pooled': {}, 'equivalence': {}}
    ps, labels = [], []
    for v in OUTLETS:
        for m in PRIMARY:
            r = main['minus_neutral'][v][m]
            out['directional_main_v1'][f'{v}:{m}'] = {**r, 'directional': bool((r['ci95'][0] > 0 or r['ci95'][1] < 0) and abs(r['estimate']) >= MARGIN[m])}
            signs = []
            ests, ses = [], []
            for name, res in variants.items():
                if name not in CONFIRMATORY_VARIANTS:
                    continue
                d = res['minus_neutral'][v][m]
                signs.append(bool(np.sign(d['estimate']) == np.sign(r['estimate']) and abs(d['estimate']) >= ROBUST[m]))
                ests.append(d['estimate'])
                ses.append(d['se'])
            pooled = random_effects(ests, ses) if len(ests) >= 2 else None
            out['pooled'][f'{v}:{m}'] = pooled
            out['robust'][f'{v}:{m}'] = {'variants_matching': sum(signs), 'of': len(signs),
                                         'robust': bool(sum(signs) >= 4 and pooled and (pooled['ci95'][0] > 0 or pooled['ci95'][1] < 0))}
            if pooled:
                z = pooled['estimate'] / pooled['se']
                ps.append(math.erfc(abs(z) / math.sqrt(2)))
                labels.append(f'{v}:{m}')
    for lab, p in zip(labels, holm(ps)):
        out['robust'][lab]['pooled_p_holm'] = float(p)
    for m in PRIMARY:
        nyt = main['minus_neutral']['nytimes'][m]
        ra = main['reuters_minus_ap'][m]
        out['equivalence'][f'nytimes_vs_neutral:{m}'] = {'ci90': nyt['ci90'], 'equivalent': tost_equivalent(nyt['ci90'], MARGIN[m])}
        out['equivalence'][f'reuters_vs_ap:{m}'] = {'ci90': ra['ci90'], 'equivalent': tost_equivalent(ra['ci90'], MARGIN[m])}
        for name, res in variants.items():
            d = res['minus_neutral']['nytimes'][m]
            out['equivalence'][f'nytimes_vs_neutral:{m}:{name}'] = {'ci90': d['ci90'], 'equivalent': tost_equivalent(d['ci90'], MARGIN[m])}
    return out


# ---------------------------------------------------------------- RQ2: generated vs real
def reality_gap(units, gen_condition, B, keyword_only):
    real = [u for u in units if u['source'] == 'real' and u['included'] and u['group'] in OUTLETS]
    gen = [u for u in units if u['condition'] == gen_condition and u['group'] in OUTLETS]
    if keyword_only:
        gen = [u for u in gen if u['keywords']]
    strat = lambda u: (u['source'], u['group'], u['predicted_topic'], u['party'])
    bs = Bootstrap(real + gen, values, strat)
    real_w = {o: Counter((u['predicted_topic'], u['party']) for u in real if u['group'] == o) for o in OUTLETS}
    gen_cells = {o: {(u['predicted_topic'], u['party']) for u in gen if u['group'] == o} for o in OUTLETS}
    support = {}
    for o in OUTLETS:
        tot = sum(real_w[o].values())
        support[o] = {'unsupported_real_weight': sum(c for s, c in real_w[o].items() if s not in gen_cells[o]) / tot,
                      'n_real': tot, 'n_generated': sum(1 for u in gen if u['group'] == o)}
    def est(means):
        rows = []
        for o in OUTLETS:
            cells = [s for s in real_w[o] if s in gen_cells[o]]
            w = [real_w[o][s] for s in cells]
            g = wmean(means, [('generated', o) + s for s in cells], w)
            r = wmean(means, [('real', o) + s for s in cells], w)
            rows += [g - r]
            # unstandardised difference
            gu = wmean(means, [('generated', o) + s for s in gen_cells[o]], [sum(1 for u in gen if u['group'] == o and (u['predicted_topic'], u['party']) == s) for s in gen_cells[o]])
            ru = wmean(means, [('real', o) + s for s in real_w[o]], list(real_w[o].values()))
            rows += [gu - ru, gu, ru]
        return np.stack(rows)
    res = bs.run(est, B)
    out = {}
    for k, o in enumerate(OUTLETS):
        out[o] = {'support': support[o],
                  'standardised_gap': {m: fmt(res, 4 * k, IDX[m]) for m in METRICS},
                  'raw_gap': {m: fmt(res, 4 * k + 1, IDX[m]) for m in METRICS},
                  'generated_level': {m: fmt(res, 4 * k + 2, IDX[m]) for m in METRICS},
                  'real_level': {m: fmt(res, 4 * k + 3, IDX[m]) for m in METRICS}}
    return out


def event_paired(units, B, seed=20241017):
    """Event-conditioned arm: per event x outlet, mean generated (that outlet's voice) minus mean real (that outlet's
    tweets on the event); averaged over events; bootstrap over events."""
    import json
    events = {e['event']: e for e in read_jsonl(RESULTS / 'events' / 'events.jsonl')}
    real_by_key = {u['id']: u for u in units if u['source'] == 'real'}
    gen = defaultdict(list)
    for u in units:
        if u['condition'] == 'event-v1':
            gen[(u['event'], u['group'])].append(values(u))
    pairs = defaultdict(list)  # outlet -> list of (event, diff vector, gen vec, real vec)
    neutral_pairs = defaultdict(list)
    for e, ev in events.items():
        for o in OUTLETS:
            reals = [values(real_by_key[m['key']]) for m in ev['members'] if m['outlet'] == o and m['key'] in real_by_key]
            if not reals or not gen.get((e, o)):
                continue
            g, r = np.nanmean(gen[(e, o)], 0), np.nanmean(reals, 0)
            pairs[o].append((e, g - r, g, r))
            if gen.get((e, 'neutral')):
                neutral_pairs[o].append((e, g - np.nanmean(gen[(e, 'neutral')], 0)))
    rng = np.random.default_rng(seed)
    out = {}
    for o in OUTLETS:
        d = np.array([p[1] for p in pairs[o]])
        dn = np.array([p[1] for p in neutral_pairs[o]]) if neutral_pairs[o] else None
        draws = np.array([np.nanmean(d[rng.integers(0, len(d), len(d))], 0) for _ in range(B)])
        lo, hi = np.nanpercentile(draws, 2.5, 0), np.nanpercentile(draws, 97.5, 0)
        out[o] = {'n_events': len(d), 'gap': {m: {'estimate': float(np.nanmean(d[:, IDX[m]])), 'ci95': [float(lo[IDX[m]]), float(hi[IDX[m]])]}
                                             for m in METRICS},
                  'generated_level': {m: float(np.nanmean([p[2][IDX[m]] for p in pairs[o]])) for m in METRICS},
                  'real_level': {m: float(np.nanmean([p[3][IDX[m]] for p in pairs[o]])) for m in METRICS}}
        if dn is not None:
            ndraws = np.array([np.nanmean(dn[rng.integers(0, len(dn), len(dn))], 0) for _ in range(B)])
            out[o]['voice_minus_neutral'] = {m: {'estimate': float(np.nanmean(dn[:, IDX[m]])),
                                                 'ci95': [float(np.nanpercentile(ndraws[:, IDX[m]], 2.5)), float(np.nanpercentile(ndraws[:, IDX[m]], 97.5))]}
                                             for m in PRIMARY + ['m3_net', 'loaded', 'stance_dem', 'stance_rep']}
    return out


def h2(rq2_kw, events):
    out = {}
    for m in PRIMARY:
        a = rq2_kw['abc']['standardised_gap'][m]
        b = events['abc']['gap'][m]
        out[m] = {'keyword_filtered_gap': a, 'event_gap': b,
                  'supported': bool(a['ci95'][1] < 0 and b['ci95'][1] < 0)}
    out['supported_all'] = all(out[m]['supported'] for m in PRIMARY)
    return out


# ---------------------------------------------------------------- RQ3: validity
def swap_analysis(units):
    from study.measures import load_cache, text_hash
    from study.data import judge_results, masked_id
    items = read_jsonl(RESULTS / 'swap' / 'items.jsonl')
    m1 = load_cache('leaning')
    judge = {**judge_results('claude-sonnet-5-5', ['all-v2']), **judge_results('claude-sonnet-5-5', ['swap-v2'])}
    rows = []
    for it in items:
        a, b = m1.get(text_hash(it['original'])), m1.get(text_hash(it['swapped']))
        ja, jb = judge.get(masked_id(it['original'])), judge.get(masked_id(it['swapped']))
        if not a or not b:
            continue
        net = lambda p: 100 * (p['right'] - p['left'])
        lab = lambda p: max(p, key=p.get)
        neutral = bool(ja and ja['slant'] == 0 and ja['stance_democrats'] in ('0', 'NA') and ja['stance_republicans'] in ('0', 'NA'))
        rows.append({'source': it['source'], 'kind': it['kind'], 'neutral': neutral or it['kind'] == 'template',
                     'm1_delta': net(b['probabilities']) - net(a['probabilities']), 'm1_flip': lab(a['probabilities']) != lab(b['probabilities']),
                     'judge_delta': (jb['slant'] - ja['slant']) if (ja and jb) else None})
    def summ(rs):
        if not rs:
            return None
        jd = [r['judge_delta'] for r in rs if r['judge_delta'] is not None]
        return {'n': len(rs), 'm1_mean_abs_delta': float(np.mean([abs(r['m1_delta']) for r in rs])),
                'm1_flip_rate': float(np.mean([r['m1_flip'] for r in rs])),
                'judge_n': len(jd), 'judge_mean_abs_delta': float(np.mean(np.abs(jd))) if jd else None,
                'judge_change_rate': float(np.mean([d != 0 for d in jd])) if jd else None}
    out = {'neutral_all': summ([r for r in rows if r['neutral']]),
           'neutral_templates': summ([r for r in rows if r['kind'] == 'template']),
           'neutral_tweets_real': summ([r for r in rows if r['neutral'] and r['source'] == 'real']),
           'neutral_tweets_generated': summ([r for r in rows if r['neutral'] and r['kind'] == 'tweet' and r['source'] != 'real']),
           'partisan_all': summ([r for r in rows if not r['neutral']])}
    part = [r for r in rows if not r['neutral'] and r['judge_delta'] is not None]
    if len(part) > 10:
        from scipy.stats import spearmanr
        out['partisan_m1_vs_judge_delta_spearman'] = float(spearmanr([r['m1_delta'] for r in part], [r['judge_delta'] for r in part])[0])
    n = out['neutral_all']
    out['h3_name_sensitive'] = bool(n and (n['m1_mean_abs_delta'] > 20 or n['m1_flip_rate'] > 0.15))
    out['h3_judge_name_sensitive'] = bool(n and n['judge_mean_abs_delta'] is not None and n['judge_mean_abs_delta'] > 0.4)
    return out


def agreement(units):
    from scipy.stats import spearmanr
    out = {}
    for name, sel in [('real', lambda u: u['source'] == 'real' and u['included']),
                      ('generated_main', lambda u: u['condition'] == 'main-v1'),
                      ('generated_event', lambda u: u['condition'] == 'event-v1')]:
        us = [u for u in units if sel(u)]
        vs = np.array([values(u) for u in us])
        pairs = {}
        for a, b in [('m1_net', 'slant'), ('m3_net', 'slant'), ('m1_net', 'm3_net'), ('loaded', 'slant')]:
            x, y = vs[:, IDX[a]], vs[:, IDX[b]]
            if b == 'slant' and a == 'loaded':
                y = np.abs(y)
            ok = ~np.isnan(x) & ~np.isnan(y)
            pairs[f'{a}~{"|slant|" if a == "loaded" else b}'] = {'spearman': float(spearmanr(x[ok], y[ok])[0]), 'n': int(ok.sum())}
        out[name] = pairs
    return out


def judge_order_effect(units):
    """Flipped instrument assigned by text hash (as-if random). Difference in mean |slant| and mean slant, flipped minus not."""
    out = {}
    for name, sel in [('real', lambda u: u['source'] == 'real'), ('generated', lambda u: u['source'] == 'generated')]:
        js = [u['judge'] for u in units if sel(u) and u['judge']]
        f = [j['slant'] for j in js if j['flipped']]
        nf = [j['slant'] for j in js if not j['flipped']]
        out[name] = {'n_flipped': len(f), 'n_not': len(nf), 'mean_slant_diff': float(np.mean(f) - np.mean(nf)),
                     'mean_abs_slant_diff': float(np.mean(np.abs(f)) - np.mean(np.abs(nf))),
                     'se_diff': float(math.sqrt(np.var(f) / len(f) + np.var(nf) / len(nf)))}
    return out


def identification(units):
    out = {}
    for name, sel in [('real', lambda u: u['source'] == 'real' and u['included']), ('main-v1', lambda u: u['condition'] == 'main-v1'),
                      ('sens-ref', lambda u: u['condition'] == 'sens-ref'), ('event-v1', lambda u: u['condition'] == 'event-v1')]:
        us = [u for u in units if sel(u) and u['group'] in OUTLETS and u.get('identify')]
        if not us:
            continue
        conf = {o: Counter(u['identify']['answer'] for u in us if u['group'] == o) for o in OUTLETS}
        out[name] = {'n': len(us), 'accuracy': float(np.mean([u['identify']['answer'] == u['group'] for u in us])),
                     'per_outlet_accuracy': {o: conf[o][o] / max(1, sum(conf[o].values())) for o in OUTLETS},
                     'confusion': {o: dict(conf[o]) for o in OUTLETS}}
    return out


def caricature(units, main, rq2):
    """Exaggeration ratio and within-outlet spread (Cheng et al. 2023; Bisbee et al. 2024)."""
    real = [u for u in units if u['source'] == 'real' and u['included'] and u['group'] in OUTLETS]
    out = {}
    for m in PRIMARY:
        pooled_real = np.nanmean([values(u)[IDX[m]] for u in real])
        out[m] = {}
        for o in OUTLETS:
            gen_dev = main['minus_neutral'][o][m]['estimate']
            real_dev = rq2[o]['real_level'][m]['estimate'] - pooled_real
            g_sd = np.nanstd([values(u)[IDX[m]] for u in units if u['condition'] == 'main-v1' and u['group'] == o])
            r_sd = np.nanstd([values(u)[IDX[m]] for u in real if u['group'] == o])
            out[m][o] = {'generated_minus_neutral': gen_dev, 'real_minus_pooled_real': float(real_dev),
                         'exaggeration_ratio': float(gen_dev / real_dev) if abs(real_dev) > 1e-9 else None,
                         'sd_generated': float(g_sd), 'sd_real': float(r_sd), 'sd_ratio': float(g_sd / r_sd) if r_sd else None}
    return out


def individuation(units, seed=7):
    """Can a classifier tell generated (main-v1) from real tweets, per outlet? Logistic regression on sentence
    embeddings of masked text, 5-fold cross-validation grouped by near-duplicate cluster; ROC AUC."""
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from study.mask import mask_outlets
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device='cpu')
    out = {}
    for name, cond in [('main-v1', 'main-v1'), ('event-v1', 'event-v1')]:
        out[name] = {}
        for o in OUTLETS:
            real = [u for u in units if u['source'] == 'real' and u['included'] and u['group'] == o]
            gen = [u for u in units if u['condition'] == cond and u['group'] == o]
            random.Random(seed).shuffle(gen)
            gen = gen[:len(real)]
            us = real + gen
            X = model.encode([mask_outlets(u['text']) for u in us], normalize_embeddings=True, show_progress_bar=False)
            y = np.array([0] * len(real) + [1] * len(gen))
            groups = [u['cluster'] for u in us]
            scores = np.zeros(len(us))
            for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
                clf = LogisticRegression(max_iter=2000, C=1.0).fit(X[tr], y[tr])
                scores[te] = clf.predict_proba(X[te])[:, 1]
            out[name][o] = {'auc': float(roc_auc_score(y, scores)), 'n_real': len(real), 'n_generated': len(gen)}
    return out


def kitagawa(units, condition='main-v1', metric='slant'):
    """Decompose (voice - neutral) in a metric into genre-mix and within-genre components (unweighted by topic)."""
    out = {}
    def by_genre(group):
        us = [u for u in units if u['condition'] == condition and u['group'] == group and u['judge']]
        shares, means = {}, {}
        for g in GENRES:
            gs = [values(u)[IDX[metric]] for u in us if u['judge']['genre'] == g]
            shares[g] = len(gs) / len(us)
            means[g] = float(np.nanmean(gs)) if gs else None
        return shares, means
    ns, nm = by_genre('neutral')
    for o in OUTLETS:
        vs, vm = by_genre(o)
        mix = within = 0.0
        for g in GENRES:
            a = vm[g] if vm[g] is not None else (nm[g] or 0.0)
            b = nm[g] if nm[g] is not None else a
            mix += (vs[g] - ns[g]) * (a + b) / 2
            within += (a - b) * (vs[g] + ns[g]) / 2
        out[o] = {'total': mix + within, 'genre_mix': mix, 'within_genre': within, 'voice_genre_shares': vs}
    out['neutral_genre_shares'] = ns
    return out


def fightin_words(units, top=20, alpha0=1000):
    """Monroe et al. (2008) log-odds with informative Dirichlet prior: each generated voice (main-v1) vs that
    outlet's real tweets. Masked text; prior from the pooled corpus of both. Exploratory."""
    import re
    from study.mask import mask_outlets
    tok = lambda t: re.findall(r"[a-z][a-z'\-]+", mask_outlets(t).lower().replace('[outlet]', ' '))
    out = {}
    for o in OUTLETS:
        a = Counter(w for u in units if u['condition'] == 'main-v1' and u['group'] == o for w in tok(u['text']))
        b = Counter(w for u in units if u['source'] == 'real' and u['included'] and u['group'] == o for w in tok(u['text']))
        prior = a + b
        n_prior = sum(prior.values())
        na, nb = sum(a.values()), sum(b.values())
        z = {}
        for w in prior:
            aw = alpha0 * prior[w] / n_prior
            la = math.log((a[w] + aw) / (na + alpha0 - a[w] - aw))
            lb = math.log((b[w] + aw) / (nb + alpha0 - b[w] - aw))
            var = 1 / (a[w] + aw) + 1 / (b[w] + aw)
            z[w] = (la - lb) / math.sqrt(var)
        ranked = sorted(z.items(), key=lambda kv: kv[1])
        out[o] = {'overused_by_generated': [(w, round(s, 2)) for w, s in ranked[::-1][:top]],
                  'overused_by_real': [(w, round(s, 2)) for w, s in ranked[:top]]}
    return out


def real_topic_weights(units):
    c = Counter(u['predicted_topic'] for u in units if u['source'] == 'real' and u['included'] and u['predicted_topic'] in TOPICS)
    n = sum(c.values())
    return {t: c[t] / n for t in TOPICS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--B', type=int, default=10_000)
    parser.add_argument('--units', default='units_j5.jsonl')
    parser.add_argument('--skip', nargs='*', default=[])
    args = parser.parse_args()
    units = read_jsonl(RESULTS / 'analysis' / args.units)
    weights = real_topic_weights(units)
    equal = {t: 1 / len(TOPICS) for t in TOPICS}
    res = {'created_at': now(), 'B': args.B, 'units_file': args.units, 'real_topic_weights': weights}
    print('RQ1 main-v1'); res['rq1_main'] = voice_contrasts(units, 'main-v1', weights, args.B)
    res['rq4_variants'] = {}
    for c in ['sens-ref'] + CONFIRMATORY_VARIANTS + ['model-opus']:
        print('RQ4', c); res['rq4_variants'][c] = voice_contrasts(units, c, equal, args.B)
    res['h1'] = h1_rules(res['rq1_main'], res['rq4_variants'])
    print('RQ2'); res['rq2_all'] = reality_gap(units, 'main-v1', args.B, keyword_only=False)
    res['rq2_keyword'] = reality_gap(units, 'main-v1', args.B, keyword_only=True)
    print('events'); res['rq2_events'] = event_paired(units, args.B)
    res['h2'] = h2(res['rq2_keyword'], res['rq2_events'])
    print('validity'); res['swap'] = swap_analysis(units)
    res['agreement'] = agreement(units)
    res['judge_order_effect'] = judge_order_effect(units)
    res['identification'] = identification(units)
    res['caricature'] = caricature(units, res['rq1_main'], res['rq2_all'])
    res['kitagawa_slant'] = kitagawa(units, metric='slant')
    res['kitagawa_m1'] = kitagawa(units, metric='m1_net')
    res['fightin_words'] = fightin_words(units)
    if 'individuation' not in args.skip:
        print('individuation'); res['individuation'] = individuation(units)
    write_json(RESULTS / 'analysis' / 'results.json', res)
    print('wrote results/v2/analysis/results.json')


if __name__ == '__main__':
    main()
