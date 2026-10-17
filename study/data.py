"""Assemble one unit table (real and generated tweets) with every measure attached.

Usage: python -m study.data   (writes results/v2/analysis/units.jsonl)
"""
import hashlib
import re

from study.common import RESULTS, ROOT, party_mentioned, read_jsonl, scoring_text, write_jsonl
from study.judge import unflip
from study.mask import mask_outlets
from study.measures import load_cache, text_hash
from study.stats import assign_clusters

# Source dataset keywords: arXiv 2411.00376 Table 5, plus "Assassination" (from 13 Jul) and "JD Vance" (from 15 Jul).
KEYWORDS = ['2024 Elections', '2024 Presidential Election', 'Biden', 'Biden2024', 'conservative', 'CPAC', 'Donald Trump', 'GOP',
            'Joe Biden and Kamala Harris', 'Joe Biden', 'Joseph Biden', 'KAG', 'MAGA', 'Nikki Haley', 'RNC', 'Ron DeSantis',
            'Snowballing', 'Trump2024', 'trumpsupporters', 'trumptrain', 'US Elections', 'thedemocrats', 'DNC', 'Kamala Harris',
            'Marianne Williamson', 'Dean Phillips', 'williamson2024', 'phillips2024', 'Democratic party', 'Republican party',
            'Third Party', 'Green Party', 'Independent Party', 'No Labels', 'RFK Jr', 'Robert F. Kennedy Jr.', 'Jill Stein',
            'Cornel West', 'ultramaga', 'voteblue2024', 'letsgobrandon', 'bidenharris2024', 'makeamericagreatagain',
            'Vivek Ramaswamy', 'Assassination', 'JD Vance']


def _kw_pattern(keyword):
    # X search semantics: every word of a multi-word keyword must appear (AND), case-insensitively.
    words = [re.escape(w.strip('.').lower()) for w in keyword.split()]
    return [re.compile(rf'(?<![a-z0-9]){w}(?![a-z0-9])') for w in words]


KW = [(k, _kw_pattern(k)) for k in KEYWORDS]


def keyword_match(text):
    t = text.lower()
    return [k for k, pats in KW if all(p.search(t) for p in pats)]


def masked_id(text):
    return hashlib.sha256(mask_outlets(text).encode()).hexdigest()[:24]


def judge_results(model, sets, task='rate'):
    out = {}
    for s in sets:
        d = RESULTS / 'judge' / model / s / task
        if not (d / 'results.jsonl').exists():
            continue
        flips = {r['item']: r['flipped'] for r in read_jsonl(d / 'items.jsonl')}
        for r in read_jsonl(d / 'results.jsonl'):
            if r.get('type') == 'succeeded' and r.get('parsed'):
                out[r['custom_id']] = unflip(r['parsed'], flips.get(r['custom_id'])) if task == 'rate' else r['parsed']
                if task == 'rate':
                    out[r['custom_id']]['flipped'] = flips.get(r['custom_id'])
    return out


def generated_sources():
    g = RESULTS / 'generations'
    return [('main-v1', ROOT / 'dataset/synthetic/main-v1/generations.jsonl'),
            ('pilot-v1', ROOT / 'dataset/synthetic/pilot-v1/generations.jsonl'),
            *[(c, g / c / 'generations.jsonl') for c in ['sens-ref', 'sens-t1', 'sens-t2', 'sens-t3', 'sens-month',
                                                         'sens-baseline', 'model-sonnet', 'model-opus', 'event-v1']],
            ('five-voice', g / 'five-voice' / 'tweets.jsonl')]


def build(cluster_threshold=0.5):
    caches = {m: load_cache(m) for m in ('leaning', 'leaning_heldout', 'loaded', 'topic', 'political')}
    judge = judge_results('claude-sonnet-5-5', ['all-v2'])
    ident = judge_results('claude-sonnet-5-5', ['id-v2'], task='identify')
    units = []
    for r in read_jsonl(RESULTS / 'real' / 'real_tweets.jsonl'):
        units.append({'id': r['key'], 'source': 'real', 'condition': 'real', 'group': r['outlet'], 'text': r['text'],
                      'included': r['included'], 'exclusions': r['exclusions'], 'period': r['period'],
                      'quote_led': r['quote_led'], 'requested_topic': None})
    for condition, path in generated_sources():
        if not path.exists():
            continue
        for r in read_jsonl(path):
            text = scoring_text(r.get('text') or '')
            if not text:
                continue
            voice = r.get('voice')
            arm = r.get('arm')
            cond = f"pilot-{arm}" if condition == 'pilot-v1' else condition
            units.append({'id': f"{cond}/{r['id']}", 'source': 'generated', 'condition': cond, 'group': voice, 'text': text,
                          'included': True, 'requested_topic': r.get('topic'), 'event': r.get('event'),
                          'flags': r.get('flags'), 'month': r.get('month'), 'position': r.get('position')})
    for u in units:
        h = text_hash(u['text'])
        for m in ('leaning', 'leaning_heldout'):
            c = caches[m].get(h)
            u[m] = c['probabilities'] if c else None
        b = caches['loaded'].get(h)
        u['loaded_biased'] = b['probabilities']['biased'] if b else None
        t = caches['topic'].get(h)
        u['predicted_topic'] = t['topic'] if t else None
        pol = caches['political'].get(h)
        u['p_political'] = pol['p_political'] if pol else None
        mid = masked_id(u['text'])
        u['judge'] = judge.get(mid)
        u['identify'] = ident.get(mid)
        u['party'] = party_mentioned(u['text'])
        u['keywords'] = keyword_match(u['text'])
    # Near-duplicate clusters within each condition x group.
    for cond in {u['condition'] for u in units}:
        subset = [u for u in units if u['condition'] == cond]
        clusters = assign_clusters(subset, 'group', cluster_threshold)
        for u in subset:
            u['cluster'] = f"{cond}|{clusters[u['id']]}"
    write_jsonl(RESULTS / 'analysis' / f'units_j{int(cluster_threshold * 10)}.jsonl', units)
    print(f'{len(units)} units; judged {sum(u["judge"] is not None for u in units)}; scored {sum(u["leaning"] is not None for u in units)}')
    return units


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=float, default=0.5)
    build(parser.parse_args().threshold)
