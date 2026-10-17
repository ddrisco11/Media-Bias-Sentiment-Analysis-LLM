"""Human validation set (protocol section 6): 450 stratified items plus 30 swap pairs and 20 repeats, two blind sheets.

Strata: source (real or generated) x genre group (from the primary judge) x M1 label x named party (any or none).
Each source gets 225 items, allocated across its strata in proportion to sqrt(N), with at least 3 per stratum
(capped at N). Items are drawn at random within strata, and each item's inclusion probability n/N is recorded in
the key, so prediction-powered or design-based estimators can use the labels later.

Coders see only the item number and the masked text. The key (item to source, stratum, probability, swap/repeat
links) is kept separately and is never shown to coders.
Usage: python -m study.human_set
Outputs (results/v2/human/): sheet_coder_A.csv, sheet_coder_B.csv, codebook.md, key.jsonl, human_set.jsonl
"""
import csv
import hashlib
import math
import random
from collections import defaultdict

from study.common import RESULTS, read_jsonl, write_jsonl, write_json
from study.judge import RUBRIC
from study.mask import mask_outlets
from study.swap import swap_text

GENRE_GROUP = {'news_report': 'news', 'breaking_or_results': 'news', 'poll': 'news', 'quote_led': 'news',
               'explainer_or_roundup': 'explainer', 'civic_reminder': 'civic', 'opinion_or_advocacy': 'other',
               'live_or_promo': 'other', 'other': 'other'}
PER_SOURCE, MIN_PER_STRATUM, SWAPS, REPEATS, SEED = 225, 3, 30, 20, 20241018
COLUMNS = ['item_no', 'text', 'genre', 'stance_democrats', 'stance_republicans', 'event_valence_democrats',
           'event_valence_republicans', 'slant', 'plausibility', 'notes']


def stratum(u):
    p = u['leaning']
    return (u['source'], GENRE_GROUP[u['judge']['genre']], max(p, key=p.get), 'party' if u['party'] != 'neither' else 'none')


def allocate(counts, total):
    roots = {s: math.sqrt(n) for s, n in counts.items()}
    z = sum(roots.values())
    alloc = {s: min(counts[s], max(MIN_PER_STRATUM, round(total * r / z))) for s, r in roots.items()}
    # trim or top up to hit the total, largest strata first
    order = sorted(counts, key=lambda s: -counts[s])
    while sum(alloc.values()) > total:
        for s in order:
            if alloc[s] > MIN_PER_STRATUM and sum(alloc.values()) > total:
                alloc[s] -= 1
    while sum(alloc.values()) < total:
        for s in order:
            if alloc[s] < counts[s] and sum(alloc.values()) < total:
                alloc[s] += 1
    return alloc


def main():
    rng = random.Random(SEED)
    units = read_jsonl(RESULTS / 'analysis' / 'units_j5.jsonl')
    pool = [u for u in units if u['judge'] and u['leaning'] and
            ((u['source'] == 'real' and u['included']) or u['condition'] in ('main-v1', 'event-v1', 'sens-ref'))]
    by_stratum = defaultdict(list)
    for u in pool:
        by_stratum[stratum(u)].append(u)
    chosen = []
    for source in ('real', 'generated'):
        counts = {s: len(v) for s, v in by_stratum.items() if s[0] == source}
        for s, n in allocate(counts, PER_SOURCE).items():
            for u in rng.sample(by_stratum[s], n):
                chosen.append({'unit': u['id'], 'source': source, 'condition': u['condition'], 'group': u['group'],
                               'stratum': list(s), 'stratum_N': counts[s], 'stratum_n': n, 'inclusion_prob': n / counts[s],
                               'masked_text': mask_outlets(u['text']), 'kind': 'sample', 'text_raw': u['text']})
    swappable = [c for c in chosen if swap_text(c['text_raw'])]
    for c in rng.sample(swappable, min(SWAPS, len(swappable))):
        chosen.append({**c, 'kind': 'swap', 'swap_of': c['unit'], 'masked_text': mask_outlets(swap_text(c['text_raw']))})
    for c in rng.sample([c for c in chosen if c['kind'] == 'sample'], REPEATS):
        chosen.append({**c, 'kind': 'repeat', 'repeat_of': c['unit']})
    for c in chosen:
        c['item'] = hashlib.sha256(c['masked_text'].encode()).hexdigest()[:24]
    out = RESULTS / 'human'
    key = []
    for coder in ('A', 'B'):
        order = chosen[:]
        random.Random(f'{SEED}-{coder}').shuffle(order)
        with open(out / f'sheet_coder_{coder}.csv', 'w', newline='', encoding='utf-8') as handle:
            w = csv.writer(handle)
            w.writerow(COLUMNS)
            for i, c in enumerate(order, 1):
                w.writerow([i, c['masked_text']] + [''] * (len(COLUMNS) - 2))
                key.append({'coder': coder, 'item_no': i, **{k: v for k, v in c.items() if k != 'text_raw'}})
    write_jsonl(out / 'key.jsonl', key)
    unique = {c['item']: c['masked_text'] for c in chosen}
    write_jsonl(out / 'human_set.jsonl', [{'item': k, 'masked_text': v, 'text': v} for k, v in unique.items()])
    (out / 'codebook.md').write_text(
        '# Coding instructions\n\nCode every row of your sheet independently; do not discuss items with the other coder '
        'until both sheets are complete. Some rows may look alike; code each on its own. Fill the columns using the '
        'definitions below. Use exactly these values: genre (one of the genre names), stances "-2".."2" or "NA", event '
        'valence "bad"/"neutral"/"good"/"NA", slant -3..3 (negative favours Democrats, positive favours Republicans), '
        'plausibility 1..5.\n\nPilot: code the first 30 rows, then meet to resolve misunderstandings of the instructions '
        '(not of specific items); record any clarification here, and freeze this file before coding the rest.\n\n'
        '## Definitions (identical to the automated judge)\n\n' + RUBRIC + '\n', encoding='utf-8')
    write_json(out / 'design.json', {'per_source': PER_SOURCE, 'min_per_stratum': MIN_PER_STRATUM, 'swaps': SWAPS,
                                     'repeats': REPEATS, 'seed': SEED, 'items_per_sheet': len(chosen),
                                     'strata': {str(k): len(v) for k, v in by_stratum.items()}})
    print(f'{len(chosen)} rows per sheet ({sum(c["kind"] == "sample" for c in chosen)} sampled, '
          f'{sum(c["kind"] == "swap" for c in chosen)} swaps, {sum(c["kind"] == "repeat" for c in chosen)} repeats)')


if __name__ == '__main__':
    main()
