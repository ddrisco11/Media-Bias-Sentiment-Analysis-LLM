"""Event-conditioned arm (protocol section 4.2): same event, every voice, compared with each outlet's real tweet on it.

1. candidates: embed cleaned real tweets (all-mpnet-base-v2); average-linkage clusters of tweets within 3 days of
   each other; keep clusters covering at least 3 outlets.
2. adjudicate: Sonnet 5.5, blind to outlet, keeps only tweets about one specific event and writes a neutral factual
   description (who/what/when; no evaluative words, outlet names or quotations).
3. generate: Haiku 5.5 writes 3 tweets per voice per event from the description (prompt event-v1).

Usage: python -m study.events candidates | adjudicate | generate
"""
import argparse
from collections import Counter
from datetime import date
import random

from study.common import OUTLETS, RESULTS, now, read_jsonl, write_json, write_jsonl
from study.llm import call_many, params
from study.mask import mask_outlets

OUT = RESULTS / 'events'
EVENT_TEMPLATE = ('It is {date}, during the US presidential campaign. Write one tweet that {voice} might post on its main '
                  'X account about this news: {event} Reply with only the tweet text: 280 characters or fewer, no links, '
                  'no hashtags.')
ADJUDICATE = """Below are news tweets posted within a few days of each other by several news accounts (account names are masked).

{tweets}

Task: decide whether several of these tweets report the SAME specific news event (one occurrence, e.g. "the first presidential debate on June 27", not a broad topic like "the economy").
- event_found: true only if at least three tweets report the same specific event.
- member_ids: the IDs of the tweets that report that event (omit tweets about other things).
- event_date: the date the event happened or was reported (YYYY-MM-DD).
- description: two or three short factual sentences describing the event: who, what, when, where. Strictly neutral: no evaluative or emotive words, no characterisation of motives, no quotations, no mention of any news organization, no opinions, nothing that is not stated in the tweets.
- topic: a few neutral words naming the subject."""
SCHEMA = {'type': 'object', 'additionalProperties': False,
          'required': ['event_found', 'member_ids', 'event_date', 'description', 'topic'],
          'properties': {'event_found': {'type': 'boolean'}, 'member_ids': {'type': 'array', 'items': {'type': 'string'}},
                         'event_date': {'type': 'string'}, 'description': {'type': 'string'}, 'topic': {'type': 'string'}}}


def candidates(threshold=0.45, max_days=3):
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import AgglomerativeClustering
    real = [r for r in read_jsonl(RESULTS / 'real' / 'real_tweets.jsonl') if r['included']]
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device='cpu')
    emb = model.encode([r['text'] for r in real], normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    dist = 1 - emb @ emb.T
    days = np.array([date.fromisoformat(r['date']).toordinal() for r in real])
    dist[np.abs(days[:, None] - days[None, :]) > max_days] = 2.0  # never merge tweets more than max_days apart
    np.fill_diagonal(dist, 0)
    labels = AgglomerativeClustering(n_clusters=None, metric='precomputed', linkage='average',
                                     distance_threshold=threshold).fit_predict(np.clip(dist, 0, 2))
    clusters = {}
    for r, l in zip(real, labels):
        clusters.setdefault(int(l), []).append(r)
    keep = [c for c in clusters.values() if len({r['outlet'] for r in c}) >= 3]
    rows = [{'cluster': f'k{i}', 'n': len(c), 'outlets': sorted({r['outlet'] for r in c}),
             'tweets': [{'key': r['key'], 'outlet': r['outlet'], 'date': r['date'], 'text': r['text']} for r in c]}
            for i, c in enumerate(sorted(keep, key=lambda c: -len({r['outlet'] for r in c})))]
    write_jsonl(OUT / 'candidates.jsonl', rows)
    write_json(OUT / 'candidates_manifest.json', {'embedding': 'sentence-transformers/all-mpnet-base-v2',
                                                 'linkage': 'average', 'distance_threshold': threshold,
                                                 'max_days': max_days, 'clusters_kept': len(rows), 'created_at': now()})
    print(f'{len(rows)} candidate clusters with >=3 outlets; sizes {Counter(len(r["outlets"]) for r in rows)}')


def adjudicate():
    cands = read_jsonl(OUT / 'candidates.jsonl')
    reqs = []
    for c in cands:
        tweets = c['tweets'][:15]
        listing = '\n'.join(f"[{i}] ({t['date']}) {mask_outlets(t['text'])}" for i, t in enumerate(tweets))
        reqs.append(params('claude-sonnet-5-5', ADJUDICATE.format(tweets=listing), SCHEMA, max_tokens=1500))
    results = call_many(reqs)
    events = []
    for c, res in zip(cands, results):
        p = res['parsed']
        if not p or not p['event_found']:
            continue
        tweets = c['tweets'][:15]
        members = [tweets[int(i)] for i in p['member_ids'] if str(i).isdigit() and int(i) < len(tweets)]
        outlets = sorted({t['outlet'] for t in members})
        if len(outlets) < 3:
            continue
        events.append({'event': f"e{len(events) + 1:03d}", 'cluster': c['cluster'], 'event_date': p['event_date'],
                       'topic': p['topic'], 'description': p['description'], 'outlets': outlets,
                       'members': members, 'adjudicator': res['model']})
    write_jsonl(OUT / 'events.jsonl', events)
    print(f'{len(events)} events confirmed from {len(cands)} candidates; outlets per event {Counter(len(e["outlets"]) for e in events)}')


def generate(repeats=3, limit=None):
    from scripts.generate_synthetic_tweets import VOICES, flags
    from study.llm import client, text_of
    events = read_jsonl(OUT / 'events.jsonl')[:limit]
    jobs = []
    for e in events:
        when = date.fromisoformat(e['event_date'][:10]).strftime('%B %-d, 2024')
        for voice in VOICES:
            for rep in range(1, repeats + 1):
                jobs.append({'id': f"{e['event']}-{voice}-{rep}", 'event': e['event'], 'voice': voice, 'repeat': rep,
                             'prompt': EVENT_TEMPLATE.format(date=when, voice=VOICES[voice], event=e['description'])})
    random.Random(20241017).shuffle(jobs)
    out = RESULTS / 'generations' / 'event-v1'
    done = {r['id'] for r in read_jsonl(out / 'generations.jsonl')} if (out / 'generations.jsonl').exists() else set()
    todo = [j for j in jobs if j['id'] not in done]
    write_json(out / 'run_manifest.json', {'condition': 'event-v1', 'model': 'claude-haiku-5-5', 'template': EVENT_TEMPLATE,
                                           'params': {'thinking': {'type': 'disabled'}, 'output_config': {'effort': 'medium'}},
                                           'events': len(events), 'repeats': repeats, 'requests': len(jobs), 'voices': VOICES})
    from concurrent.futures import ThreadPoolExecutor
    def one(j):
        m = client().messages.create(model='claude-haiku-5-5', max_tokens=4096, thinking={'type': 'disabled'},
                                     output_config={'effort': 'medium'}, messages=[{'role': 'user', 'content': j['prompt']}])
        t = text_of(m)
        return {**j, 'model': m.model, 'stop_reason': m.stop_reason, 'text': t, 'flags': flags(t), 'created_at': now(),
                'usage': {'input_tokens': m.usage.input_tokens, 'output_tokens': m.usage.output_tokens}}
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(one, todo))
    existing = read_jsonl(out / 'generations.jsonl') if (out / 'generations.jsonl').exists() else []
    write_jsonl(out / 'generations.jsonl', existing + rows)
    print(f'event-v1: {len(existing) + len(rows)} tweets for {len(events)} events')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('step', choices=['candidates', 'adjudicate', 'generate'])
    args = parser.parse_args()
    {'candidates': candidates, 'adjudicate': adjudicate, 'generate': generate}[args.step]()


if __name__ == '__main__':
    main()
