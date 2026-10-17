"""Validity check: does a trailing t.co link change the leaning classifier's output?

Real tweets are scored as posted (with their links) and with links removed. Generated tweets (main-v1) are scored as
written and with a synthetic t.co link appended. Any difference is a measurement artifact, since a link carries no slant.
Usage: python -m study.link_artifact
"""
from collections import Counter
import random

from study.common import RESULTS, ROOT, read_jsonl, write_json, URL
from study.measures import score, text_hash

FAKE_LINK = ' https://t.co/{}'


def summary(cache, texts):
    rows = [cache[text_hash(t)] for t in texts]
    shares = Counter(r['label'] for r in rows)
    return {'n': len(rows), **{f'share_{l}': shares[l] / len(rows) for l in ('left', 'center', 'right')},
            'net_soft': 100 * sum(r['probabilities']['right'] - r['probabilities']['left'] for r in rows) / len(rows)}


def main():
    rng = random.Random(7)
    real = [r for r in read_jsonl(RESULTS / 'real' / 'real_tweets.jsonl') if r['included'] and URL.search(r['raw_text'])]
    gen = read_jsonl(ROOT / 'dataset' / 'synthetic' / 'main-v1' / 'generations.jsonl')
    pairs = {'real_without_link': [r['text'] for r in real], 'real_with_link': [r['raw_text'].strip() for r in real]}
    gen_texts = [g['text'].strip() for g in gen if g['text'].strip()]
    pairs['generated_as_written'] = gen_texts
    pairs['generated_with_link'] = [t + FAKE_LINK.format(''.join(rng.choice('abcdefghijkmnopqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789') for _ in range(10))) for t in gen_texts]
    cache = score([t for ts in pairs.values() for t in ts], ['leaning'])['leaning']
    out = {k: summary(cache, v) for k, v in pairs.items()}
    for a, b, name in [('real_without_link', 'real_with_link', 'real'), ('generated_as_written', 'generated_with_link', 'generated')]:
        flips = sum(cache[text_hash(x)]['label'] != cache[text_hash(y)]['label'] for x, y in zip(pairs[a], pairs[b]))
        out[f'{name}_label_flip_rate'] = flips / len(pairs[a])
    write_json(RESULTS / 'validity' / 'link_artifact.json', out)
    for k, v in out.items():
        print(k, v)


if __name__ == '__main__':
    main()
