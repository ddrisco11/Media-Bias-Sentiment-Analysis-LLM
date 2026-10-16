"""Summarize a synthetic-tweet run: declines, format problems, party mentions, repetition and cost.

Usage: python -m scripts.summarize_synthetic_tweets --run dataset/synthetic/pilot-v1
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

# Haiku 5.5 list prices per million tokens (prompts under 100K tokens), as of October 2026.
PRICE_PER_MTOK = {'input': 0.10, 'output': 0.50}
REPUBLICAN = re.compile(r'\b(trump|vance|republicans?|gop|maga)\b', re.I)
DEMOCRAT = re.compile(r'\b(harris|biden|walz|democrats?|democratic)\b', re.I)


def party_mentioned(text):
    r, d = bool(REPUBLICAN.search(text)), bool(DEMOCRAT.search(text))
    return 'both' if r and d else 'republican only' if r else 'democrat only' if d else 'neither'


def declined(row):
    return row['stop_reason'] == 'refusal' or row['flags'].get('possible_decline', False)


def rate(part, whole):
    return round(part / whole, 3) if whole else None


def summarize(rows):
    cells = defaultdict(list)
    for row in rows:
        cells[(row['arm'], row['voice'])].append(row)
    by_cell = {}
    for (arm, voice), group in sorted(cells.items()):
        written = [r for r in group if not declined(r)]
        texts = [r['text'].strip().lower() for r in written]
        by_cell[f'{arm} / {voice}'] = {
            'requests': len(group),
            'declined': sum(map(declined, group)),
            'decline_rate': rate(sum(map(declined, group)), len(group)),
            'tweets_written': len(written),
            'over_280_chars': sum(r['flags']['over_280_chars'] for r in written),
            'format_problems': sum(any(r['flags'][k] for k in ('has_link', 'has_hashtag', 'possible_preface', 'quoted'))
                                   for r in written),
            'median_chars': sorted(len(t) for t in texts)[len(texts) // 2] if texts else None,
            'exact_duplicates': len(texts) - len(set(texts)),
            'party_mentioned': dict(Counter(party_mentioned(r['text']) for r in written)),
        }
    by_topic = defaultdict(Counter)
    for row in rows:
        by_topic[f"{row['arm']} / {row['topic']}"]['requests'] += 1
        by_topic[f"{row['arm']} / {row['topic']}"]['declined'] += declined(row)
    tokens = {k: sum(r['usage'][f'{k}_tokens'] for r in rows) for k in ('input', 'output')}
    return {
        'requests': len(rows),
        'declined': sum(map(declined, rows)),
        'api_refusals': sum(r['stop_reason'] == 'refusal' for r in rows),
        'by_arm_and_voice': by_cell,
        'decline_rate_by_arm_and_topic': {k: rate(v['declined'], v['requests']) for k, v in sorted(by_topic.items())},
        'tokens': tokens,
        'estimated_cost_usd': round(sum(tokens[k] * PRICE_PER_MTOK[k] for k in tokens) / 1e6, 4),
        'notes': ['Declines combine API refusals with the possible_decline text flag; check a sample by hand.',
                  'Party mentions use the same keyword lists as the real-tweet review.'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.run / 'generations.jsonl').read_text().splitlines() if line.strip()]
    summary = summarize(rows)
    (args.run / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: summary[k] for k in ('requests', 'declined', 'api_refusals', 'estimated_cost_usd')}))


if __name__ == '__main__':
    main()
