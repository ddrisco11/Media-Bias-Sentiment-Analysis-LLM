"""Build the real-tweet analysis set with the pre-specified exclusions (protocol section 4.1).

Every tweet is kept in the output with its exclusion reasons, so the full and cleaned sets come from one file.
Usage: python -m study.clean_real
"""
from collections import Counter
import re

from study.common import RESULTS, ROOT, read_jsonl, scoring_text, sha256, write_json, write_jsonl, now

OPINION = re.compile(r'@nytopinion|\bopinion\s*[:|]|\bop-ed\b|\bnews analysis\s*:|^\s*analysis\s*:|\bin @nytopinion', re.I)


def reasons(row, seen):
    out = []
    if row.get('lang') != 'en':
        out.append('non_english')
    if row.get('in_reply_to_status_id_str') is not None or row.get('in_reply_to_screen_name'):
        out.append('reply')
    if str(row.get('retweetedTweet')) == 'True' or (row.get('text') or '').startswith('RT @'):
        out.append('retweet')
    text = row.get('text') or row.get('rawContent') or ''
    if OPINION.search(text):
        out.append('opinion_or_analysis')
    key = (row['analysis_account'], re.sub(r'\W+', ' ', scoring_text(text).lower()).strip())
    if key in seen:
        out.append('duplicate_within_outlet')
    seen.add(key)
    return out


def main():
    source = ROOT / 'dataset' / 'annotated_tweets.jsonl'
    rows, seen, out = read_jsonl(source), set(), []
    for row in sorted(rows, key=lambda r: (r['analysis_account'], r['date'] or '', r['analysis_tweet_id'])):
        text = row.get('text') or row.get('rawContent') or ''
        why = reasons(row, seen)
        out.append({'key': f"real/{row['analysis_tweet_id']}", 'tweet_id': row['analysis_tweet_id'],
                    'outlet': row['analysis_account'], 'date': row.get('date'), 'lang': row.get('lang'),
                    'raw_text': text, 'text': scoring_text(text), 'exclusions': why, 'included': not why})
    counts = Counter(r for o in out for r in o['exclusions'])
    by_outlet = {o: {'all': sum(1 for x in out if x['outlet'] == o), 'included': sum(1 for x in out if x['outlet'] == o and x['included'])}
                 for o in sorted({x['outlet'] for x in out})}
    write_jsonl(RESULTS / 'real' / 'real_tweets.jsonl', out)
    write_json(RESULTS / 'real' / 'cleaning.json', {
        'source': str(source.relative_to(ROOT)), 'source_sha256': sha256(source), 'created_at': now(),
        'rows': len(rows), 'included': sum(o['included'] for o in out), 'exclusion_counts': dict(counts),
        'by_outlet': by_outlet, 'rules': {
            'non_english': 'source lang is not "en"', 'reply': 'in_reply_to_status_id_str or in_reply_to_screen_name set',
            'retweet': 'retweetedTweet true or text starts with "RT @"', 'opinion_or_analysis': OPINION.pattern,
            'duplicate_within_outlet': 'same outlet and identical normalized text (links, quotes, punctuation, case removed); first by date kept'}})
    print(f"{sum(o['included'] for o in out)} of {len(out)} included; exclusions {dict(counts)}")
    print(by_outlet)


if __name__ == '__main__':
    main()
