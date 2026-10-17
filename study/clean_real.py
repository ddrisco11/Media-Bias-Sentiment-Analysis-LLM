"""Build the real-tweet analysis set with the pre-specified exclusions (protocol section 4.1).

Every tweet is kept in the output with its exclusion reasons, so the full and cleaned sets come from one file.
Usage: python -m study.clean_real
"""
from collections import Counter
import re

from study.common import RESULTS, ROOT, read_jsonl, scoring_text, sha256, write_json, write_jsonl, now

OPINION = re.compile(r'@nytopinion|\bopinion\s*[:|]|\bop-ed\b|\bnews analysis\s*:|^\s*analysis\s*:|\bin @nytopinion', re.I)
LIVE = re.compile(r'\bwatch live\b|^\s*live\s*:|^\s*watch\s*:|\blive coverage\b|\bwatch now\b', re.I)
QUOTE_LED = re.compile(r'^\W{0,3}["“‘\']|^(\S+\s+){0,14}\S*\b(says|said)\b', re.I)
# Source IDs above 2^53 were sometimes stored as floats (rounding error up to 256), so a thread
# continuation is only declared when the conversation ID differs from the tweet ID by more than that.
ROUNDING_TOLERANCE = 2048


def period(date):
    return 'P1' if date < '2024-07-21' else 'P2' if date <= '2024-11-05' else 'P3'


def thread_continuation(row):
    cid = row.get('conversationId')
    if cid is None:
        return False
    try:
        return abs(int(float(cid)) - int(row['analysis_tweet_id'])) > ROUNDING_TOLERANCE if \
            isinstance(cid, float) or 'e' in str(cid).lower() else str(cid) != row['analysis_tweet_id']
    except ValueError:
        return False


def reasons(row, seen):
    out = []
    text = row.get('text') or row.get('rawContent') or ''
    if row.get('lang') != 'en':
        out.append('non_english')
    if row.get('in_reply_to_status_id_str') is not None or row.get('in_reply_to_screen_name'):
        out.append('reply')
    elif thread_continuation(row):
        out.append('thread_continuation')
    if str(row.get('retweetedTweet')) == 'True' or text.startswith('RT @'):
        out.append('retweet')
    if (row.get('date') or '') < '2024':
        out.append('before_2024')
    if OPINION.search(text):
        out.append('opinion_or_analysis')
    if LIVE.search(text):
        out.append('live_or_video_promo')
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
                    'outlet': row['analysis_account'], 'date': row.get('date'), 'period': period(row.get('date') or ''),
                    'lang': row.get('lang'), 'raw_text': text, 'text': scoring_text(text),
                    'quote_led': bool(QUOTE_LED.search(scoring_text(text))), 'exclusions': why, 'included': not why})
    counts = Counter(r for o in out for r in o['exclusions'])
    by_outlet = {o: {'all': sum(1 for x in out if x['outlet'] == o), 'included': sum(1 for x in out if x['outlet'] == o and x['included'])}
                 for o in sorted({x['outlet'] for x in out})}
    write_jsonl(RESULTS / 'real' / 'real_tweets.jsonl', out)
    write_json(RESULTS / 'real' / 'cleaning.json', {
        'source': str(source.relative_to(ROOT)), 'source_sha256': sha256(source), 'created_at': now(),
        'rows': len(rows), 'included': sum(o['included'] for o in out), 'exclusion_counts': dict(counts),
        'by_outlet': by_outlet, 'rules': {
            'non_english': 'source lang is not "en"', 'reply': 'in_reply_to_status_id_str or in_reply_to_screen_name set',
            'thread_continuation': f'conversationId differs from the tweet ID (by more than {ROUNDING_TOLERANCE} when stored as a float)',
            'before_2024': 'source date before 2024-01-01', 'live_or_video_promo': LIVE.pattern,
            'quote_led (flag only)': QUOTE_LED.pattern, 'period': 'P1 < 2024-07-21 <= P2 <= 2024-11-05 < P3',
            'retweet': 'retweetedTweet true or text starts with "RT @"', 'opinion_or_analysis': OPINION.pattern,
            'duplicate_within_outlet': 'same outlet and identical normalized text (links, quotes, punctuation, case removed); first by date kept'}})
    print(f"{sum(o['included'] for o in out)} of {len(out)} included; exclusions {dict(counts)}")
    print(by_outlet)


if __name__ == '__main__':
    main()
