"""Generate outlet-voice tweets with Claude Haiku 5.5, one independent request per tweet.

Each request gives only a topic and a voice (one of the five outlets, or a neutral
news organization). Rows are appended to JSONL as they finish, so an interrupted run
resumes where it stopped. The API key is read from the macOS Keychain item
`anthropic-api-key` (or ANTHROPIC_API_KEY when set) and is never printed or saved.

Usage: python -m scripts.generate_synthetic_tweets --out dataset/synthetic/pilot-v1
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import re
import subprocess
import threading

from topic_classification.topics import DEFAULT_TOPICS
from topic_classification.classifier import TOPIC_POLICY_VERSION

MODEL = 'claude-haiku-5-5'
PROMPT_VERSION = 'outlet-voice-v1'
TEMPLATE = ('It is 2024, during the US presidential campaign. Write one tweet that {voice} might post '
            'on its main X account about {topic}. Reply with only the tweet text: 280 characters or '
            'fewer, no links, no hashtags.')
VOICES = {
    'nytimes': 'The New York Times (@nytimes)',
    'foxnews': 'Fox News (@FoxNews)',
    'reuters': 'Reuters (@Reuters)',
    'ap': 'Associated Press (@AP)',
    'abc': 'ABC News (@ABC)',
    'neutral': 'a US news organization',
}
# Neutral prompt wording for each classifier label; "other" is not a topic and is not prompted.
TOPIC_WORDING = {
    'president': 'the president and the presidential candidates',
    'elections': 'voting and the election',
    'crime and justice': 'crime, courts and policing',
    'gun issues': 'guns and gun laws',
    'foreign affairs': 'foreign policy and international conflicts',
    'immigration': 'immigration and the border',
    'economy': 'the economy',
    'race and civil rights': 'race and civil rights',
    'healthcare': 'health care and public health',
    'gender rights': 'gender identity and LGBTQ issues',
    'reproductive rights': 'abortion',
    'education': 'schools and education policy',
    'technology': 'technology and AI',
    'environment': 'climate and energy policy',
}
# Both arms send the same effort and max_tokens, so thinking is the only difference.
ARMS = {
    'thinking-on': {'thinking': {'type': 'adaptive', 'display': 'summarized'}, 'output_config': {'effort': 'medium'}},
    'thinking-off': {'thinking': {'type': 'disabled'}, 'output_config': {'effort': 'medium'}},
}
MAX_TOKENS = 4096
SEED = 20241105
KEYCHAIN_SERVICE = 'anthropic-api-key'
LINK = re.compile(r'https?://|www\.|\bt\.co/', re.I)
HASHTAG = re.compile(r'(?<!\w)#\w')
PREFACE = re.compile(r'^\s*(here(\'s| is)|sure\b|tweet:)|\b(could|might) plausibly post\b|something like this:|'
                     r'\bhere(\'s| is) a (tweet|post)\b', re.I)
# Haiku sometimes declines in plain text (stop_reason end_turn) rather than with an API refusal.
# Words like "impersonating" also appear in real tweets, so only first-person refusals count.
DECLINE = re.compile(r"\bI (can['’]?t|cannot|won['’]t|am not able to)\b|\bI['’]m not able to\b", re.I)

if set(TOPIC_WORDING) != set(DEFAULT_TOPICS) - {'other'}:
    raise RuntimeError('TOPIC_WORDING must cover every classifier topic except "other"')


def slug(text):
    return re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')


def build_prompt(voice, topic):
    return TEMPLATE.format(voice=VOICES[voice], topic=TOPIC_WORDING[topic])


def uniform_allocation(repeats):
    return {topic: repeats for topic in TOPIC_WORDING}


def real_allocation(topic_counts, per_voice, min_per_topic):
    """Split per_voice tweets by the real topic mix (largest remainder), then raise small topics to the floor."""
    counts = {t: topic_counts.get(t, 0) for t in TOPIC_WORDING}
    total = sum(counts.values())
    shares = {t: per_voice * c / total for t, c in counts.items()}
    allocation = {t: int(s) for t, s in shares.items()}
    for t in sorted(shares, key=lambda t: (allocation[t] - shares[t], t))[:per_voice - sum(allocation.values())]:
        allocation[t] += 1
    return {t: max(n, min_per_topic) for t, n in allocation.items()}


def build_jobs(arms, allocation):
    """Every arm x voice x topic x repeat, in a fixed shuffled order so run time doesn't track voice."""
    jobs = [{'id': f'{arm}-{voice}-{slug(topic)}-{rep}', 'arm': arm, 'voice': voice, 'topic': topic,
             'topic_wording': TOPIC_WORDING[topic], 'repeat': rep, 'prompt': build_prompt(voice, topic)}
            for arm in arms for voice in VOICES for topic in TOPIC_WORDING
            for rep in range(1, allocation[topic] + 1)]
    random.Random(SEED).shuffle(jobs)
    return jobs


def flags(text):
    return {'empty': not text.strip(), 'over_280_chars': len(text) > 280, 'has_link': bool(LINK.search(text)),
            'has_hashtag': bool(HASHTAG.search(text)), 'possible_preface': bool(PREFACE.search(text)),
            'possible_decline': bool(DECLINE.search(text)),
            'quoted': len(text) > 1 and text[0] in '"“' and text[-1] in '"”'}


def generate(client, job):
    response = client.messages.create(model=MODEL, max_tokens=MAX_TOKENS,
                                      messages=[{'role': 'user', 'content': job['prompt']}], **ARMS[job['arm']])
    text = ''.join(b.text for b in response.content if b.type == 'text')
    thinking = ''.join(b.thinking for b in response.content if b.type == 'thinking')
    details = response.stop_details
    return {**job, 'model_requested': MODEL, 'model': response.model, 'response_id': response.id,
            'stop_reason': response.stop_reason,
            'refusal_category': getattr(details, 'category', None) if details else None,
            'text': text, 'thinking_summary': thinking or None, 'flags': flags(text),
            'usage': {'input_tokens': response.usage.input_tokens, 'output_tokens': response.usage.output_tokens},
            'created_at': datetime.now(timezone.utc).isoformat()}


def api_key():
    if os.environ.get('ANTHROPIC_API_KEY'):
        return os.environ['ANTHROPIC_API_KEY']
    found = subprocess.run(['security', 'find-generic-password', '-s', KEYCHAIN_SERVICE, '-w'],
                           capture_output=True, text=True)
    if found.returncode != 0 or not found.stdout.strip():
        raise SystemExit(f'No API key: set ANTHROPIC_API_KEY or store one in Keychain as "{KEYCHAIN_SERVICE}"')
    return found.stdout.strip()


def manifest(arms, repeats, topic_mix):
    """repeats is the per-topic count for a uniform mix; a real mix records its allocation in topic_mix instead."""
    return {'model': MODEL, 'prompt_version': PROMPT_VERSION, 'template': TEMPLATE, 'voices': VOICES,
            'topic_wording': TOPIC_WORDING, 'taxonomy_version': TOPIC_POLICY_VERSION,
            'arms': {a: ARMS[a] for a in arms}, 'max_tokens': MAX_TOKENS, 'repeats': repeats,
            'topic_mix': topic_mix, 'tweets_per_request': 1, 'system_prompt': None, 'order_seed': SEED,
            'sampling': 'API defaults; Haiku 5.5 does not accept other temperature values',
            'packages': {'anthropic': importlib.metadata.version('anthropic')}, 'python': platform.python_version()}


def run(client, out, arms, repeats=None, concurrency=8, limit=None, log=print, topic_mix='uniform', allocation=None):
    """Generate every job not yet in generations.jsonl; failures go to errors.jsonl and are retried next run."""
    out.mkdir(parents=True, exist_ok=True)
    allocation = allocation or uniform_allocation(repeats)
    settings = manifest(arms, repeats, topic_mix)
    manifest_path = out / 'run_manifest.json'
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if {k: v for k, v in previous.items() if k != 'started_at'} != settings:
            raise SystemExit(f'{manifest_path} has different settings; use a new --out directory')
    else:
        manifest_path.write_text(json.dumps({**settings, 'started_at': datetime.now(timezone.utc).isoformat()},
                                            indent=2, ensure_ascii=False) + '\n')
    rows_path, errors_path = out / 'generations.jsonl', out / 'errors.jsonl'
    done = {json.loads(line)['id'] for line in rows_path.read_text().splitlines() if line.strip()} \
        if rows_path.exists() else set()
    todo = [j for j in build_jobs(arms, allocation) if j['id'] not in done][:limit]
    log(f'{len(done)} already generated, {len(todo)} to request')
    lock, counts = threading.Lock(), {'ok': 0, 'failed': 0}
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(generate, client, job): job for job in todo}
        for future in as_completed(futures):
            job = futures[future]
            try:
                row, path, key = future.result(), rows_path, 'ok'
            except Exception as error:  # Recorded and retried on the next run; the key is never in these messages.
                row, path, key = {'id': job['id'], 'error': type(error).__name__, 'message': str(error)[:500],
                                  'at': datetime.now(timezone.utc).isoformat()}, errors_path, 'failed'
            with lock:
                with path.open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + '\n')
                counts[key] += 1
                if (counts['ok'] + counts['failed']) % 50 == 0:
                    log(f"{counts['ok']} generated, {counts['failed']} failed")
    log(f"Finished: {counts['ok']} generated, {counts['failed']} failed")
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--arms', nargs='+', choices=list(ARMS), default=list(ARMS))
    parser.add_argument('--topic-mix', choices=['uniform', 'real'], default='uniform')
    parser.add_argument('--repeats', type=int, default=5, help='Tweets per topic and voice for a uniform mix')
    parser.add_argument('--per-voice', type=int, default=500, help='Tweets per voice before the floor, for a real mix')
    parser.add_argument('--min-per-topic', type=int, default=10, help='Floor per topic and voice, for a real mix')
    parser.add_argument('--topic-source', type=Path, default=Path('dataset/topic_scores.jsonl'))
    parser.add_argument('--concurrency', type=int, default=8)
    parser.add_argument('--limit', type=int, help='Request at most this many tweets (for a smoke test)')
    parser.add_argument('--dry-run', action='store_true', help='Print the job count and one prompt; no requests')
    args = parser.parse_args()
    if min(args.repeats, args.per_voice, args.concurrency) < 1 or args.min_per_topic < 0:
        parser.error('counts must be positive')
    if args.topic_mix == 'uniform':
        repeats, topic_mix, allocation = args.repeats, 'uniform', uniform_allocation(args.repeats)
    else:
        source = args.topic_source.read_bytes()
        counts = Counter(json.loads(line)['topic'] for line in source.decode().splitlines() if line.strip())
        allocation = real_allocation(counts, args.per_voice, args.min_per_topic)
        repeats, topic_mix = None, {
            'kind': 'real pooled mix with a per-topic floor', 'source': str(args.topic_source),
            'source_sha256': hashlib.sha256(source).hexdigest(), 'source_counts': dict(sorted(counts.items())),
            'per_voice_before_floor': args.per_voice, 'min_per_topic': args.min_per_topic,
            'allocation_per_voice': allocation}
    if args.dry_run:
        jobs = build_jobs(args.arms, allocation)
        print(f'{len(jobs)} requests ({sum(allocation.values())} per voice per arm), for example:\n{jobs[0]["prompt"]}')
        return
    import anthropic
    client = anthropic.Anthropic(api_key=api_key(), max_retries=5)
    run(client, args.out, args.arms, repeats, args.concurrency, args.limit, topic_mix=topic_mix, allocation=allocation)


if __name__ == '__main__':
    main()
