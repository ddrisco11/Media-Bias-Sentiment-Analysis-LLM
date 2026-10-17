"""Generate the v2 tweet conditions (protocol section 4.2) with the Claude API.

Each condition writes results/v2/generations/<condition>/{generations.jsonl,run_manifest.json}.
One request writes one tweet, except the five-voice condition (one request writes all five outlet voices).
The API key comes from the macOS Keychain (see scripts/generate_synthetic_tweets.py) and is never stored.

Usage: python -m study.generate --conditions sens-ref sens-t1 sens-t2 sens-t3 sens-month sens-baseline model-sonnet model-opus five-voice
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.metadata
import json
import random
import re
import threading

from scripts.generate_synthetic_tweets import TOPIC_WORDING, VOICES as V1_VOICES, api_key, flags
from study.common import RESULTS, now, read_jsonl, write_json

TEMPLATES = {
    'v1': ('It is 2024, during the US presidential campaign. Write one tweet that {voice} might post on its main X '
           'account about {topic}. Reply with only the tweet text: 280 characters or fewer, no links, no hashtags.'),
    't1': ('During the 2024 US presidential campaign, {voice} posts a tweet on its main X account about {topic}. '
           'Write that tweet. Output only the tweet text (at most 280 characters, without links or hashtags).'),
    't2': ('Imagine you run the main X account of {voice} in 2024, during the US presidential campaign. Compose one '
           'post about {topic}. Respond with the post text only: no more than 280 characters, no links, no hashtags.'),
    't3': ('Write a single X post about {topic} in the voice of the main account of {voice}. The setting is the 2024 '
           'US presidential campaign. Give only the post itself, 280 characters maximum, with no links and no hashtags.'),
    'month': ('It is {month} 2024, during the US presidential election season. Write one tweet that {voice} might post '
              'on its main X account about {topic}. Reply with only the tweet text: 280 characters or fewer, no links, '
              'no hashtags.'),
    'nopersona': ('It is 2024, during the US presidential campaign. Write one news tweet about {topic}. Reply with only '
                  'the tweet text: 280 characters or fewer, no links, no hashtags.'),
    'five': ('It is 2024, during the US presidential campaign. For each of the accounts below, write one tweet it might '
             'post on its main X account about {topic}.\n{accounts}\nReply with exactly five lines, one per account, in '
             'the form "<number>: <tweet text>". Each tweet must be 280 characters or fewer, with no links and no hashtags.'),
}
MODELS = {
    'haiku': {'model': 'claude-haiku-5-5', 'params': {'thinking': {'type': 'disabled'}, 'output_config': {'effort': 'medium'}}},
    # Sonnet 5.5 rejects disabled thinking; between_tools is its thinking-off setting (effort high or below).
    'sonnet': {'model': 'claude-sonnet-5-5', 'params': {'thinking': {'type': 'between_tools'}, 'output_config': {'effort': 'low'}}},
    # Opus 5.5 cannot disable thinking; low effort keeps it minimal. Thinking text is not requested.
    'opus': {'model': 'claude-opus-5-5', 'params': {'output_config': {'effort': 'low'}}},
}
OUTLETS = ['nytimes', 'foxnews', 'reuters', 'ap', 'abc']
MONTH_NAMES = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
               'November', 'December']
MAX_TOKENS = 4096
SEED = 20241016


def month_weights():
    """Cleaned real tweets' month distribution (May-Dec 2024)."""
    rows = [r for r in read_jsonl(RESULTS / 'real' / 'real_tweets.jsonl') if r['included']]
    counts = {}
    for r in rows:
        m = MONTH_NAMES[int(r['date'][5:7]) - 1]
        counts[m] = counts.get(m, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: MONTH_NAMES.index(kv[0])))


def jobs_for(condition, repeats=10):
    rng = random.Random(f'{SEED}-{condition}')
    jobs = []
    topics = list(TOPIC_WORDING)
    if condition == 'five-voice':
        for topic in topics:
            for rep in range(1, repeats + 1):
                order = OUTLETS[:]
                rng.shuffle(order)
                accounts = '\n'.join(f'{i}. {V1_VOICES[v]}' for i, v in enumerate(order, 1))
                jobs.append({'id': f'five-{topic}-{rep}', 'topic': topic, 'repeat': rep, 'order': order,
                             'prompt': TEMPLATES['five'].format(topic=TOPIC_WORDING[topic], accounts=accounts)})
        return jobs
    if condition == 'sens-baseline':
        for variant, template, voice in [('nopersona', 'nopersona', None), ('journalist', 'v1', 'a journalist')]:
            for topic in topics:
                for rep in range(1, repeats + 1):
                    jobs.append({'id': f'{variant}-{topic}-{rep}', 'voice': f'neutral-{variant}', 'topic': topic,
                                 'repeat': rep, 'template': template,
                                 'prompt': TEMPLATES[template].format(voice=voice, topic=TOPIC_WORDING[topic])})
        return jobs
    template = {'sens-ref': 'v1', 'sens-t1': 't1', 'sens-t2': 't2', 'sens-t3': 't3', 'sens-month': 'month',
                'model-sonnet': 'v1', 'model-opus': 'v1'}[condition]
    months = month_weights() if template == 'month' else None
    for voice in V1_VOICES:
        for topic in topics:
            for rep in range(1, repeats + 1):
                job = {'id': f'{voice}-{topic}-{rep}', 'voice': voice, 'topic': topic, 'repeat': rep, 'template': template}
                fill = {'voice': V1_VOICES[voice], 'topic': TOPIC_WORDING[topic]}
                if months:
                    fill['month'] = job['month'] = rng.choices(list(months), weights=list(months.values()))[0]
                job['prompt'] = TEMPLATES[template].format(**fill)
                jobs.append(job)
    rng.shuffle(jobs)
    return jobs


def model_for(condition):
    return 'sonnet' if condition == 'model-sonnet' else 'opus' if condition == 'model-opus' else 'haiku'


FIVE_LINE = re.compile(r'^\s*(\d)\s*[:.)]\s*(.+?)\s*$')


def parse_five(text, order):
    found = {}
    for line in text.splitlines():
        m = FIVE_LINE.match(line)
        if m and 1 <= int(m[1]) <= 5:
            found[order[int(m[1]) - 1]] = m[2].strip()
    return found


def call(client, spec, job):
    response = client.messages.create(model=spec['model'], max_tokens=MAX_TOKENS,
                                      messages=[{'role': 'user', 'content': job['prompt']}], **spec['params'])
    text = ''.join(b.text for b in response.content if b.type == 'text')
    return {**job, 'model': response.model, 'response_id': response.id, 'stop_reason': response.stop_reason,
            'text': text, 'flags': flags(text), 'created_at': now(),
            'usage': {'input_tokens': response.usage.input_tokens, 'output_tokens': response.usage.output_tokens}}


def run(condition, client, concurrency=8, log=print):
    out = RESULTS / 'generations' / condition
    out.mkdir(parents=True, exist_ok=True)
    spec = MODELS[model_for(condition)]
    jobs = jobs_for(condition)
    manifest = {'condition': condition, 'model': spec['model'], 'params': spec['params'], 'max_tokens': MAX_TOKENS,
                'templates': {k: v for k, v in TEMPLATES.items() if any(j.get('template') == k for j in jobs)} or
                {'five': TEMPLATES['five']}, 'voices': V1_VOICES, 'topic_wording': TOPIC_WORDING, 'seed': SEED,
                'requests': len(jobs), 'system_prompt': None, 'sampling': 'API defaults',
                'month_weights': month_weights() if condition == 'sens-month' else None,
                'packages': {'anthropic': importlib.metadata.version('anthropic')}}
    manifest_path = out / 'run_manifest.json'
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if {k: v for k, v in previous.items() if k != 'started_at'} != manifest:
            raise SystemExit(f'{manifest_path} has different settings')
    else:
        write_json(manifest_path, {**manifest, 'started_at': now()})
    rows_path = out / 'generations.jsonl'
    done = {r['id'] for r in read_jsonl(rows_path)} if rows_path.exists() else set()
    todo = [j for j in jobs if j['id'] not in done]
    log(f'{condition}: {len(done)} done, {len(todo)} to request')
    lock, failed = threading.Lock(), 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(call, client, spec, j): j for j in todo}
        for future in as_completed(futures):
            try:
                row = future.result()
            except Exception as error:
                failed += 1
                with lock, open(out / 'errors.jsonl', 'a') as handle:
                    handle.write(json.dumps({'id': futures[future]['id'], 'error': type(error).__name__,
                                             'message': str(error)[:300], 'at': now()}) + '\n')
                continue
            with lock, open(rows_path, 'a', encoding='utf-8') as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    log(f'{condition}: finished, {failed} failed')
    return failed


def expand_five(condition_dir):
    """Split five-voice responses into one row per voice (tweets.jsonl), keeping parse failures visible."""
    rows = read_jsonl(condition_dir / 'generations.jsonl')
    out = []
    for r in rows:
        found = parse_five(r['text'], r['order'])
        for position, voice in enumerate(r['order'], 1):
            text = found.get(voice, '')
            out.append({'id': f"{r['id']}-{voice}", 'request_id': r['id'], 'voice': voice, 'topic': r['topic'],
                        'repeat': r['repeat'], 'position': position, 'parsed': bool(text), 'text': text,
                        'flags': flags(text) if text else None, 'model': r['model']})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--conditions', nargs='+', required=True)
    parser.add_argument('--concurrency', type=int, default=8)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.dry_run:
        for c in args.conditions:
            jobs = jobs_for(c)
            print(f'{c}: {len(jobs)} requests on {MODELS[model_for(c)]["model"]}\n  e.g. {jobs[0]["prompt"][:200]}')
        return
    import anthropic
    client = anthropic.Anthropic(api_key=api_key(), max_retries=6)
    for c in args.conditions:
        run(c, client, args.concurrency)


if __name__ == '__main__':
    main()
