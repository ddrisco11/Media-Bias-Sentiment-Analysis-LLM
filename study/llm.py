"""Claude API helpers: a client from the Keychain key, structured (JSON-schema) calls, and resumable Message Batches.

Batch state (batch IDs and collected results) is saved under the given directory so an interrupted job resumes
without resubmitting or paying twice.
"""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

from scripts.generate_synthetic_tweets import api_key
from study.common import now, read_jsonl, write_json

_client = None


def client():
    global _client
    if _client is None:
        import anthropic
        _client = anthropic.Anthropic(api_key=api_key(), max_retries=6)
    return _client


def params(model, prompt, schema=None, max_tokens=2048, effort='low', system=None):
    p = {'model': model, 'max_tokens': max_tokens, 'messages': [{'role': 'user', 'content': prompt}]}
    if system:
        p['system'] = system
    config = {'effort': effort}
    if schema:
        config['format'] = {'type': 'json_schema', 'schema': schema}
    p['output_config'] = config
    if model == 'claude-haiku-5-5' and effort in ('low', 'medium', 'high'):
        p['thinking'] = {'type': 'disabled'}
    if model == 'claude-sonnet-5-5' and effort in ('low', 'medium', 'high'):
        p['thinking'] = {'type': 'between_tools'}
    return p


def text_of(message):
    return ''.join(b.text for b in message.content if b.type == 'text')


def call_json(p):
    message = client().messages.create(**p)
    return {'parsed': json.loads(text_of(message)) if message.stop_reason == 'end_turn' else None,
            'stop_reason': message.stop_reason, 'model': message.model,
            'usage': {'input_tokens': message.usage.input_tokens, 'output_tokens': message.usage.output_tokens}}


def call_many(param_list, concurrency=8):
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(call_json, param_list))


def run_batch(state_dir, requests, poll_seconds=30, log=print):
    """requests: {custom_id: params}. Submits only IDs without a stored result, in chunks of 10,000, waits for
    every batch to end, and returns {custom_id: result}. Results persist in state_dir/results.jsonl."""
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    results_path, batches_path = state_dir / 'results.jsonl', state_dir / 'batches.json'
    results = {r['custom_id']: r for r in read_jsonl(results_path)} if results_path.exists() else {}
    batches = json.loads(batches_path.read_text()) if batches_path.exists() else []
    pending_ids = {cid for b in batches if not b.get('collected') for cid in b['ids']}
    todo = [cid for cid in requests if cid not in results and cid not in pending_ids]
    for i in range(0, len(todo), 10000):
        chunk = todo[i:i + 10000]
        batch = client().messages.batches.create(requests=[
            Request(custom_id=cid, params=MessageCreateParamsNonStreaming(**requests[cid])) for cid in chunk])
        batches.append({'id': batch.id, 'ids': chunk, 'submitted_at': now(), 'collected': False})
        write_json(batches_path, batches)
        log(f'submitted batch {batch.id} with {len(chunk)} requests')
    for b in batches:
        if b['collected']:
            continue
        while True:
            status = client().messages.batches.retrieve(b['id'])
            if status.processing_status == 'ended':
                break
            log(f"batch {b['id']}: {status.request_counts.processing} processing, {status.request_counts.succeeded} done")
            time.sleep(poll_seconds)
        with open(results_path, 'a', encoding='utf-8') as handle:
            for item in client().messages.batches.results(b['id']):
                row = {'custom_id': item.custom_id, 'type': item.result.type}
                if item.result.type == 'succeeded':
                    m = item.result.message
                    text = text_of(m)
                    try:
                        row['parsed'] = json.loads(text) if m.stop_reason == 'end_turn' else None
                    except json.JSONDecodeError:
                        row['parsed'] = None
                    row.update({'text': text, 'stop_reason': m.stop_reason, 'model': m.model,
                                'usage': {'input_tokens': m.usage.input_tokens, 'output_tokens': m.usage.output_tokens}})
                results[item.custom_id] = row
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        b['collected'] = True
        write_json(batches_path, batches)
    failed = [cid for cid in requests if results.get(cid, {}).get('type') != 'succeeded']
    if failed:
        log(f'{len(failed)} requests did not succeed; rerun to retry them')
        # Forget failed results so the next run resubmits them.
        keep = [r for cid, r in results.items() if r.get('type') == 'succeeded']
        with open(results_path, 'w', encoding='utf-8') as handle:
            for r in keep:
                handle.write(json.dumps(r, ensure_ascii=False) + '\n')
    return results
