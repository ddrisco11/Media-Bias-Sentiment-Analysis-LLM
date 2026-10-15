"""Run both project classifiers on raw tweets with resumable, ID-keyed outputs.

Usage: python -m scripts.score_media_dataset --dataset dataset --device mps
"""
import argparse
from collections import Counter, defaultdict
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

from scripts.media_dataset import URL_PATTERN
from political_leaning import PoliticalLeaningClassifier
from political_leaning.classifier import MODEL_ID as LEANING_MODEL
from topic_classification import TopicClassifier, DEFAULT_TOPICS
from topic_classification.classifier import MODEL_ID as TOPIC_MODEL, MAX_CHARACTERS


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def tweet_id(row):
    match = URL_PATTERN.match(row.get("url") or "")
    if not match:
        raise ValueError("Tweet URL has no canonical ID")
    return match[2]


def input_text(row):
    for field in ('text', 'rawContent'):
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value, field
    raise ValueError(f'Tweet {row["id"]} has no usable text')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('dataset'))
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--stage', choices=['all', 'political', 'topic'], default='all')
    args = parser.parse_args()
    if args.batch_size < 1: parser.error('--batch-size must be positive')
    root = args.dataset
    rows = read_jsonl(root / 'raw_tweets.jsonl')
    selection = json.loads((root / 'selection.json').read_text())
    accounts = {m['tweet_id']: m['account'] for m in selection}
    if len({tweet_id(r) for r in rows}) != len(rows):
        raise ValueError('Raw tweet IDs are not unique')
    if set(accounts) != {tweet_id(r) for r in rows}:
        raise ValueError('Selection and raw tweet IDs differ')
    texts = {tweet_id(r): input_text(r) for r in rows}
    from huggingface_hub import try_to_load_from_cache
    def cached_revision(model):
        config = try_to_load_from_cache(model, 'config.json')
        if not isinstance(config, str):
            raise RuntimeError(f'Load {model} once before scoring to resolve its revision')
        return Path(config).parent.name
    signature = {
        'raw_sha256': hashlib.sha256((root / 'raw_tweets.jsonl').read_bytes()).hexdigest(),
        'models': {m: cached_revision(m) for m in (LEANING_MODEL, TOPIC_MODEL)},
        'topics': DEFAULT_TOPICS,
        'processing': {'political_truncation_tokens': 512, 'topic_truncation_characters': MAX_CHARACTERS,
                       'text_preference': ['text', 'rawContent']},
        'device': args.device,
        'packages': {p: importlib.metadata.version(p) for p in ('torch', 'transformers', 'gliner2', 'sentencepiece')},
        'python': platform.python_version(),
    }
    run_path = root / 'run_manifest.json'
    if run_path.exists() and json.loads(run_path.read_text()) != signature:
        raise RuntimeError('Run configuration changed; use a fresh output directory')
    write_json(run_path, signature)
    for stage in ('political', 'topic'):
        if args.stage not in ('all', stage): continue
        path = root / (stage + '_scores.jsonl')
        done = {r['tweet_id'] for r in read_jsonl(path)} if path.exists() else set()
        pending = [r for r in rows if tweet_id(r) not in done]
        model = PoliticalLeaningClassifier(device=args.device) if stage == 'political' else TopicClassifier(device=args.device)
        started = time.monotonic()
        with path.open('a') as handle:
            for offset in range(0, len(pending), args.batch_size):
                batch = pending[offset:offset + args.batch_size]
                source = [texts[tweet_id(r)][0] for r in batch]
                if stage == 'political':
                    results = model.grade_many(source, truncate=True, batch_size=args.batch_size)
                else:
                    results = model.classify_many([t[:MAX_CHARACTERS] for t in source], batch_size=args.batch_size)
                for row, text, result in zip(batch, source, results):
                    item = dict(tweet_id=tweet_id(row), account=accounts[tweet_id(row)],
                                text_field=texts[tweet_id(row)][1], **result.to_dict())
                    if stage == 'topic':
                        item.update(input_characters=len(text), truncated=len(text) > MAX_CHARACTERS)
                    handle.write(json.dumps(item, allow_nan=False) + '\n')
                handle.flush()
                if offset % (args.batch_size * 10) == 0:
                    print(stage, 'completed', len(done) + min(offset + args.batch_size, len(pending)),
                          '/', len(rows), 'seconds', round(time.monotonic() - started), flush=True)
        del model
        gc.collect()
        import torch
        if args.device == 'mps': torch.mps.empty_cache()
        print('FINISHED', stage, flush=True)
    paths = [root / f'{stage}_scores.jsonl' for stage in ('political', 'topic')]
    if not all(path.exists() for path in paths): return
    political, topics = ({r['tweet_id']: r for r in read_jsonl(path)} for path in paths)
    expected = {tweet_id(row) for row in rows}
    if set(political) != expected or set(topics) != expected:
        print('One stage is incomplete; combined outputs not generated', flush=True)
        return
    summary = {}
    annotated_path = root / 'annotated_tweets.jsonl'
    with annotated_path.open('w') as handle:
        for row in rows:
            key = tweet_id(row)
            record = dict(row, analysis_tweet_id=key, analysis_account=accounts[key],
                          political_leaning=political[key], topic_classification=topics[key])
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
    for account in sorted(set(accounts.values())):
        ids = [key for key, value in accounts.items() if value == account]
        summary[account] = {
            'tweets': len(ids),
            'political_leaning': dict(Counter(political[key]['label'] for key in ids)),
            'topics': dict(Counter(topics[key]['topic'] for key in ids)),
            'political_truncated': sum(political[key]['truncated'] for key in ids),
            'topic_truncated': sum(topics[key]['truncated'] for key in ids),
        }
    write_json(root / 'summary.json', summary)
    print('COMPLETE', len(rows), 'tweets classified by both modules', flush=True)


if __name__ == '__main__': main()
