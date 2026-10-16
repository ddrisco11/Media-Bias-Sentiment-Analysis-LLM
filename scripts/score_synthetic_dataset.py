"""Score Haiku generation runs, with input-hash-checked resumable checkpoints."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

from political_leaning import PoliticalLeaningClassifier
from political_leaning.classifier import MODEL_ID as POLITICAL_MODEL
from topic_classification import TopicClassifier, DEFAULT_TOPICS
from topic_classification.classifier import MODEL_ID as TOPIC_MODEL, TOPIC_POLICY_VERSION, MAX_CHARACTERS
from scripts.score_media_dataset import read_jsonl, write_json


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_checkpoint(path, expected):
    rows = read_jsonl(path) if path.exists() else []
    keys = [r['key'] for r in rows]
    if len(keys) != len(set(keys)) or not set(keys) <= expected:
        raise ValueError(f'Invalid checkpoint IDs: {path}')
    return {r['key']: r for r in rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', default=['main-v1', 'pilot-v1'])
    parser.add_argument('--output', type=Path, default=Path('reports/v1'))
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1: parser.error('batch-size must be positive')
    records, sources = [], {}
    for run in args.runs:
        source = Path('dataset/synthetic') / run / 'generations.jsonl'
        rows = read_jsonl(source)
        sources[str(source)] = {'sha256': sha(source), 'rows': len(rows)}
        for row in rows:
            if not isinstance(row.get('text'), str) or not row['text'].strip():
                raise ValueError(f'Empty text: {run}/{row.get("id")}')
            records.append(dict(row, key=f'{run}/{row["id"]}', run=run))
    expected = {r['key'] for r in records}
    if len(expected) != len(records): raise ValueError('Duplicate generation IDs')
    from huggingface_hub import try_to_load_from_cache
    def revision(model):
        file = try_to_load_from_cache(model, 'config.json')
        if not isinstance(file, str): raise RuntimeError(f'Model not cached: {model}')
        return Path(file).parent.name
    signature = {
        'sources': sources,
        'models': {m: revision(m) for m in (POLITICAL_MODEL, TOPIC_MODEL)},
        'topic_policy_version': TOPIC_POLICY_VERSION, 'topics': DEFAULT_TOPICS,
        'classifier_code_sha256': {str(p): sha(p) for p in map(Path, [
            'political_leaning/classifier.py', 'topic_classification/classifier.py',
            'topic_classification/topics.py', 'scripts/score_synthetic_dataset.py'])},
        'device': args.device, 'batch_size': args.batch_size,
        'python': platform.python_version(),
        'packages': {p: importlib.metadata.version(p) for p in ('torch','transformers','gliner2','sentencepiece')},
        'input_policy': 'Original generated text only; no prompts, outlet names or thinking summaries added. All nonempty responses scored, including flagged declines.',
        'truncation': {'political_tokens':512,'topic_characters':MAX_CHARACTERS},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output/'run_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != signature:
        raise ValueError('Run configuration changed; select a fresh output directory')
    write_json(manifest, signature)
    for stage, factory in [('political', PoliticalLeaningClassifier), ('topic', TopicClassifier)]:
        path=args.output/f'{stage}_scores.jsonl'
        done=checked_checkpoint(path, expected)
        pending=[r for r in records if r['key'] not in done]
        if not pending: continue
        model=factory(device=args.device)
        start=time.monotonic()
        with path.open('a') as handle:
            for offset in range(0,len(pending),args.batch_size):
                batch=pending[offset:offset+args.batch_size]
                texts=[r['text'] for r in batch]
                if stage=='political':
                    results=model.grade_many(texts,truncate=True,batch_size=args.batch_size)
                else:
                    results=model.classify_many([t[:MAX_CHARACTERS] for t in texts],batch_size=args.batch_size)
                if len(results)!=len(batch): raise RuntimeError('Result count differs')
                for row,result in zip(batch,results):
                    item=dict(key=row['key'],**result.to_dict())
                    if stage=='topic': item.update(input_characters=len(row['text']),truncated=len(row['text'])>MAX_CHARACTERS)
                    handle.write(json.dumps(item,allow_nan=False)+'\n')
                handle.flush()
                if offset%(args.batch_size*25)==0:
                    print(stage,len(done)+offset+len(batch),'/',len(records),'seconds',round(time.monotonic()-start),flush=True)
        del model
        gc.collect()
        if args.device=='mps':
            import torch
            torch.mps.empty_cache()
        print('FINISHED',stage,flush=True)
    political=checked_checkpoint(args.output/'political_scores.jsonl',expected)
    topics=checked_checkpoint(args.output/'topic_scores.jsonl',expected)
    if set(political)!=expected or set(topics)!=expected: raise ValueError('Incomplete results')
    with (args.output/'scored_generations.jsonl').open('w') as handle:
        for row in records:
            item={k:row[k] for k in ('key','run','id','arm','voice','topic','text','flags','stop_reason')}
            item.update(political_leaning=political[row['key']],topic_classification=topics[row['key']])
            handle.write(json.dumps(item,ensure_ascii=False,allow_nan=False)+'\n')
    print('COMPLETE',len(records),'synthetic responses',flush=True)


if __name__=='__main__': main()
