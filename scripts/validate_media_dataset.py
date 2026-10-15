"""Validate the completed five-outlet dataset and both classifier outputs."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from scripts.media_dataset import URL_PATTERN
from scripts.score_media_dataset import read_jsonl, input_text, tweet_id
from political_leaning.classifier import MODEL_ID as POLITICAL_MODEL
from topic_classification.classifier import MODEL_ID as TOPIC_MODEL
from topic_classification import DEFAULT_TOPICS


def validate(root=Path('dataset')):
    manifest = json.loads((root / 'manifest.json').read_text())
    raw = read_jsonl(root / 'raw_tweets.jsonl')
    selected = json.loads((root / 'selection.json').read_text())
    annotated = read_jsonl(root / 'annotated_tweets.jsonl')
    political = read_jsonl(root / 'political_scores.jsonl')
    topics = read_jsonl(root / 'topic_scores.jsonl')
    schema = json.loads((root / 'source_schema.json').read_text())
    run = json.loads((root / 'run_manifest.json').read_text())
    def require(condition, message):
        if not condition: raise ValueError(message)
    require(len(raw) == len(selected) == len(annotated) == len(political) == len(topics) == 2500,
            'Expected 2,500 rows in every data/result artifact')
    raw_by_id = {tweet_id(row): row for row in raw}
    require(len(raw_by_id) == 2500, 'Duplicate raw tweet IDs')
    expected = set(raw_by_id)
    for name, records, id_field in [('selection', selected, 'tweet_id'), ('annotated', annotated, 'analysis_tweet_id'),
                                   ('political', political, 'tweet_id'), ('topics', topics, 'tweet_id')]:
        require({str(row[id_field]) for row in records} == expected, f'{name}: tweet ID set differs')
    counts = Counter(row['account'] for row in selected)
    require(set(counts) == set(manifest['selected_accounts']) and all(v == 500 for v in counts.values()),
            'Sample is not balanced at 500 per outlet')
    columns = set(manifest['raw_columns'])
    require(columns == set().union(*(set(fields) for fields in schema.values())), 'Source columns were lost')
    for match in selected:
        row = raw_by_id[match['tweet_id']]
        require(set(row) == columns, 'Inconsistent raw columns')
        parsed = URL_PATTERN.match(row['url'])
        require(parsed and parsed[1].lower() == match['account'] and parsed[2] == match['tweet_id'],
                'Author URL mismatch')
        input_text(row)
    for row in annotated:
        require({key: row[key] for key in columns} == raw_by_id[tweet_id(row)], 'Annotation modified raw data')
    for row in political:
        require(row['model'] == POLITICAL_MODEL and row['label'] in ('left', 'center', 'right'), 'Invalid political output')
        p = row['probabilities']
        require(set(p) == {'left', 'center', 'right'} and
                all(math.isfinite(v) and 0 <= v <= 1 for v in p.values()) and
                math.isclose(sum(p.values()), 1, abs_tol=1e-5), 'Invalid political probabilities')
        require(row['confidence'] == p[row['label']], 'Political confidence mismatch')
        require(row['truncated'] == (row['input_tokens'] > 512), 'Political truncation metadata mismatch')
    for row in topics:
        require(row['model'] == TOPIC_MODEL and row['topic'] in DEFAULT_TOPICS, 'Invalid topic output')
        require(math.isfinite(row['confidence']) and 0 <= row['confidence'] <= 1, 'Invalid topic confidence')
        text, field = input_text(raw_by_id[row['tweet_id']])
        require(row['text_field'] == field and row['input_characters'] == len(text) and
                row['truncated'] == (len(text) > 1000), 'Topic input metadata mismatch')
    checksum = hashlib.sha256((root / 'raw_tweets.jsonl').read_bytes()).hexdigest()
    require(checksum == manifest['raw_sha256'] == run['raw_sha256'], 'Raw checksum mismatch')
    result = {'status': 'passed', 'tweets': len(raw), 'unique_tweet_ids': len(expected),
              'columns_preserved': len(columns), 'outlet_counts': dict(counts),
              'political_results': len(political), 'topic_results': len(topics),
              'raw_sha256': checksum}
    (root / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': validate()
