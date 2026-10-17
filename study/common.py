"""Shared paths, loaders and text utilities for the v2 study."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results' / 'v2'
OUTLETS = ['nytimes', 'foxnews', 'reuters', 'ap', 'abc']
VOICES = OUTLETS + ['neutral']
OUTLET_NAMES = {'nytimes': 'New York Times', 'foxnews': 'Fox News', 'reuters': 'Reuters', 'ap': 'AP',
                'abc': 'ABC News', 'neutral': 'Neutral'}
LABELS = ['left', 'center', 'right']

URL = re.compile(r'\s*https?://\S+')
REPUBLICAN = re.compile(r'\b(trump|vance|republicans?|gop|maga)\b', re.I)
DEMOCRAT = re.compile(r'\b(harris|biden|walz|democrats?|democratic)\b', re.I)
# "Harris County" and similar place names are not party mentions.
NOT_PARTY = re.compile(r'\bHarris County\b|\bDemocratic Republic\b', re.I)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    with open(path, encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    tmp.replace(path)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def scoring_text(text):
    """Text as given to every measure: links removed, wrapping quotation marks stripped, whitespace trimmed.
    The stored raw text is never changed."""
    text = URL.sub('', text).strip()
    if len(text) > 1 and text[0] in '"“' and text[-1] in '"”':
        text = text[1:-1].strip()
    return text


def party_mentioned(text):
    text = NOT_PARTY.sub(' ', text)
    r, d = bool(REPUBLICAN.search(text)), bool(DEMOCRAT.search(text))
    return 'both' if r and d else 'republican only' if r else 'democrat only' if d else 'neither'


def shingles(text, n=3):
    words = re.findall(r"[a-z0-9']+", text.lower())
    return {' '.join(words[i:i + n]) for i in range(max(1, len(words) - n + 1))}


def near_duplicate_clusters(keys, texts, threshold=0.6):
    """Single-linkage clusters of texts whose word-3-gram Jaccard similarity is at least threshold.
    Returns {key: cluster_id}. Quadratic, fine for groups of a few thousand texts."""
    sets = [shingles(t) for t in texts]
    parent = list(range(len(texts)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            a, b = sets[i], sets[j]
            if a and b and len(a & b) / len(a | b) >= threshold:
                parent[find(i)] = find(j)
    return {k: f'c{find(i)}' for i, k in enumerate(keys)}
