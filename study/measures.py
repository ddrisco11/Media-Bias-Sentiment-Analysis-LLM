"""Text classifiers used as measures, with a shared text-hash cache.

Measures:
  leaning          M1, matous-volf/political-leaning-deberta-large (the v1 classifier, also the RL reward model)
  leaning_heldout  M3, premsa/political-bias-prediction-allsides-DeBERTa (evaluation only; never a reward)
  loaded           M3b, mediabiasgroup/magpie-babe-ft (sentence-level biased/loaded wording, no direction)
  topic            GLiNER2 topic classifier with the project taxonomy

Usage: python -m study.measures --input results/v2/real/real_tweets.jsonl --measures leaning leaning_heldout loaded topic
Cached results live in results/v2/scores/<measure>.jsonl keyed by sha256 of the scored text.
"""
import argparse
import hashlib
import json

from study.common import RESULTS, read_jsonl

MODELS = {
    'leaning': {'model': 'matous-volf/political-leaning-deberta-large', 'tokenizer': 'microsoft/deberta-v3-large',
                'labels': ['left', 'center', 'right']},
    'leaning_heldout': {'model': 'premsa/political-bias-prediction-allsides-DeBERTa',
                        'tokenizer': 'premsa/political-bias-prediction-allsides-DeBERTa', 'labels': None},
    'loaded': {'model': 'mediabiasgroup/magpie-babe-ft', 'tokenizer': 'mediabiasgroup/magpie-babe-ft', 'labels': None},
}
# Label names checked against each model's config at load time (see LABEL_CHECK).
LABEL_CHECK = {
    'leaning_heldout': {'left', 'center', 'right'},
    'loaded': {'unbiased', 'biased'},
}


def text_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def device():
    import torch
    return 'mps' if torch.backends.mps.is_available() else 'cpu'


class SequenceClassifier:
    """Batched softmax probabilities with an explicit, verified label order."""

    def __init__(self, name, dev=None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        spec = MODELS[name]
        self.name, self.device, self.torch = name, dev or device(), torch
        self.tokenizer = AutoTokenizer.from_pretrained(spec['tokenizer'])
        self.model = AutoModelForSequenceClassification.from_pretrained(spec['model']).to(self.device).eval()
        id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        if spec['labels']:
            self.labels = spec['labels']
        else:
            self.labels = [normalise_label(name, id2label[i]) for i in range(len(id2label))]
            if set(self.labels) != LABEL_CHECK[name]:
                raise RuntimeError(f'{name}: unexpected labels {id2label}')

    def probabilities(self, texts, batch_size=32, max_length=512):
        out = []
        for i in range(0, len(texts), batch_size):
            batch = self.tokenizer(texts[i:i + batch_size], padding=True, truncation=True, max_length=max_length,
                                   return_tensors='pt').to(self.device)
            with self.torch.no_grad():
                p = self.model(**batch).logits.float().softmax(-1).cpu().tolist()
            out += [dict(zip(self.labels, row)) for row in p]
        return out


def normalise_label(name, raw):
    raw = raw.lower().strip()
    table = {'leaning_heldout': {'left': 'left', 'center': 'center', 'centre': 'center', 'right': 'right',
                                 'label_0': 'left', 'label_1': 'center', 'label_2': 'right'},
             'loaded': {'non-biased': 'unbiased', 'non_biased': 'unbiased', 'unbiased': 'unbiased', 'neutral': 'unbiased',
                        'biased': 'biased', 'label_0': 'unbiased', 'label_1': 'biased'}}
    if raw not in table[name]:
        raise RuntimeError(f'{name}: unknown label {raw!r}; add an explicit mapping after checking the model card')
    return table[name][raw]


def cache_path(measure):
    return RESULTS / 'scores' / f'{measure}.jsonl'


def load_cache(measure):
    path = cache_path(measure)
    return {r['text_sha256']: r for r in read_jsonl(path)} if path.exists() else {}


def score(texts, measures, batch_size=32, log=print):
    """Score unique texts not yet cached; append to each measure's cache. Returns {measure: {hash: result}}."""
    unique = {text_hash(t): t for t in texts}
    results = {}
    for measure in measures:
        cache = load_cache(measure)
        todo = [(h, t) for h, t in unique.items() if h not in cache]
        log(f'{measure}: {len(unique) - len(todo)} cached, {len(todo)} to score')
        if todo:
            new = score_measure(measure, [t for _, t in todo], batch_size)
            path = cache_path(measure)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'a', encoding='utf-8') as handle:
                for (h, _), r in zip(todo, new):
                    row = {'text_sha256': h, **r}
                    cache[h] = row
                    handle.write(json.dumps(row) + '\n')
        results[measure] = cache
    return results


def score_measure(measure, texts, batch_size):
    if measure == 'topic':
        from topic_classification import TopicClassifier
        classifier = TopicClassifier(device=device())
        return [r.to_dict() for r in classifier.classify_many(texts, batch_size=batch_size)]
    classifier = SequenceClassifier(measure)
    rows = []
    for probs in classifier.probabilities(texts, batch_size):
        rows.append({'probabilities': probs, 'label': max(probs, key=probs.get)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', nargs='+', required=True, help='JSONL files whose rows have a "text" field')
    parser.add_argument('--measures', nargs='+', default=['leaning', 'leaning_heldout', 'loaded', 'topic'])
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    texts = [r['text'] for path in args.input for r in read_jsonl(path) if r.get('text', '').strip()]
    score(texts, args.measures, args.batch_size)


if __name__ == '__main__':
    main()
