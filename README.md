# Media Bias and Topic Analysis of News Tweets

This project continues from [KuvaKodes/Media-Bias-Sentiment-Analysis](https://github.com/KuvaKodes/Media-Bias-Sentiment-Analysis).
The original study, **Quantifying Media Bias: Sentiment Analysis and Corpus
Linguistics in Televised News Discourse**, was authored by **Kushaan Vardhan**,
Department of Computational Linguistics, Georgetown University. Its Git history,
paper and plots are retained for provenance.

Current development focuses on a balanced sample of news-account tweets from
[deadbirds/usc-x-24-us-election-parquet](https://huggingface.co/datasets/deadbirds/usc-x-24-us-election-parquet):
500 unique tweets each from New York Times, Fox News, Reuters, Associated Press,
and ABC News. Raw records and classification outputs belong in `dataset/`.

The old COCA broadcast datasets and the three legacy topic-analysis folders
have been removed from this repository's current tree. They remain available in
Git history and in the linked starting project.

The current modules classify political leaning and general topic independently.

## Optional Political-Leaning Classifier

The independent `political_leaning` module grades an English passage using
[`matous-volf/political-leaning-deberta-large`](https://huggingface.co/matous-volf/political-leaning-deberta-large).
It returns `left`, `center`, or `right`, the winning softmax score (`confidence`),
and all three class probabilities. These are model estimates, not calibrated
certainty, sentiment scores, or a measure of factual accuracy. Labels reflect
U.S. political context. The model does not detect whether text is political;
the authors recommend filtering nonpolitical inputs first. Model weights are
licensed CC BY-NC 4.0; consult the linked model card for terms and attribution.

From the repository root, use Python 3.10+ in a virtual environment:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-political-leaning.txt
python -m political_leaning --text "The government should increase public spending."
python -m political_leaning --file passage.txt
cat passage.txt | python -m political_leaning --stdin
```

The first call downloads the large model and its tokenizer into Hugging Face's
normal cache outside the project; allow several GB of disk space and RAM.
Inference runs locally on CPU by default; use `--device cuda:0` for a compatible
GPU. Input text is not uploaded to an inference service or saved by this module.
Dependencies are optional and do not affect the existing sentiment scripts.

```python
from political_leaning import PoliticalLeaningClassifier

classifier = PoliticalLeaningClassifier()  # Reuse across passages.
result = classifier.grade("The government should increase public spending.")
print(result.to_dict())
```

Inputs exceeding 512 tokens (including special tokens) raise an error. To grade
only the initial portion, opt in with `--truncate` or `grade(text, truncate=True)`.
The result reports `input_tokens` and `truncated`; no automatic chunk averaging
is applied. Empty text is rejected. The CLI emits JSON to stdout and errors to
stderr with exit code 2.

Run lightweight wrapper tests without downloading model weights:

```sh
python -m unittest discover -s tests -v
```

These tests use a simulated model backend; run a CLI example after installing
dependencies to verify actual model inference on your machine.

## Full Paper

The complete write-up, including background, full methodology, results, and discussion, is available in **`KRV14FinalPaper.pdf`**.

## Optional GLiNER2 Topic Classifier

`topic_classification` assigns one general topic to short English text using
[`fastino/gliner2-base-v1`](https://huggingface.co/fastino/gliner2-base-v1), the
GLiNER 2.0 checkpoint. It is independent of the sentiment analysis scripts and
political-leaning classifier. No training dataset or API key is required.

From the repository root, with Python 3.10+:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-topics.txt
python -m topic_classification --text "Donald Trump spoke at a rally today."
python -m topic_classification --file tweet.txt
cat tweet.txt | python -m topic_classification --stdin
```

The first classification downloads model weights into Hugging Face's cache;
subsequent calls reuse them. Inference runs locally on CPU. The module does not
save or send input text to a hosted inference service. JSON output contains
`topic`, `confidence`, and `model`. Confidence is the model's score, not a
calibrated guarantee of correctness.

```python
from topic_classification import TopicClassifier

classifier = TopicClassifier()  # Reuse for multiple tweets.
result = classifier.classify("A mass shooting has prompted debate on gun laws.")
print(result.to_dict())
```

The default labels are president, gun issues, gender rights, reproductive rights,
immigration, economy, elections, foreign affairs, healthcare, environment,
crime and justice, education, race and civil rights, technology, and other.
Definitions are in `topic_classification/topics.py`. Intended examples include
Donald Trump → president, mass shootings → gun issues, and transgender rights
→ gender rights. These guide the model rather than imposing keyword rules.
Overlapping subjects can receive different labels depending on context; inspect
representative tweets before relying on labels for research.

To replace the taxonomy, pass a dictionary with at least two label/description
pairs to `TopicClassifier(topics=...)`, or save that dictionary as a JSON object
and use `--topics topics.json`. Descriptions help distinguish related categories.
The classifier selects a single topic; `other` handles subjects outside the
provided default taxonomy. It rejects blank inputs and passages over 1,000
characters, and is intended for short tweets rather than whole articles.

```sh
python -m unittest discover -s tests -v
```

Unit tests use a simulated backend and do not download model weights. The library
version in `requirements-topics.txt` is separate from the model's GLiNER 2.0
version. See the [official classification tutorial](https://github.com/fastino-ai/GLiNER2/blob/main/tutorial/1-classification.md)
for schema and confidence-score details.

An optional real-model smoke test checks the three representative topics above:

```sh
RUN_GLINER_MODEL_TESTS=1 python -m unittest discover -s tests -p test_topics_model.py -v
```

These three examples passed with the actual checkpoint during implementation;
they are smoke checks, not an accuracy benchmark.

## Reproducing the Tweet Dataset

The qualifying main accounts are `@nytimes`, `@FoxNews`, `@Reuters`, `@AP`, and
`@ABC`. `dataset/discovery_counts.json` records distinct URL tweet IDs counted
across all 834 source shards (40,072,625 rows), including other candidate outlets.

Install the combined dependencies, then run from the repository root:

```sh
python -m pip install -r requirements-dataset.txt
python -m scripts.media_dataset --cache /path/outside/repo/media-cache
python -m scripts.media_dataset --cache /path/outside/repo/media-cache --import-sample
python -m scripts.score_media_dataset --dataset dataset --device cpu
```

Use `--device mps` for Apple Silicon or `--device cuda:0` for an available CUDA
GPU. Both models must be loaded once before scoring so their cached revisions
can be recorded; the module usage examples above perform that initialization.

Discovery reads only URL columns. Import retrieves the selected source shards
because the viewer API cannot preserve their heterogeneous schemas. Only selected
records are retained locally. Sampling is deterministic: deduplicate by tweet ID,
then choose the 500 lowest SHA-256 hashes of `media-sample-v1:<tweet_id>` per account.
This samples the available election-related dataset, not each outlet's complete
posting history. Replies and quoted posts are not separately excluded.

Raw JSONL retains the union of source fields and original ID values; absent fields
and non-finite floating values become JSON null. Source field types are recorded
per shard. Treat fields such as `user` as untrusted strings, not executable Python.
The dataset's source card does not declare a license; source content and the
political model's CC BY-NC 4.0 terms should be considered before redistribution.

Scoring prefers `text`, falling back to `rawContent` only when text is missing or
blank. Political inputs may be truncated to 512 tokens and topic inputs to 1,000
characters; outputs explicitly flag either case. The raw records remain intact.
Classifier scores are not calibrated certainty or factual-accuracy ratings.

Some original numeric `id` values were rounded upstream. The exact tweet ID parsed
from `url` is used for sampling and joining results; raw IDs are never overwritten.

### CSV Manifest

`dataset/tweet_manifest.csv` contains one row per sampled tweet with these columns:
`news_organization`, `tweet_id`, `raw_tweet`, `topic`, `political_leaning_label`,
and `political_leaning_confidence`. The confidence is the winning left/center/right
class score from the existing political-leaning module, not positive/negative
emotional sentiment. Topic and leaning labels are model predictions, not manually
verified ground-truth annotations. CSV quoting preserves commas, quotes and line
breaks in raw tweet text. Run `python -m scripts.validate_media_dataset` to check
coverage, balance, raw-field preservation and consistency between CSV and JSONL.
