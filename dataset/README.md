# News tweet sample and model results

The sample contains 2,500 distinct tweet URL IDs: 500 each from The New York Times
(`@nytimes`), Fox News (`@FoxNews`), Reuters (`@Reuters`), Associated Press (`@AP`),
and ABC News (`@ABC`). The source is
[deadbirds/usc-x-24-us-election-parquet](https://huggingface.co/datasets/deadbirds/usc-x-24-us-election-parquet).

## Outlet coverage

| Organization | Unique tweets available | Selected |
| --- | ---: | ---: |
| The New York Times | 913 | 500 |
| Fox News | 2,274 | 500 |
| Reuters | 1,275 | 500 |
| Associated Press | 543 | 500 |
| ABC News | 1,273 | 500 |

## Main output

`tweet_manifest.csv` contains the organization, exact tweet ID, raw tweet text,
predicted topic, predicted political-leaning label, and that label's confidence.
The confidence is a left/center/right model score, not emotional sentiment.
These are machine-generated labels, not manually verified ground truth.

## Supporting files

- `raw_tweets.jsonl`: all 39 source columns, before classification.
- `annotated_tweets.jsonl`: the raw records plus both complete model results.
- `political_scores.jsonl` and `topic_scores.jsonl`: separate ID-keyed results.
- `selection.json`: source shard, row index, account and exact tweet ID for every selected record.
- `manifest.json`: source revisions, counts, sampling rule and raw-data checksum.
- `source_schema.json`: original column names and scalar types for each selected shard.
- `discovery_counts.json`: unique account counts across all 40,072,625 source rows.
- `run_manifest.json`: model revisions, dependency versions, taxonomy and processing limits.
- `summary.json`: model-label counts by outlet.
- `validation.json`: validation of coverage, uniqueness, CSV consistency and raw-field preservation.

## Selection and preservation

Account identity is matched case-insensitively in the tweet URL. Tweet IDs are
deduplicated, retaining the first occurrence in source-shard order. Each outlet's
500 records are selected by the lowest SHA-256 hashes of
`media-sample-v1:<tweet_id>`. This is a reproducible balanced sample of the available
source, not an exhaustive collection of the outlets' tweets. Replies and quoted
posts are retained when selected.

The source has heterogeneous schemas. JSONL preserves the union of fields and
the original scalar values without coercing all IDs to floating-point numbers.
Missing fields and non-finite floating values are represented by JSON null.
In 985 records, the source's numeric `id` differs from the exact ID in `url`.
Those raw values are retained, while URL IDs are used for sampling and joins.
Serialized fields such as `user` are retained as strings and must not be evaluated
as Python code.

Source language metadata marks 2,478 records `en`, 20 `es`, one `da`, and one `fr`.
Both models target English, so the 22 records not marked English warrant manual
review. No translation or language filtering was applied. Original `date` values
span 2021-03-31 through 2024-11-30, despite the dataset's 2024 election title;
these values were preserved rather than repaired or filtered.

The raw text is passed to both modules. The pipeline explicitly records any
512-token political-input truncation or 1,000-character topic-input truncation.
Raw text remains unchanged in the source records and CSV. See the repository
README for installation and reproduction commands.

The completed run produced both predictions for every one of the 2,500 tweets.
No tweet text was truncated by either classifier. All dataset and CSV integrity
checks passed; the results are recorded in `validation.json`.
