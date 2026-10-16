# Synthetic outlet-voice tweets

Tweets written by Claude Haiku 5.5 in the voice of each outlet, for comparison with
the real tweets in `dataset/`. They are model-generated and often contain invented
facts and figures. They must never be presented or shared as real outlet posts.

Each run lives in its own directory:

- `run_manifest.json`: every setting that affects the output (model, prompt template
  and version, voices, topic wording, thinking settings, max tokens, repeats, topic
  mix, order seed, taxonomy version, package versions, start time). The script
  refuses to add rows with different settings to an existing run directory.
- `generations.jsonl`: one row per request. Each row holds the prompt and its parts,
  the model that answered, the response ID, the stop reason, the unedited tweet text,
  the thinking summary (thinking-on only), token usage, a timestamp and the flags.
- `errors.jsonl`: failed requests, retried on the next run. It is absent when
  nothing failed.
- `summary.json`: counts from `scripts/summarize_synthetic_tweets.py`.

## Procedure

1. **Prompt.** One user message, with no system prompt and no example tweets, facts or
   events:
   `It is 2024, during the US presidential campaign. Write one tweet that {voice} might
   post on its main X account about {topic}. Reply with only the tweet text: 280
   characters or fewer, no links, no hashtags.`
2. **Voices.** The five outlets by name and handle, plus "a US news organization" as
   a neutral baseline.
3. **Topics.** Each topic classifier label except "other", reworded neutrally
   (for example "reproductive rights" → "abortion"). The mapping is in the manifest.
4. **Independence.** One tweet per request, so no request sees another tweet.
5. **Settings.** Model `claude-haiku-5-5`, effort `medium`, `max_tokens` 4096, API
   default sampling (Haiku 5.5 rejects other temperature values). Thinking is either
   adaptive with summaries saved, or disabled. Nothing else differs between the two.
6. **Topic mix.** The pilot used an equal number per topic. The main run splits 500
   tweets per voice by the real tweets' topic mix ("other" excluded), then raises
   each topic to at least 10 per voice. Results are weighted back to the real mix.
   The allocation and a checksum of `dataset/topic_scores.jsonl` are in the manifest.
7. **Order.** All requests are shuffled with a fixed seed, so run time does not track
   voice or topic. Up to 8 requests run at once. The order doesn't affect outputs,
   because requests are independent.
7. **Key handling.** The API key is read at run time from the macOS Keychain item
   `anthropic-api-key` and passed only to the SDK. It is never printed, logged or
   written to disk.
8. **Flags.** Each tweet's text is checked automatically and never edited. The flags
   mark text that is empty, over 280 characters, contains a link or hashtag, is
   wrapped in quotes, has a preface, or is a possible decline (a first-person "I
   can't…"). Flags are derived from the text, so they can be recomputed when a
   pattern is corrected.
9. **Summary.** Counts of declines, flags, length, exact duplicates and the party each
   tweet names. The party keyword lists match the real-tweet review: Trump, Vance,
   Republican, GOP, MAGA versus Harris, Biden, Walz, Democrat(ic). Cost uses Haiku 5.5
   list prices of $0.10 / $0.50 per million input / output tokens.

## Run log

### pilot-v1 (2026-10-16 UTC)

- **Design.** 2 thinking settings × 6 voices × 14 topics × 5 repeats = 840 requests,
  with an equal number per topic.
- **Smoke test.** A 6-request smoke test ran first with identical settings in the same
  directory. Those 6 rows are part of the 840, and the full run skipped them.
- **Outcome.** All 840 succeeded with no errors or retries. 72,060 input and 138,091
  output tokens, about $0.08.
- **Flag corrections.** After hand-checking samples, two flag patterns were corrected
  and the flags recomputed for every row; no tweet text changed. The decline pattern
  had matched "impersonating" inside a genuine tweet. The preface pattern had missed
  "Fox News could plausibly post something like this:". Final counts: 32 declines,
  all with thinking on, and 1 preface.
- **Findings.**
  - Thinking-on declines cluster on ABC and Reuters (13 of 70 each) and on gender
    rights and race topics.
  - 95% of thinking-off tweets name no candidate or party, compared with 4% of real
    tweets.
  - The median length is about 250 characters, and 28% of thinking-off tweets exceed
    280.
- **Not yet done.** The tweets have not been scored by the leaning or topic classifiers.

### main-v1 (2026-10-16 UTC)

- **Design.** Prompt v1 unchanged, with thinking off only. The pilot showed that
  thinking on declines non-randomly; it will be revisited separately. The real topic
  mix uses 500 per voice before the floor, then at least 10 per topic: 525 per voice,
  3,150 requests. The genre was deliberately not nudged toward news reports, so the
  model's unprompted choices stay intact.
- **Smoke test.** A 6-request smoke test ran in a scratch folder outside the repo,
  not in this directory.
- **Outcome.** All 3,150 succeeded with no errors or retries, and every stop reason
  was `end_turn`. Every voice-topic quota was met. 270,036 input and 244,781 output
  tokens, about $0.15.
- **Flag correction.** One reply described Fox's coverage and then gave "Here's a
  tweet in that style:" mid-text. The preface pattern was extended to catch that,
  and flags were recomputed for both runs; no tweet text changed.
- **Counts.**
  - 0 declines, 595 tweets over 280 characters (19%), 60 with hashtags (mostly Fox,
    e.g. #FoxNews), 39 wrapped in quotes, 8 with link text and 2 prefaces.
  - At most 8 exact duplicates per voice.
- **Observations.** These are not yet analysis results.
  - No voice ever names only Republicans. Candidate mentions are both parties or
    Democrats only, often "President Biden and his challengers".
  - Haiku's 2024 is mostly Biden against Trump: Harris appears in 6% of tweets,
    against 20% of real tweets. Vance and Walz never appear.
  - The Fox voice names a party least often (21%, against 37% for AP).
  - The party keyword lists are crude; for example, "Harris County" counts as a
    Democrat mention (3 tweets).
- **Not yet done.** No classifier scoring, no five-voice set, and no hand labels.
