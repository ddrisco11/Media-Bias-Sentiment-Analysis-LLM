# Project status and handoff (updated 2026-10-17)

Branch: `experiment/v2-robustness-rl` (PR target: `main`). This file records where the v2 study stands, so work
can resume from any machine.

## Where things stand

- **Part 1 (measurement and portrayal, RQ1–RQ4):** complete apart from human validation. Results are in
  `docs/paper/paper.md`, the working paper, and all numbers come from `results/v2/analysis/*.json`.
- **Part 2 (RL calibration study, RQ5):** deferred; it will be redesigned from scratch.
  - Its code, protocol sections, partial runs and logs are preserved at git tag `archive/rl-attempt-1` and were
    removed from this branch.
  - Lessons for the redesign are in `docs/protocol_v2.md`, change log v2.8.
- **Pre-registration and every deviation:** `docs/protocol_v2.md`, change log v2.0–v2.9.

## Main findings so far (part 1)

1. The off-the-shelf leaning classifier (M1) is unreliable for real-vs-generated comparisons.
   - A trailing t.co link changes its label for 37% of tweets.
   - Swapping party names in neutral sentences moves its score by about 80 points.
   - v1's headline gaps vanish, reverse, or exist only under M1 (paper section 6.4).
2. Under a blind Claude judge, only Haiku's Fox voice differs in framing from its neutral voice. The NYT, Reuters, AP
   and ABC voices are equivalent to the neutral voice. This holds across prompt wordings, models and time anchors.
3. Against the real outlets, the model writes a different genre and a flattened politics.
   - Its non-Fox voices are rated exactly neutral even when they name the parties.
   - Its Fox voice matches real Fox's overall slant only through opinion and promotional tweets.
   - Given a real event, all voices converge on similar news reports.

## Open items (part 1)

| Item | Owner | Notes |
|---|---|---|
| **Human coding**, the validation gate | Kushaan + David | `results/v2/human/sheet_coder_A.csv` / `sheet_coder_B.csv` (500 rows each, blind, different orders), codebook in `codebook.md`. Pilot 30 rows, then freeze the codebook. Save filled sheets as `coded_A.csv` / `coded_B.csv`, then run `python -m study.reliability --human`. |
| Non-Claude judge (Qwen2.5-7B, local) | — | Paused; 0 of 480 done with the final (unflipped) instrument. Run `python -m study.judge_local --items results/v2/human/human_set.jsonl` (one GPU job at a time). A stronger non-Claude judge, such as Gemini or OpenAI, would need an API key. |
| Optional extensions | — | More outlets from the source dataset (CNN, WSJ, WaPo, NBC, CBS, USA Today); a reworded "elections" topic; a persona system prompt; a "news report" prompt variant; non-Claude generators. See paper section 9. |
| Release hygiene | authors | The repo holds the full text of real tweets; a public release should include tweet IDs only (X terms). |

## Reproducing from scratch

```sh
/opt/homebrew/bin/python3.13 -m venv .venv          # any Python 3.10+; results were produced on 3.13.13
.venv/bin/python -m pip install -r requirements-study.txt
security add-generic-password -a "$USER" -s anthropic-api-key -w   # only needed for new API calls
```

Then follow paper section 11 in order. Notes:
- **Cached outputs:** classifier scores are cached by text hash in `results/v2/scores/`, and judge results are in
  `results/v2/judge/`. Re-running uses those caches, and API batches resume rather than resubmitting.
- **Models:** they download into `~/.cache/huggingface` on first use, about 6 GB for the part-1 classifiers and
  15 GB more for the optional Qwen-7B judge.
- **GPU:** run one GPU-heavy job at a time. Two at once exhausted 64 GB of unified memory.

## Layout

- `study/`: v2 code (cleaning, generation, events, masking, measures, judges, swap test, statistics, analyses,
  figures, barrier audit, human-set builder)
- `results/v2/`: all v2 data and outputs (real tweets, generations, scores, judge results, analysis JSON, human
  sheets, external ratings)
- `docs/protocol_v2.md`: pre-registration and change log
- `docs/paper/`: working paper and figures
- `dataset/`, `reports/v1/`, `V1Results.md`: v1 data and reports (unchanged; re-analysed in paper section 6.4)

## Cost so far

About US$35 of Claude API usage, covering generation (Haiku, Sonnet, Opus) and judging (Sonnet and Opus through the
Batch API).
