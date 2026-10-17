# Study protocol v2.1: expected versus real political framing of news tweets

Status: v2.0 was written before any v2 data. v2.1 revises it after three independent
pre-data reviews: media-bias methodology, statistics and measurement, and RL and
reward hacking. Nothing here was changed after looking at v2 data; later changes go
in the change log and are labelled exploratory.

## 1. Background: what v1 found and why v2 exists

v1 compared 2,500 real tweets from five outlets (NYT, Fox News, Reuters, AP, ABC)
with 3,150 tweets that Claude Haiku 5.5 wrote "as" each outlet. A neutral "US news
organization" voice was the baseline. Each prompt gave only a topic and a voice, and
one tweet was written per request. Both sets were scored by a left/center/right
DeBERTa classifier (M1).

v1 found large differences between voices. It also showed that M1 is fragile:
- M1 tracks which party a tweet names, and labels voting reminders "left".
- The generated tweets are a different genre: issue explainers and reminders. They
  rarely name candidates, and they are set in the Biden-versus-Trump period.
- Real-versus-generated gaps changed sign when tweets were matched on party mention.

v2 asks which conclusions survive better measurement. It then asks whether
optimising a model against this kind of bias proxy calibrates it or games the proxy.

## 2. Constructs and claims

M1 measures **partisan-coded language as seen by one classifier**, not outlet bias.
v2 reports three constructs separately:

- **Agenda:** what is talked about. This covers requested and predicted topic, which
  actors and parties are named, and genre.
- **Tone toward actors:** a blind judge's stance toward Democrats and Republicans,
  with "not mentioned" kept separate from 0.
- **Framing:** a blind judge's slant on a 7-point scale, separated from event
  valence. Framing is isolated most cleanly in the event-conditioned arm (section 4.2).

The claim is always "how an LLM portrays an outlet compared with what the outlet
posted", never "how biased the outlet is". The real distribution is descriptive, not
a normative target.

## 3. Research questions and pre-specified rules

- **RQ1 Expectation.** How do Haiku's outlet voices differ from its neutral voice
  (agenda, tone, framing)?
- **RQ2 Reality gap.** How do the voices differ from the outlets' real tweets under
  symmetric selection and matched comparisons?
- **RQ3 Measurement validity.** Is M1 valid on these texts? Do the measures agree?
- **RQ4 Robustness.** Do RQ1 conclusions hold across wording, time anchor, generator
  model, and one versus five voices per request?
- **RQ5 Optimising a bias proxy.** *(Deferred; see change log v2.8.)* Does RL against M1 calibrate an open model's outlet
  voices toward the real outlets, judged by measures it was not trained on? Or does
  it reward-hack M1?

Decision rules, fixed before data:

- **H1 (RQ1/RQ4), directional.** A voice-minus-neutral difference Δ is "directional" in
  main-v1 if its 95% interval excludes 0 and |Δ| ≥ 15 points (judge: ≥ 0.4 on the
  7-point slant). It is "robust" if, in at least 4 of these 5 named variants (sens-t1,
  sens-t2, sens-t3, sens-month, model-sonnet), the sign matches and |Δ| ≥ 7.5, and a
  random-effects estimate pooled over the variants excludes 0.
  - Other variants (model-opus, model-qwen, five-voice, baselines) are exploratory.
  - The main-v1 ordering was seen in v1, so H1 counts as confirmed only on the new runs.
  - Holm correction across 5 voices × 2 measures (M1, judge).
- **H1-eq, equivalence.** A pair is "equivalent" (e.g. NYT versus neutral) if the 90%
  interval lies within ±15 points (judge ±0.4), tested with TOST.
- **H2 (RQ2).** "Haiku portrays ABC as further left than ABC's real tweets" is supported
  only if every condition holds:
  - the judge's slant gap and M1's gap are both below 0 with 95% intervals excluding 0;
  - this holds in the keyword-filtered comparison and in the event-conditioned arm;
  - any stratum with fewer than 40 near-duplicate clusters is reported as
    inconclusive.
- **H3, name-sensitivity.** Measured on neutral items only: 60 built templates with a
  party or candidate slot, plus real and generated items that the judge rates slant 0
  with both stances 0 or N/A. M1 is "name-sensitive" if swapping parties gives a mean
  |Δ| > 20 points or a label-flip rate > 15%. On partisan items, report how well M1's
  change agrees with the judge's change. Name-sensitivity promotes the judge to
  co-primary.
- **H4 (RQ5).** *(Deferred; see change log v2.8.)*
  - **Calibration counts as real if** the gap to the real outlet under the judge (M2)
    and under the held-out classifier (M3) shrinks, relative to the format-only
    control, by at least 50% of M1's relative shrinkage (transfer ratio ≥ 0.5). The
    95% intervals must exclude 0 in at least 2 of 3 seeds, pooled over outlets, at the
    pre-specified final checkpoint.
  - **Reward hacking is flagged if** any of these hold:
    - the transfer ratio is below 0.5;
    - party-mention rates overshoot the real outlet's rate by more than 10 points;
    - distinct-2 falls by more than 20%;
    - the near-duplicate rate rises by more than 10 points;
    - the M1 gain shrinks by more than half when names are masked.

## 4. Data

### 4.1 Real tweets
There are 2,500 tweets, 500 per outlet. The cleaned set excludes, with counts:
- non-English tweets;
- replies (reply fields set) and thread continuations (`conversationId` ≠ `id`);
- retweets;
- opinion and analysis promotions (`@nytopinion`, `Opinion:`, `Op-Ed`,
  `News Analysis:`, `Analysis:`);
- live and video promotions (`WATCH LIVE`, `LIVE:`, `WATCH:`);
- tweets dated before 2024;
- within-outlet duplicates.

Quote-led tweets are flagged, not dropped, and reported separately. Quote-led means a
quotation mark, or a "says"/"said" attribution, within the first 15 words.

Each tweet is assigned a period: P1 before 21 Jul 2024 (Biden as candidate), P2 from
21 Jul to 5 Nov, P3 after 5 Nov. The full set is a sensitivity check.

### 4.2 Generated tweets
| Run | Generator | Variant | Design | Status |
|---|---|---|---|---|
| main-v1 | Haiku 5.5, thinking off | prompt v1 | real topic mix, 525/voice | exists; exploratory for H1 |
| pilot-v1 | Haiku 5.5 | thinking on/off | uniform, 70/voice/arm | exists |
| sens-ref | Haiku 5.5 | prompt v1, re-run the same day as the variants | uniform 14 topics × 10 = 140/voice | confirmatory reference |
| sens-t1..t3 | Haiku 5.5 | three reworded templates | uniform 140/voice | confirmatory |
| sens-month | Haiku 5.5 | "It is {Month} 2024"; months drawn from the cleaned real date mix | uniform 140/voice | confirmatory |
| model-sonnet | Sonnet 5.5, low effort | prompt v1 | uniform 140/voice | confirmatory |
| model-opus | Opus 5.5, low effort (thinking cannot be disabled) | prompt v1 | uniform 140/voice | exploratory |
| model-qwen | Qwen2.5 (the RL base model), local | prompt v1 | uniform 140/voice | exploratory; RL base |
| sens-baseline | Haiku 5.5 | neutral voice only: no persona; "a journalist" | 140 each | exploratory |
| five-voice | Haiku 5.5 | all five outlets in one request, randomized order | 14 topics × 10 | exploratory |
| event-v1 | Haiku 5.5 | **event-conditioned**: "Write the tweet {voice} might post about this news: {event}" | ~50 events × 6 voices × 3 | confirmatory for H2 |

Event-conditioned arm:
- Events are clusters of real tweets that at least 3 outlets posted about within 3
  days of each other.
- Each event's description is written by Sonnet from all the outlets' tweets on it. It
  states facts only: who, what, when. It has no evaluative wording, outlet names or
  quotes.
- Each voice is then compared with that outlet's real tweet on the same event, which
  holds event, time and genre fixed.

### 4.3 Symmetric selection
The real tweets passed the source dataset's keyword filter: 44 tracked keywords (arXiv
2411.00376, Table 5), plus "Assassination" (from 13 Jul) and "JD Vance" (from 15 Jul).
The same filter is applied to the generated tweets. RQ2 is reported on keyword-passing
generated tweets, with all tweets as a sensitivity check. The keyword pass rate is
itself reported as an agenda outcome.

## 5. Measures

- **M1:** `matous-volf/political-leaning-deberta-large`. This is the v1 classifier and
  the RL reward. It reports label shares (primary) and the net score 100·(right share −
  left share), soft version (secondary).
- **M2, primary judge:** Claude Sonnet 5.5, blind to source. Its input is masked:
  - links removed;
  - outlet names, handles and self-references (e.g. "our team", "Read more on abc
    news", "#FoxNews") replaced with "[OUTLET]";
  - wrapping quotation marks removed.

  It produces structured output with these items:
  - **Stance toward Democrats** and **stance toward Republicans:** how the author, not
    the people quoted, portrays each party. −2 clearly unfavourable (loaded wording,
    selective emphasis, unbalanced attribution), −1 mildly unfavourable, 0
    even-handed, +1 mildly favourable, +2 clearly favourable, or N/A if the party is
    neither mentioned nor implied.
  - **Slant** from −3 to +3. The instruction: judge framing, not the news itself. A
    factual report of an event that is bad for one party is 0 unless wording,
    selection or omission tilts it. Each point has an anchor example.
  - **Event valence** for each party: is the event itself good or bad for it?
  - **Genre:** news report, breaking/results, poll, explainer/issue roundup,
    voting/civic reminder, opinion/advocacy, live/video promo, quote-led, other.
  - **Plausibility:** could a major US news outlet plausibly post this verbatim? (1–5)

  On a random half of items, the stance-field order and the slant direction are
  swapped and mapped back afterwards, to check for order and anchor effects.
- **M2b, second judge:** Claude Opus 5.5 on at least 300 items, including the
  human-label set.
- **M2c, non-Claude judge:** Qwen2.5-7B-Instruct run locally on the human-label set.
  It partly addresses the concern that a Claude judge shares the generator's
  stereotypes.
- **Judge test-retest:** 200 items re-judged in a separate batch.
- **M3, held-out classifier:** `premsa/political-bias-prediction-allsides-DeBERTa`.
  - It is never used as a reward.
  - Its training data (Article Bias Prediction / AllSides) partly overlaps M1's, so it
    is only partly independent.
  - It must be shown to separate Fox from NYT on real tweets before it is used for
    transfer claims.
- **M3b, loaded wording:** `mediabiasgroup/magpie-babe-ft`, a classifier of biased
  wording with no direction, trained on expert-annotated sentences.
- **M4, word-level:** log-odds with an informative Dirichlet prior, comparing each
  generated voice with the real outlet within topic and period. Top 20 terms; exploratory.
- **M5, descriptive:**
  - named parties, candidates and periods;
  - keyword-filter pass;
  - length and format flags;
  - distinct-n diversity;
  - near-duplicate clusters.
- **M6, outlet identification:** the judge is given a masked tweet and the five
  outlets, and picks one. Accuracy on real tweets against generated tweets measures
  how stereotyped the generated voices are, without any leaning model.
- **External reference:** each outlet's published rating (AllSides, plus an audience-
  based measure where one is available). These are compared with the rank order of the
  generated voices and of the real tweets.

## 6. Human validation (prepared; the gate for judge-based claims)

The human-label set has 450 items, stratified by source (real or generated), genre,
M1 label and named party, with real "center" items oversampled.
- It includes 30 party-swap pairs and 20 repeated items.
- Coders use the judge's instrument with the same masking.
- Every item is coded by two people. The codebook is piloted on 30 items, then frozen.

Agreement is Krippendorff's α: ordinal for slant and stances, nominal for genre.
- α ≥ 0.80: the labels count as gold.
- 0.667–0.80: tentative.
- Below 0.667: not usable as gold.

Gwet's AC2 is reported as well.

Validation against these labels:
- M1, per genre: confusion matrix and macro-F1 (mapping slant ≤ −1 / 0 / ≥ +1);
  Spearman correlation of M1's net score with human slant; ECE.
- The judge: judge–human α should fall inside the human–human interval.

**The study's authors have not yet produced these labels. Until they do, every
judge-based conclusion is provisional, and the paper says so.**

## 7. RL study (RQ5): deferred (change log v2.8)

*The RL design below is kept for the record. Its code, partial runs and logs are preserved at git tag
`archive/rl-attempt-1` and are not on the active branch.*

- **Policy:**
  - Qwen2.5-Instruct with a LoRA adapter (r 16, α 32, dropout 0) on q, k, v, o, gate,
    up and down projections. bf16 base weights, fp32 adapter.
  - Model size is chosen by a pre-specified feasibility rule: time 10 steps of
    Qwen2.5-1.5B. Use 1.5B if a full run takes ≤ 2.5 h; otherwise use 0.5B.
  - Hyperparameters are tuned only on 0.5B, using only reward and KL.
- **Prompts:** prompt v1, five outlet voices.
  - Training topics: 10. Held-out topics: immigration, economy, gender rights,
    environment.
  - Each outlet's training topics are drawn from that outlet's cleaned-real
    predicted-topic frequencies, renormalised over the training topics.
- **Target:** p̃_o, each outlet's M1 label shares on the cleaned real set, smoothed as
  (count + 1)/(N + 3). The bootstrap uncertainty of the target is reported.
- **Reward:**
  r(x) = Σ_l P_M1(l|x)·(p̃_o(l) − q̂_o(l)) − 1[format fail] − β·Σ_t sg(log π − log π_ref)
  - q̂_o is the batch estimate of the policy's label marginal for outlet o, shrunk
    toward the previous estimate (κ = 16) and initialised from 200 base samples.
  - The fixed point is q = p, and the reward is bounded.
  - Format fail: over 280 characters, link, hashtag, preface, decline, or no EOS.
  - β = 0.05.
- **Algorithm:**
  - On-policy policy gradient with group-mean-centred advantages, without dividing by
    the standard deviation (Dr. GRPO style).
  - Loss −Σ A·log π / total tokens.
  - 10 prompts per step (2 per outlet) × G = 8, so 80 rollouts.
  - AdamW, LR 2e-5 (β2 0.99, no weight decay), 10-step warmup, gradient clip 1.0.
  - 300 steps.
  - Reference log-probabilities computed with the adapter disabled.
  - Sampling: temperature 1.0, top_p 1, top_k 0, no repetition penalty, max 96 new
    tokens. Qwen's default generation config is overridden.
- **Conditions:**
  - **calibrated:** 3 seeds.
  - **format-only:** reward is the format penalty and KL only; 2 seeds.
  - **shuffled:** cyclic derangement of targets (NYT←Fox, Fox←NYT, ABC←AP, AP←Reuters,
    Reuters←ABC); 1 seed.
  - **pooled:** every outlet gets the pooled real target; 1 seed.
- **Evaluation:**
  - Checkpoints at steps 0, 50, 100, 200, 300. The final checkpoint is step 300.
  - Samples per checkpoint: 20 per outlet × training topic, and 50 per outlet ×
    held-out topic, under prompt v1.
  - The final checkpoint is also run under template sens-t1.
  - Measures: label shares, total variation (TV) and Jensen–Shannon divergence (JS)
    against p̃_o under M1, M3 and M2. M2's target is the judge distribution on that
    outlet's real tweets.
  - Noise floor: TV and JS between real subsamples of the same size.
  - Hacking checks:
    - KL within each named-party stratum;
    - party-mention rates against the real outlet;
    - M1 with names masked;
    - party-swap flip rates;
    - distinct-2 and duplicates;
    - reference-model perplexity;
    - judge genre and plausibility.
  - Goodhart curves: M1 and M2 divergence against KL to the reference.
- **Ethics:**
  - Adapters are not released.
  - Generated tweets are stored only as labelled research data, with masked outlet
    references in any published examples.
  - The paper discusses dual use: impersonation, persuasion, and the fact that real
    outlets cannot consent to being imitated.

## 8. Statistics

- **Primary intervals:** a cluster bootstrap with B = 10,000. Near-duplicate clusters
  are resampled within strata: voice × requested topic for generated tweets, outlet ×
  period for real ones. Percentile intervals.
- **Near-duplicate clusters:**
  - Lowercase the text; remove links and punctuation; mask outlet names, handles and
    numbers.
  - Build word 3-gram shingles and link pairs with Jaccard ≥ 0.5 (exact). Take
    connected components within each voice or outlet.
  - Re-run at Jaccard 0.3 and 0.7, and report the size distribution and the largest
    cluster.
- **RQ1 estimand:** Δ_v = Σ_t w_t(μ_vt − μ_nt).
  - Main-v1 uses w = pooled real predicted-topic shares.
  - Sensitivity runs use equal topic weights and are compared with sens-ref.
- **RQ2 estimand:** standardize generated and real tweets on the same measured variables
  (predicted topic × named-party stratum) to each outlet's real mix. Report the
  effective n and the share of real weight with no generated support. If more than 20%
  is unsupported, RQ2 is "partial".
  - Named-party strata condition on an outcome. They are reported as conditional, and
    party naming is reported as an agenda outcome in its own right.
- **Genre decomposition:** each voice difference is split into the effect of a
  different genre mix and the change within genres (Kitagawa decomposition).
- **Divergences:** JS and TV, with a permutation null. Plug-in KL is biased upward and
  is not used as a headline.
- **Multinomial logit:** voice + topic, with cluster-robust SEs, as a secondary model.
- **Fixed seeds:** all randomness uses fixed seeds. MPS training is not bit-
  reproducible, and the paper says so.

## 9. Budget and reproducibility

The API budget target is under US$40, with judges run through the Message Batches API.
Every run directory has a manifest of settings, model IDs, package versions and
source checksums. Generated texts are synthetic and are never presented as real
outlet posts.

## 10. Information barriers

**Principle:** whatever is being evaluated never sees the evaluation criteria, and evaluators never see what they
would need to be blind to.

| Component | Sees | Never sees |
|---|---|---|
| Generators (Haiku, Sonnet, Opus, base Qwen) | a topic and a voice (or a neutral event description), in the protocol's prompt | rubric, classifiers, hypotheses, real tweets, the words slant/stance/left/right |
| Event-description writer (Sonnet) | masked real tweets on one event; an instruction to write neutral facts | rubric, hypotheses, which outlet wrote which tweet |
| RL policy | the RL prompt (same as generator prompt v1) | reward definition, M1 itself, all evaluation measures (M2, M3, M3b, M6), real tweets |
| RL reward | M1 only | M2, M3, M3b and M6 are evaluation-only and never enter training, tuning or checkpoint selection |
| Judges (M2, M2b, M2c) | the rubric and one masked text | source (real or generated), voice, condition, model, M1 scores, hypotheses |
| Outlet-identification judge (M6) | the five outlet names (the task) and one masked text | whether the text is real or generated, and its condition |
| Human coders | the instrument and masked texts in shuffled order | source, condition, M1 and judge outputs |
| Analysis subagents | data and code they are asked to produce | the review criteria later used to evaluate their work; a separate reviewer applies those |

`study/audit_barriers.py` checks the first three rows and the human sheets mechanically:
- prompts contain no evaluation vocabulary;
- judge inputs carry no label fields and no unmasked outlet names;
- sheets contain no source or score columns.

The audit runs before every judging round and again before release. Its last result is in
`results/v2/validity/barrier_audit.json`.

A known residual channel: event descriptions are written from the real tweets that the event-conditioned
outputs are then compared with. They are restricted to neutral facts, with no quotations or outlet names. Lexical
overlap with the member tweets is reported.

## Change log

- 2026-10-16, v2.0: protocol written before v2 data.
- 2026-10-16, v2.1, still before data, after three independent reviews:
  - construct renamed and split into agenda / tone / framing;
  - symmetric keyword selection; event-conditioned arm;
  - more real exclusions, periods and quote-led flag;
  - non-Claude judge, masking, rubric with N/A and event valence, test-retest and
    order checks;
  - outlet-identification test and external ratings;
  - H1/H2/H3/H4 rules made numeric, with equivalence testing and Holm correction;
  - cluster bootstrap primary; estimands fixed; human set enlarged to 450, double-coded;
  - RL reward changed to a bounded squared-error form with batch-estimated q̂;
  - Dr. GRPO advantages; generation config overridden; k1 KL inside the reward;
  - shuffled and pooled controls; 3 seeds; hacking diagnostics; ethics section.
- 2026-10-16, v2.2, procedural, no outcome data analysed:
  - **Section 10 (information barriers)** added at the authors' request, with a mechanical audit script.
  - **RL model:** the feasibility rule selected Qwen2.5-0.5B. The 1.5B model ran at 63 s per step, a projected
    5.3 h per run; 0.5B ran at 27 s per step, 2.25 h per run.
  - **Learning rate:** kept at the pre-specified 2e-5, with no tuning sweep. In the 10-step pilot, reward and
    format compliance improved steadily.
  - **Link validity check:** run before any judge or RL analysis. A trailing t.co link flips M1's label for 37%
    of tweets and removes almost all "center" labels. All v2 measures therefore score text with links removed
    (`study.common.scoring_text`).
  - **Event-conditioned arm:** 77 adjudicated events instead of about 50. All events confirmed by the
    adjudicator are used.
- 2026-10-16, v2.3, adopted from a literature and tooling survey before any judge or RL outcome was examined:
  - **Politicalness (M5b):** `mlburnham/Political_DEBATE_large_v1.0`, an NLI model, as the M1 authors recommend.
    It is reported for all texts and RL outputs. Results are also reported for the political subset.
  - **Caricature metrics (Cheng et al. 2023; Bisbee et al. 2024):** for each outlet, report:
    - the within-outlet spread of scores, generated against real;
    - an exaggeration ratio: the distance of the generated voice from the neutral voice, divided by the real
      outlet's distance from the pooled real mean;
    - individuation: whether a logistic-regression discriminator on sentence embeddings, cross-validated by
      cluster, can tell generated tweets from real ones for each outlet.
  - **Human set:** drawn as a stratified random sample with recorded inclusion probabilities. Once human labels
    exist, judge-based estimates can be corrected with prediction-powered inference or design-based supervised
    learning (Egami et al. 2023).
  - **RL reporting:**
    - Goodhart curves (Gao et al. 2023): M1, M2, M3 and M5b divergence against KL to the reference.
    - Per-seed results, the mean and the range. With 3 seeds, an IQM is not meaningful.
    - The loss divides by the batch's total completion tokens. This is a constant per batch, close to the
      Dr. GRPO constant normalisation. Advantages are not divided by the standard deviation. Unit tests cover the
      advantage and reward computations.
  - **Release:** synthetic tweets are labelled as synthetic everywhere. The paper recommends that real tweets be
    released as IDs only, per X's terms. The repository currently holds full text from the source dataset (flagged
    to the authors).
- 2026-10-17, v2.4, deviations during execution. These are numerical and procedural, and were decided before any RL
  evaluation output existed:
  - **RL precision:** the first RL launch used bf16 weights. One run (format-only, seed 1) crashed at step 19 with
    NaN sampling probabilities on MPS. All runs were stopped and restarted in fp32 for every condition, so
    conditions stay identical. The aborted logs are kept in `results/v2/rl/aborted_bf16/`.
  - **Intermediate checkpoints:** steps 50, 100 and 200 are evaluated on smaller samples (5 per outlet × training
    topic, 10 per outlet × held-out topic) and are used only for the Goodhart curves. The pre-registered final
    checkpoint (step 300) gets the full evaluation.
  - **Masking strengthened after the first judging round:** all @handles, outlet sub-brands (Breakingviews,
    VoteCast) and polling partners (Ipsos, Siena) are now masked. 148 of 13,718 items changed and were re-judged.
    Judge inputs for every analysis use the final masking.
  - **Qwen2.5-7B judge (M2c):** uses the normal, unflipped instrument for every item. In a first pass, 12 of its
    first 40 replies were invalid and it appeared to mishandle the reversed scale. The flip exists only to detect
    order effects in the primary judge (M2), where they were negligible. The local judge was also paused while
    RL trained, because the two processes competed for the GPU.
- 2026-10-17, v2.5. After the first full analysis, an independent peer review and an independent code audit were
  done by subagents that were not involved in the analysis. Everything here was recorded after the main results were
  seen.
  - **Code fixes from the audit. None reverses a headline result.**
    - Table 2 levels are now plain means over units with a value.
    - Judge results from earlier masking versions are ignored, so a flip is never guessed.
    - Show names that are also common phrases ("This Week", "The Daily", "The Times") are masked only in title
      case. Previously 231 ordinary uses were masked. The changed items were re-judged.
    - Party swaps skip texts naming both Harris and Biden, and replace names whatever their case.
    - Party keywords add RNC, DNC, Dems, DeSantis and Haley.
    - In random-effects pooling, a zero bootstrap SE is replaced by the smallest positive SE in the set, instead of
      1e-6.
  - **Deviations found by the review that were not logged earlier:**
    - **H1 robustness threshold for the judge.** We used |Δ| ≥ 0.2, half the judge's directional margin. This
      mirrors the M1 rule (7.5 is half of 15), but the protocol did not state it. It was set in code before the
      variant results were computed. The paper reports how sensitive the conclusion is to this threshold.
    - **Built neutral templates:** 30 were used, against 60 in the protocol.
    - **Bootstrap strata for real tweets:** outlet × predicted topic × named party, not outlet × period. Periods are
      analysed separately.
    - **Headline measure:** the paper headlines the net score, not label shares. Label shares are reported in the
      results files.
    - **Table 1 uses main-v1.** The protocol made main-v1 exploratory for H1. H1 is decided on the five named
      variants.
    - **Human set:** 450 sampled items plus 30 swap pairs and 20 repeats, giving 500 rows and 480 unique texts.
  - **Equivalence margin.** The pre-registered ±0.4 judge margin is too loose to be informative: real NYT sits
    0.30 from the real mean. A data-driven margin of ±0.097, half the range of real non-Fox outlets, is added and
    reported alongside it.
  - **Post-review exploratory analyses** (`study/analyze_extra.py`), all labelled exploratory in the paper:
    - the party-named stratum;
    - Fox within genre;
    - event-arm TOST, lexical overlap and event-matched individuation;
    - the period-matched (P2) comparison;
    - slant conditional on outlet misidentification;
    - Jaccard 0.3/0.7 and full-set sensitivity (both pre-registered);
    - bootstrap CIs for benchmark correlations;
    - judge sign-reversal on swaps;
    - judge variance by condition.
- 2026-10-17, v2.6. RL feasibility deviation, decided before any RL evaluation output existed. Only the training-time
  reward and KL logs had been seen.
  - **Why:** with fp32, which the v2.4 NaN fix requires, and two runs sharing the GPU, a step took about 130 s. The full
    design (7 runs × 300 steps) would have taken about 40 hours.
  - **Final checkpoint:** step 200 instead of 300, for every run.
    - calibrated-s1 and format-s1 were launched with 300 steps and are stopped right after their step-200 checkpoint.
    - All later runs use `--steps 200`.
  - **Order:** runs go in priority pairs: (calibrated-s1, format-s1), (calibrated-s2, shuffled-s1), (format-s2,
    pooled-s1), then calibrated-s3. Results are reported for whatever has completed, with the completed set stated.
  - **Evaluation:** intermediate checkpoints are 50 and 100. The step-200 checkpoint gets the full evaluation sample.
- 2026-10-17, v2.7. Execution changes and additions, made before any RL evaluation output existed:
  - **One GPU job at a time.** Two fp32 runs in parallel exhausted unified memory and stalled on swap. The stalled
    calibrated-s1 run, at step 25, was discarded and is rerun from scratch. Running alone, a 0.5B step takes about
    25 s.
  - **Memory:** log-probabilities are computed in chunks of 4 rollouts, and the MPS cache is emptied every step.
    Neither changes the training computation.
  - **Exploratory scale check:** Qwen2.5-1.5B-Instruct, calibrated vs format-only, seed 1, 200 steps, run after the
    pre-registered 0.5B set. Its base model is also evaluated.
  - **Base models as generators:** the untrained base models (0.5B and 1.5B) are added to the portrayal analysis as
    generators "model-qwen" and "model-qwen-1.5b" (exploratory), from their prompt-v1 evaluation samples.
  - **Revisiting v1 (`study/v1_revisited.py`):** starting from v1's published comparison, apply one correction at a
    time and report how each moves the outlet gaps.
- 2026-10-17, v2.8. **Scope change: the RL study (RQ5, H4, section 7) is deferred.** The authors chose to focus on
  part 1 (RQ1–RQ4) and to redesign the RL study from scratch later.
  - **State when stopped:**
    - format-only seed 1 had reached step 66 of 200 (fp32, sequential);
    - no RL checkpoint had been evaluated;
    - no RL result was seen or reported.
  - **Archive:** code, the protocol at v2.7, partial runs and logs are preserved at git tag `archive/rl-attempt-1`
    (commit 71a0ddb). They were removed from the active branch.
  - **Lessons for the redesign:**
    - bf16 sampling on MPS produced NaNs;
    - two fp32 runs in parallel exhausted the 64 GB of unified memory;
    - running alone, a 0.5B step takes about 25 s and a 1.5B step about 63 s (bf16 pilot);
    - the reward classifier's name and link sensitivity (H3) should be designed around from the start.
- 2026-10-17, v2.9. **Environment consistency.** Installing `sentence-transformers` mid-study had upgraded `transformers`
  from 4.57.6 to 5.20.1. Real tweets were scored under 4.57.6 and most generated tweets under 5.20.1. The two versions
  give DeBERTa-v3 probabilities that differ by up to 0.10 for individual texts; labels did not change in a 200-text
  check. The environment was pinned back to 4.57.6 (`requirements-study.txt`), which reproduces the original
  real-tweet scores to within 2.5e-6. All M1 and M3 scores were recomputed under it (22,714 texts), and every analysis
  was rerun. No reported number changed at the reported precision; the largest change was 0.1 points, in the
  partisan swap mean. The topic and loaded-wording models were unaffected. The human-coding sample was not redrawn,
  because its strata and inclusion probabilities were fixed when it was drawn.
