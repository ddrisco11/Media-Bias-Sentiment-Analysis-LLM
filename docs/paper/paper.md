# Imitating News Outlets with a Language Model: Classifier Artifacts, What Survives Them, and Optimising Against a Bias Proxy

*Working paper, draft of 2026-10-17. Authors: Kushaan Vardhan, David Driscoll. The analysis pipeline was built with
Claude Code.*
- *Pre-registration:* `docs/protocol_v2.md`, committed before any v2 data, with every later revision logged.
- *Reproducibility:* every number is produced by the code in `study/` from the files in `results/v2/` (Section 13).

> **Synthetic-content notice.** This paper analyses tweets written by language models "in the voice of" real news
> organisations. They are not posts by those organisations. They often contain invented facts, and they must not be
> shared or presented as real.

> **Validation status.** Framing and tone results rest on a blind LLM judge (Claude Sonnet 5.5). It is reliable on
> retest and agrees with a second Claude judge. It has **not yet been validated against human coders**, and the
> study's generator also belongs to the Claude family. A double-coded human sample has been prepared (Section 4). Until
> it is coded, read every judge-based claim as "framing as perceived by a Claude judge".

## Abstract

[Written once the RL results are final.]

## 1. Introduction

Language models are increasingly asked to simulate people and institutions. They stand in for survey respondents
("silicon samples"), draft newsroom content, and answer questions about what a source would say. Ask a model to write
*as* The New York Times or *as* Fox News: what does it produce, and does its picture of each outlet's politics match
what the outlet actually publishes?

We study Claude Haiku 5.5 writing election-season tweets in the voices of five US news outlets (NYT, Fox News, Reuters,
AP, ABC News) and of a neutral "US news organization". We compare these with 2,282 real tweets those outlets posted
during the 2024 campaign.

An earlier version of this study (v1) measured political leaning with a widely downloaded left/center/right
classifier. It reported large differences between outlets: "the model portrays ABC far to the left of the real ABC",
and "real tweets are almost never centrist, generated ones often are". This paper makes measurement a first-class
question and asks which conclusions survive. Our findings are:

1. **The classifier is unusable for this comparison as deployed.**
   - A trailing t.co link changes its label for 37% of tweets and almost eliminates "center". This fully explains
     v1's centrism contrast.
   - Swapping party names in politically neutral sentences moves its score by 81 points on average and flips 47% of
     its labels.
2. **Under a blind judge, only the Fox News voice frames the news differently** from the model's neutral voice. The
   NYT, Reuters, AP and ABC voices are statistically equivalent to it, within ±0.1 on a 7-point slant scale. The
   classifier's large and prompt-robust differences for Reuters, AP and ABC do not appear under the judge. v1's ABC
   finding does not survive.
3. **Compared with the real outlets, the model writes a different genre and a flattened politics.**
   - **Genre:** topic-only prompts produce issue explainers, TV promotions and voting reminders; real outlets post
     news reports.
   - **Flattened politics:** within every non-Fox voice, the judge rates every party-naming tweet exactly 0. Real
     tweets vary, and real NYT leans left.
   - **Fox:** the voice's overall slant matches real Fox, but only through its opinion and promotional tweets. In news
     reports about the same events it is far milder (+0.13 against +0.62).
4. **Optimising an open model against the classifier.** We train Qwen2.5-0.5B with RL, rewarding it for matching each
   outlet's real label distribution. We use format-only, shuffled-target and pooled-target controls and measures the
   model never trains on. This tests whether a bias proxy can be optimised without reward hacking [Section 9].

Contributions:
- a pre-registered, multi-measure design for comparing LLM outlet portrayals with real outlet output (symmetric
  keyword selection, an event-conditioned arm, information barriers, built counterfactuals);
- two quantified measurement artifacts in an off-the-shelf political-leaning classifier, each large enough to flip a
  substantive conclusion;
- a decomposition of "how the model imagines outlets" into agenda, genre, tone and framing;
- an RL study of optimising against that classifier, with controls and held-out measures.

## 2. Related work

**LLMs as simulators.**
- Argyle et al. (2023) argue that conditioned LLMs can reproduce subgroup response distributions ("algorithmic
  fidelity").
- The cautionary literature is larger:
  - Santurkar et al. (2023) find substantial misalignment between LLM opinions and US demographic groups.
  - Bisbee et al. (2024) find that synthetic survey responses match means but understate variance, and drift with
    prompt wording and over time.
  - Cheng et al. (2023, CoMPosT) formalise caricature as individuation plus exaggeration and find that political
    personas are especially prone to it.
  - Wang et al. (2025) document that LLMs flatten and misportray groups.

  Our design adopts their recommendations: report variance and exaggeration, not only means; test individuation;
  and pre-specify a robustness grid.

**Measuring political slant.** Media-bias classifiers trained on outlet-labelled articles tend to learn outlet
identity (Baly et al. 2020). Sentence-level bias classifiers rely on connotated words and on entity names (Wessel &
Horych 2024). LLMs infer leaning from cues about the source rather than its content (Dai et al. 2025; Park & Kim
2025). Our classifier (arXiv:2507.13913) was trained on articles, posts and LLM-generated statements, and its
authors recommend a politicalness pre-filter.

**LLM-as-judge.** Judges show position and verbosity biases (Zheng et al. 2023) and self-preference that grows with
self-recognition (Panickssery et al. 2024). Their reliability varies by task, and they need validation against
humans (Bavaresco et al. 2025). Design-based supervised learning (Egami et al. 2023) and prediction-powered
inference give valid estimates from LLM labels combined with a random human-coded subset. We use a blinded,
pointwise judge with flip randomisation. We add a second Claude judge and a non-Claude judge, and prepare a
human-coding sample with known inclusion probabilities.

**Reward hacking.** Optimising against a learned proxy improves the proxy while the true objective eventually
degrades. The gap is predictable in KL distance (Gao et al. 2023), and hacking can appear abruptly (Pan et al.
2022). Dr. GRPO (Liu et al. 2025) removes length and standard-deviation normalisation biases from group-relative
policy gradients. We use its advantage form, and we report Goodhart curves against held-out measures.

## 3. Data

### 3.1 Real tweets
The real tweets are 2,500 posts, 500 each from the main X accounts of @nytimes, @FoxNews, @Reuters, @AP and @ABC.
They were drawn by deterministic hashing from the USC 2024 election dataset (arXiv:2411.00376). That
dataset was collected with 44 election keywords, plus "Assassination" from 13 July and "JD Vance" from 15 July 2024.
It therefore samples each outlet's election-related output, not its full feed.

Before analysis we excluded, with pre-registered rules:
- non-English tweets (22);
- replies (25) and thread continuations (9);
- tweets dated before 2024 (7);
- opinion or analysis promotions (68);
- live or video promotions (46);
- within-outlet duplicates (45).

This leaves **2,282** tweets: NYT 417, Fox 467, Reuters 443, AP 479, ABC 476. We assign each to a period: before
Biden's withdrawal on 21 July (P1), 21 July to Election Day (P2), and after Election Day (P3). Quote-led tweets
(292) are flagged, not dropped.

### 3.2 Generated tweets
Each request produced one tweet, with no system prompt, from a prompt giving only the year, a topic and a voice
(prompt v1):

> It is 2024, during the US presidential campaign. Write one tweet that {voice} might post on its main X account about
> {topic}. Reply with only the tweet text: 280 characters or fewer, no links, no hashtags.

**Voices.** The five outlets, each named with its handle, and "a US news organization".

**Topics.** 14, matching the topic taxonomy used on the real tweets. They are worded neutrally; for example,
"abortion" rather than "reproductive rights".

| Condition | Generator | Variation | Tweets |
|---|---|---|---:|
| main-v1 | Haiku 5.5, thinking off | real topic mix, ≥10/topic | 3,150 |
| pilot-v1 | Haiku 5.5 | thinking on vs off | 840 |
| sens-ref | Haiku 5.5 | prompt v1, re-run alongside the variants | 840 |
| sens-t1, t2, t3 | Haiku 5.5 | three reworded templates | 840 each |
| sens-month | Haiku 5.5 | "It is {Month} 2024", months drawn from the real date mix | 840 |
| model-sonnet / model-opus | Sonnet 5.5 / Opus 5.5 | prompt v1 | 840 each |
| sens-baseline | Haiku 5.5 | neutral voice as "no persona" or "a journalist" | 280 |
| five-voice | Haiku 5.5 | all five outlets in one request, order randomised | 700 |
| event-v1 | Haiku 5.5 | same news event for every voice (77 events × 6 voices × 3) | 1,386 |

Sensitivity conditions use 10 tweets per voice × topic. The **event-conditioned arm** works in three steps:
1. Embedding-based clustering of the real tweets (average linkage, at most 3 days apart) proposed 111 candidate
   events covered by at least 3 outlets.
2. Sonnet 5.5, blind to which outlet wrote each tweet, confirmed 77 single events. For each, it wrote a two- or
   three-sentence neutral factual description with no evaluative words, quotations or outlet names.
3. Each voice then wrote a tweet "about this news: {description}", dated to the event.

Real and generated tweets about the same event can then be compared directly.

**Information barriers.**
- No generator ever saw evaluation vocabulary, the rubric, the classifiers or real tweets.
- An automated audit (`study/audit_barriers.py`) checked every one of more than 1,400 distinct prompts.
- Judges saw only masked text: links removed, outlet names, handles, sub-brands and polling partners replaced.

## 4. Measures and validity checks

**M1, the classifier.** `matous-volf/political-leaning-deberta-large` gives P(left/center/right). We report label
shares and the net score 100·[P(right) − P(left)]. All texts are scored with links removed (Section 7.1).

**M2, the blind judge.** Claude Sonnet 5.5, given a codebook and anchored examples, rates:
- **stance** toward Democrats and toward Republicans (−2..+2, or NA if not mentioned);
- **event valence** for each party, separately from framing;
- **slant** on a 7-point scale. The instruction: "judge framing, not the news itself; a factual report of an event
  that is bad for one party is 0 unless word choice, selection, emphasis or omission tilts it";
- **genre**;
- **plausibility** as a real outlet post.

Half of the items, chosen by hash, receive a flipped instrument: the field order is reversed and the scale direction
inverted. Their answers are mapped back afterwards.

**Reliability checks on the judge.**
- **M2b:** Opus 5.5 on the 480-item human set.
- **M2c:** Qwen2.5-7B-Instruct, run locally with the same rubric, as a non-Claude judge.
- **Test-retest:** 200 items re-judged in a separate batch.

**M3, held-out classifier.** `premsa/political-bias-prediction-allsides-DeBERTa`. It is used for evaluation only. It
separates Fox from NYT on real tweets (net +7 vs −37), which was the pre-registered condition for using it.

**M3b, loaded wording.** MAGPIE-BABE, a non-directional bias classifier.

**M5b, politicalness.** Political DEBATE, an NLI classifier, scoring "This text is about politics."

**M6, outlet identification.** The judge picks which of the five outlets posted a masked tweet.

**External benchmarks.**
- Audience partisanship of each outlet's domain (Robertson et al. 2018), the primary benchmark.
- AllSides, Ad Fontes, Bakshy et al. (2015) and MBFC as secondary benchmarks.

**Built counterfactuals.** Party-swap pairs (4,148), including 30 built neutral templates with a party slot.

**Human validation set (prepared, not yet coded).** 450 items stratified by source × genre × M1 label × named
party, with recorded inclusion probabilities. It also contains 30 swap pairs and 20 repeats. Two blind coding sheets
and a codebook identical to the judge rubric are provided. **No human labels exist yet. Every judge-based conclusion
is provisional on them.**

## 5. RL study design

Haiku cannot be fine-tuned, so the RL study uses Qwen2.5-0.5B-Instruct with a LoRA adapter (r = 16, all linear
layers). Its pre-RL outputs are its own baseline. The 0.5B model was selected by a pre-registered feasibility rule:
1.5B ran at 63 s per step on the available Apple M2 Max.

**Prompts.** Prompt v1 for the five outlets. 10 training topics are drawn from each outlet's real topic mix; 4 topics
are held out (immigration, economy, gender rights, environment).

**Reward.**
r(x) = Σ_l P_M1(l|x)·(p_o(l) − q̂_o(l)) − 1[format fail] − β·Σ_t (log π − log π_ref)
- p_o is the outlet's smoothed real M1 label distribution.
- q̂_o is the policy's batch label marginal, shrunk toward its previous value (κ = 16).
- The leaning term rewards labels the policy under-produces relative to the real outlet. Its fixed point is
  q = p, and it is bounded.

**Training.**
- Advantages are group-mean-centred, with groups of 8 completions per prompt and no division by the standard deviation.
- 80 rollouts per step. The final checkpoint is step 200 (300 in the protocol; reduced for feasibility, change log v2.6).
- AdamW at learning rate 2e-5.
- β = 0.05, using a sequence-level k1 KL estimate placed in the reward.
- Sampling at temperature 1 with no top-k or top-p truncation.
- fp32 weights. A first bf16 launch produced NaN sampling probabilities and was restarted in fp32 for every
  condition.

**Conditions.**
- calibrated: 3 seeds;
- format-only: 2 seeds;
- shuffled targets, a cyclic derangement of outlets: 1 seed;
- pooled target, with every outlet given the pooled real distribution: 1 seed.

**Evaluation.**
- Every checkpoint and the base model are sampled on training topics, held-out topics, and a reworded template.
- Each sample is scored by M1 (the reward model), M3 and M2, none of which except M1 enter training.
- Reward-hacking diagnostics:
  - M1 with party names masked;
  - party-swap flip rates;
  - party-mention overshoot;
  - distinct-2 and duplicate rates;
  - reference-model NLL;
  - politicalness.

## 6. Statistics

- **Intervals:** a cluster bootstrap with B = 10,000. Near-duplicate clusters are formed by word-3-gram Jaccard
  ≥ 0.5 on masked text, and resampled within strata: voice × requested topic for generated tweets, and
  outlet × predicted topic × named party for real ones.
- **RQ1:** voice-minus-neutral contrasts are standardised to the pooled real topic mix (main-v1), or use equal topic
  weights (variants).
- **RQ2:** generated-versus-real gaps are standardised on predicted topic × named party to each outlet's real mix. We
  report the share of real weight that has no generated support.
- **Decision rules:** pre-registered (protocol §3), with equivalence tests (TOST, ±15 M1 points or ±0.4 judge
  points), Holm correction across 5 voices × 2 measures, and DerSimonian–Laird pooling across the five pre-named
  variants.


## 7. Results I: measurement

### 7.1 A trailing link changes the classifier's answer
Real outlet tweets usually end in a t.co link, and v1 scored them with it. Removing the link changes M1's label for
**37.5%** of the 2,275 real tweets that have one. The share labelled "center" goes from 1.6% with links to 27.5%
without. Appending a random t.co link to generated tweets, which have none, changes the label for 36.7% of them, and
the center share falls from 37.8% to 4.2% (Figure 4).

v1's contrast ("1.6% center among real tweets against 39% among generated") is therefore mostly a link artifact. With
links removed from both, the shares are 27.5% and 37.8%. All v2 measures score text without links.

### 7.2 The classifier tracks party names
The clean test uses 30 sentences that are neutral by construction, for example "{candidate} will hold a campaign rally
in Pennsylvania on Saturday, the campaign said" (Figure 3).
- **The classifier:** swapping Harris for Trump changes M1's net score by **81 points** on average and flips **47%**
  of its labels.
- **The judge:** its slant does not change for any of them.

On tweets the judge rated neutral, swapping names moves M1 by 45 points (real) and 25 points (generated). The judge
moves by 0.17 and 0.03 points on its 7-point scale. On neutral real tweets, though, the judge's rating changes for
**15.4%** of swaps. That is just above the 15% flip threshold pre-registered for M1. Some of these swaps are factually
awkward ("former President Harris"), so the figure mixes name sensitivity with real changes in meaning. The judge is
**not fully name-invariant either**; it is far less sensitive than M1.

On partisan tweets, a valid measure should change. There the judge's slant reverses sign for 57% of swaps, keeps its
sign for 20%, and goes to 0 for 22%. M1's change correlates with the judge's at Spearman 0.55.

Under the pre-registered rule (H3), M1 is name-sensitive, and the judge becomes co-primary.

### 7.3 Judge reliability, and the limits of what it shows
- **Test-retest:** 200 items re-judged in a separate batch give Krippendorff's α = 0.90 for slant and 0.92–0.96 for
  stances, genre and the NA flags.
- **Second judge:** Opus 5.5 on the 480-item human set agrees with Sonnet at α = 0.77 for slant, 0.76 and 0.78 for the
  stances, and 0.83 for genre.
- **Order and anchor effects:** reversing the field order and scale direction changes mean slant by −0.04 ± 0.03
  (real) and −0.01 ± 0.01 (generated), which is negligible.

**These are reliability results within one model family, not validity results.** The local non-Claude judge
(Qwen2.5-7B) is still being run [Section 9]. Human coding remains the gate.

Two observations bound how far the judge can be trusted:
- **It is not fully blind in practice.** With outlet names masked, it still identifies the outlet of 67.5% of real
  tweets (chance 20%), and of 92% of Fox tweets. Real Fox tweets it fails to identify as Fox have a mean slant of 0.00
  (n = 39), against +0.77 for those it identifies. The same pattern holds for the generated Fox voice: 0.00 against
  +0.91. Either slant is what makes a tweet recognisable as Fox, or recognising Fox raises the slant rating. The data
  cannot separate the two.
- **External benchmarks do not discriminate between the measures.** Against Robertson et al.'s (2018) audience
  partisanship, the judge's ordering of the five real outlets has Spearman ρ = 0.8, but its bootstrap 95% interval is
  [0.3, 0.9]. M1's is 0.3. With five outlets, Fox at one end and NYT at the other fix most of the ranking, so the
  benchmark comparison is descriptive only.

**Agreement among measures is modest, and collapses on event-conditioned text.**
- **Real tweets:** M1 correlates with the judge at ρ = 0.39, and the held-out classifier M3 at 0.23.
- **Event-conditioned tweets:** ρ = 0.05 and 0.03, although M1 and M3 still agree with each other (0.54).

Range restriction contributes: the judge rates 93% of event-conditioned tweets 0, against 67% of real tweets. So the
collapse does not show that the classifiers measure something the judge misses. It does show that the two kinds of
measure cannot stand in for each other.

### 7.4 Revisiting v1: one correction at a time
v1 reported, for each outlet, the classifier's real score minus its "expected" (Haiku) score. Positive means the real
outlet scores further right. Table 3 starts from v1's published numbers and applies the v2 corrections cumulatively,
changing one thing per row. This also shows which kinds of mistake matter most in studies like this one.

**Table 3.** Real minus expected, by correction step (M1 net, −100..+100; the judge rows use slant, −3..+3).
95% bootstrap intervals in `results/v2/analysis/v1_revisited.json`.

| Step | NYT | Fox News | Reuters | AP | ABC News | Center share, real / expected (range over outlets) |
|---|---:|---:|---:|---:|---:|---|
| S0. v1 as published | +1 | **+20** | −1 | **+28** | **+68** | 0–6% / 3–63% |
| S1. Links removed from real tweets | **+12** | **+2** | +5 | +19 | +64 | 10–53% / 1–63% |
| S2. Real sample cleaned | +13 | +8 | +6 | +19 | +64 | 11–53% / 1–63% |
| S3. Same keyword selection for generated | +3 | **−25** | 0 | +12 | +64 | 11–53% / 2–79% |
| S4. Only tweets naming both parties | +2 | −23 | −4 | +10 | +60 | 10–56% / 2–84% |
| S5. Judge instead of M1 (S2 sample) | −0.23 | +0.06 | −0.05 | −0.08 | −0.07 | — |
| S5b. Judge, only tweets naming both parties | −0.34 | **+0.63** | −0.04 | −0.17 | −0.23 | — |

What each step teaches:
- **The link artifact alone rewrote two of v1's three headline gaps.** Fox's +20 fell to +2. NYT's gap went from +1 to
  +12. The "real tweets are almost never centrist" pattern disappeared: real center shares rose to 10–53%.
- **Asymmetric selection biases the comparison.** v1 compared real tweets that passed an election-keyword filter
  with generated tweets that mostly would not have. Applying the same filter to generated tweets flips Fox's gap to
  −25: keyword-matching Fox-voice tweets are more extreme than real Fox under M1.
- **ABC's +68 survives every data correction under M1 and disappears under the judge.** It is a property of how M1
  reads the ABC voice's generic explainers and reminders, not of how the model portrays ABC.
- **Under the judge, comparing like with like (S5b) gives the clearest picture.**
  - **Real outlets are more partisan than the model's versions of them,** in both directions: real NYT −0.34, ABC
    −0.23, AP −0.17 on the left; real Fox +0.63 on the right.
  - **The model's outlet voices are close to neutral** whenever they name the parties.

Three more lessons come from v1's supporting analyses:
1. **Topic recognition.** v1 found the topic classifier recognised only 1 of 1,170 tweets requested as "president".
   This follows from genre: the model writes about "the president and the presidential candidates" as generic
   campaign roundups ("Biden and Trump are both pressing their cases…"). The topic model files these under elections
   or under issues they list. Real "president" tweets report specific actions by named people.
2. **Thinking mode.** With thinking on, Haiku declined 8% of outlet-voice requests. The declines clustered on ABC and
   Reuters (19% each) and on gender and race topics (17–27%). Any thinking-on sample therefore under-represents
   exactly those cells. All main analyses use thinking off.
3. **Topic wording.** Requesting "voting and the election" produced voting reminders in every voice. That one choice
   of wording accounts for about 23% of all generated tweets.

## 8. Results II: how the model portrays each outlet

### 8.1 Against the model's own neutral voice (RQ1)
Table 1 compares each voice with Haiku's own neutral voice. Main-v1 is standardised to the real topic mix, with
cluster-bootstrap 95% intervals (B = 10,000).

**Table 1.** Voice minus neutral voice, main-v1 (Haiku 5.5, prompt v1, 525 tweets per voice).

| Voice | M1 net | Judge slant | Stance → Dem | Stance → Rep | M3 net | Names a party (Δ) |
|---|---|---|---|---|---|---|
| NYT | +2.4 [−4.5, +9.6] | +0.00 [−0.01, +0.02] | −0.01 | +0.01 | +4.4 [+2.2, +6.7] | +0.15 |
| Fox News | **+100.3** [+92.1, +108.0] | **+0.66** [+0.62, +0.70] | **−0.60** | **+0.42** | +52.7 [+46.6, +59.8] | +0.08 |
| Reuters | +41.0 [+34.1, +48.0] | +0.02 [+0.01, +0.03] | −0.02 | +0.02 | +26.3 [+23.1, +29.5] | +0.19 |
| AP | +37.8 [+31.1, +44.8] | +0.02 [+0.01, +0.03] | −0.02 | +0.02 | +26.9 [+24.3, +29.6] | +0.23 |
| ABC News | −29.6 [−35.7, −23.5] | +0.02 [+0.00, +0.03] | −0.02 | +0.02 | −16.6 [−19.3, −13.7] | +0.22 |

*Neutral voice: M1 net −46.2, judge slant −0.02. Stance intervals are within ±0.05 except for Fox.*

**Pre-registered robustness (H1)** is decided on five named variants: three reworded templates, a month-anchored
prompt, and Sonnet 5.5 as generator (Figure 6).
- **Fox News: robust on both measures** (5 of 5 variants). Its pooled judge slant is +0.94 [0.60, 1.28], ranging from
  +0.39 (Sonnet) to +1.40 (template t3).
- **Reuters and AP: robust on M1** (5 of 5; pooled +40.7 and +38.9). **ABC: robust on M1** (4 of 5; pooled −11.9).
  **None of the three is robust on the judge** (0 of 5).
- **NYT: robust on neither measure.**

The H1 rule for the judge used a robustness threshold of |Δ| ≥ 0.2. This was set in code, but not in the protocol
(change log v2.5). The conclusion does not depend on it: the judge differences for Reuters, AP and ABC lie within ±0.04
in every variant.

**Equivalence.**
- The pre-registered margin (±0.4) is uninformative, because real NYT sits 0.30 from the real mean.
- At a data-driven margin of ±0.099 (half the range of the real non-Fox outlets), NYT, Reuters, AP and ABC are each
  equivalent to the neutral voice in main-v1 (TOST, 90% CIs).
- Pooling the variants detects a tiny NYT difference (judge −0.025 [−0.042, −0.008]; M1 +5.7 [1.6, 9.8]; Holm p =
  0.017 and 0.025), far inside that margin.

**So the classifier differences for Reuters, AP and ABC are robust, but they are not framing.** A genre decomposition
attributes 0.4–9 points of each M1 difference to genre mix and the rest to differences within genre. What the voices
write about and how they word it moves both classifiers, and the judge treats none of it as slant.

**What does differ is agenda.** Every outlet voice names a party more often than the neutral voice (+8 to +23 points).

**Exploratory variations.** Writing all five voices in one request did not exaggerate their differences: the spread
of voice means was 0.36 for slant, against 0.43 when written separately. Replacing the neutral persona with "no
persona" or "a journalist" left framing neutral (−0.04 and −0.11, against −0.03).

### 8.2 Against the real outlets (RQ2)
**Table 2.** Real (cleaned) tweets against Haiku's voice (main-v1). Plain means.

| Outlet | Judge slant, real / generated | Stance → Rep, real / generated | Names a party, real / generated | Generated tweets that pass the source keyword filter |
|---|---|---|---|---|
| NYT | **−0.24** / −0.01 | −0.30 / −0.00 | 98% / 29% | 21% |
| Fox News | +0.71 / +0.64 | +0.27 / **+0.46** | 98% / 21% | 20% |
| Reuters | −0.05 / +0.00 | −0.10 / +0.00 | 98% / 32% | 29% |
| AP | −0.08 / +0.00 | −0.12 / +0.00 | 94% / 37% | 38% |
| ABC News | −0.07 / −0.00 | −0.11 / +0.00 | 99% / 35% | 21% |

**1. Genre: the model writes a different kind of tweet.**
- **Real tweets:** mostly news reports (52–71%), plus breaking news, polls and quote-led posts. None are voting
  reminders.
- **Topic-only non-Fox voices:** issue explainers (33–54%).
- **Every voice:** exactly 23% of each voice's tweets are voting reminders, which is every tweet from the "voting
  and the election" prompts. This genre is an artifact of our neutral topic wording, not of how the model imagines
  outlets.
- **The Fox voice:** no news reports at all; 43% are TV promotions ("Tune in tonight…") and 29% opinion. Real Fox
  tweets are 52% news reports and 33% quote-led.
- **Consequences:** generated tweets rarely name a party (21–37%, against 94–99% of real tweets), and only 20–38%
  would have passed the source dataset's own election-keyword filter.
- **Distinguishability:** a classifier on sentence embeddings, cross-validated by near-duplicate cluster, tells
  topic-only generated tweets from real ones with AUC ≈ 1.00 for every outlet.

**2. Haiku flattens every outlet except Fox.**
- **Generated non-Fox tweets naming a party:** the judge gives every one of these, in every non-Fox voice, slant
  exactly 0 (SD 0.00).
- **Real tweets naming a party:** they vary (SD about 0.5), and real NYT averages −0.25.
- **This is not a "no party named means slant 0" floor.** It holds within the party-naming stratum.
- **In the event arm:** generated tweets vary somewhat more, at 36–48% of the real spread.

**3. The model erases the NYT's left tilt, and the campaign-period tilt of the other mainstream outlets.**
- **Over the whole period:** real NYT leans left under the judge (slant −0.24; stance toward Republicans −0.30), and
  the generated NYT voice is neutral.
- **In the general-election period** (21 July to 5 November), real NYT, ABC, AP and Reuters all lean left: −0.37,
  −0.28, −0.22 and −0.16.
- **Month-anchored prompts for August–October** produce neutral voices for all four (−0.03 to 0.00, n = 34–48 per
  voice).
- **The Fox voice overshoots in that period:** +1.37 against +0.74 for real Fox (n = 30, exploratory).

**4. The Fox voice gets the right overall level for the wrong reasons.**
- Its mean slant (+0.64) matches real Fox (+0.71), but the agreement comes from genre mix. Its opinion tweets score
  +1.80, and it writes no news reports.
- **Among tweets that name a party,** the voice scores +0.26 against real Fox's +0.73.
- **In news reports about the same events,** it scores +0.13 against real Fox's +0.62.
- The voice's tone is also off: it is warmer toward Republicans than real Fox (+0.46 against +0.27).
- The "exaggeration ratio" of 1.02 depends on the reference point (about 0.78 against the mean of voices), so we do not
  interpret it.

**5. Identification.**
- **Real tweets:** the blind judge picks the right outlet for 67.5%.
- **Generated tweets:** 51.8% topic-only and 37.4% event-conditioned.
- **Outlet distinctiveness:** apart from ABC and AP, generated voices carry less outlet-specific signal than the
  real accounts.

**The standardised comparison and H2.**
- Standardising on predicted topic × named party is only *partial*: 73–79% of each outlet's real weight has no
  generated counterpart, because real tweets almost always name a party. We therefore rely on the event arm and on
  the stratified analyses above.
- **H2 is not supported.** v1's claim that Haiku portrays ABC far to the left of the real ABC rests entirely on M1
  (keyword-filtered gap −74 [−93, −57]). Under the judge the gap has the *opposite* sign: +0.12 [+0.01, +0.23], with
  the generated ABC voice slightly *right* of real ABC because real ABC leans slightly left. It is +0.05 [−0.02, +0.13]
  in the event arm.

### 8.3 Holding the news fixed: the event-conditioned arm
Given the same neutral description of a real event, every voice writes a news report: 78–85%, and 65% for Fox, which
adds breaking-news framing. Real and generated tweets about the same events then become hard to tell apart. An
embedding classifier trained only on same-event pairs reaches AUC 0.46 (NYT), 0.43 (ABC), 0.54 (AP), 0.58 (Reuters)
and 0.70 (Fox).

Voice-minus-neutral slant is small. Fox is the largest at +0.11 [0.02, 0.21]; the others range from +0.03 to +0.05,
with intervals including zero.

**Gaps to the real outlets on the same events** (judge slant, generated minus real):
- NYT +0.09 [−0.01, +0.18];
- Fox −0.19 [−0.46, +0.10], 28 events;
- Reuters −0.03;
- AP +0.01;
- ABC +0.05.

Only AP is equivalent at ±0.099. The others are inconclusive at that margin, not "no gap". Under M1, the Reuters event
gap is −32 [−52, −12], with the generated voice more "left". The judge sees no such difference (−0.03), another case of
the two measures diverging.

The event descriptions were written from the real tweets, which opens a channel between the two. But generated tweets
overlap lexically with the description (word Jaccard 0.51) far more than with the same outlet's real tweets (0.21).
That is lower than the description's own overlap with them (0.25). Generated tweets paraphrase the description; they do
not copy the real tweets.

### 8.4 Word-level differences (exploratory)
Log-odds with an informative Dirichlet prior show the genre gap at the word level.
- **Generated voices overuse:**
  - second-person and civic words ("your", "you", "voters", "where … stand", "candidates", "check", "polling",
    "know");
  - issue nouns ("economy", "abortion", "immigration", "inflation", "border").
- **Real outlets overuse** people and events: "Biden", "Trump", "Harris", "president", "former", "says", "after",
  "GOP", "House".

### 8.5 Sensitivity checks
- **Near-duplicate threshold:** clustering at word-3-gram Jaccard 0.3 or 0.7 instead of 0.5 leaves every Table 1
  estimate and interval essentially unchanged (Fox slant +0.66 [0.62, 0.74] and [0.62, 0.70]).
- **Uncleaned real set:** using all 2,500 real tweets gives the same outlet slant levels within 0.03.

## 9. Results III: optimising against the bias proxy (RL)

[Pending: 7 runs of 300 steps on Qwen2.5-0.5B: 3 calibrated, 2 format-only, 1 shuffled, 1 pooled. Interim reports are
in `docs/paper/rl_interim.md`. This section will report:
- divergence from each outlet's real distribution under M1 (the reward), M3 and the judge, on training topics,
  held-out topics and a reworded template, at the pre-registered step-300 checkpoint;
- transfer ratios against the format-only control;
- the pre-registered reward-hacking diagnostics;
- Goodhart curves (Figure 7).
The non-Claude judge results also go here, because that judge shares the GPU with RL.]

## 10. Discussion

**What the measurement results mean for this literature.** Political-leaning classifiers are convenient for
large-scale comparisons of real and generated text. We found two artifacts in one such classifier that are large
enough to manufacture findings:
- a URL suffix;
- party names in otherwise neutral text.

Both are invisible in aggregate accuracy figures, and both interact with the very differences being studied: real
tweets carry links and name parties, while generated tweets rarely do. Any comparison of real and generated political
text with such a classifier needs, at minimum:
- link-stripped inputs;
- a party-swap check;
- a second, differently built measure.

**What the model does when asked to "be" an outlet.** Asked only for a topic and a voice, Haiku changes **genre and
agenda** far more than **framing**. It writes explainers, reminders and promotions rather than news reports.
- **Fox News** is the only voice it gives a distinct framing, and it locates that framing in opinion and promotional
  content rather than in news reporting.
- **The mainstream outlets** are rendered as a single neutral register. They lose the measurable left tilt that real
  NYT tweets, and in the general-election period ABC, AP and Reuters tweets, carry under the same judge.

This is the "flattening and misportrayal" pattern documented for LLM portrayals of social groups (Wang et al. 2025),
here for institutions. It is an asymmetric stereotype: one outlet is coded as partisan, the others as neutral.

When the news is fixed, the voices converge on near-identical paraphrases. This matches the intuition behind the
study's design: giving the model facts largely removes its own picture of the outlet.

**What we can and cannot claim.** The framing results are "as perceived by a Claude judge". That judge identifies Fox
tweets from style alone, so its slant ratings and its recognition of the source are entangled. Human coding of the
prepared sample is the next step, and it could change the framing conclusions, especially for the small tilts of
mainstream outlets.

## 11. Limitations

1. **No human labels yet.** Judge validity is unestablished. Both validated judges are Claude models, like the
   generator, and the non-Claude judge is pending.
2. **One generator family for the main analyses.** Sonnet and Opus are robustness variants. The Qwen base model is the
   RL base, not a comparison generator.
3. **The real sample** is the election-keyword subset of each outlet's feed in one dataset. Fox (55%) and AP (61%)
   tweets fall disproportionately before 21 July.
4. **Prompt wording matters.**
   - "Voting and the election" produced all of the voting reminders.
   - "It is 2024" places the model in the Biden-versus-Trump period: Harris appears in 6% of generated tweets against
     20% of real ones.
   - The month-anchored variant only partly corrects this.
5. **Event descriptions** were written by a Claude model from the real tweets; lexical overlap analysis suggests this
   channel is weak.
6. **Data-driven equivalence margin.** The ±0.099 margin was set after the data were seen (logged as a deviation). The
   pre-registered ±0.4 margin is uninformative.
7. **The RL study** uses a 0.5B model on consumer hardware, few seeds, and a single reward model. Its conclusions
   concern optimisation against this proxy, not the outlets.

## 12. Ethics

This study generates text impersonating real news organisations:
- **Labelling and storage:** generated tweets are stored only as labelled research data, and published examples are
  masked.
- **RL adapters:** these are trained to imitate outlets and are not released.
- **Real outlet distributions** are used descriptively, never as normative targets.
- **Consent:** real outlets cannot consent to being imitated.
- **Real tweets:** the repository currently contains the full text of the source dataset. Following X's terms and
  common practice, a public release should include tweet IDs only (flagged to the authors).

## 13. Reproducibility

Everything in this paper is reproduced by the commands below, from the repository root with the project environment.
API steps read the key from the macOS Keychain.

1. `python -m study.clean_real` — cleaned real set
2. `python -m study.generate --conditions …` and `python -m study.events candidates|adjudicate|generate` — generations
3. `python -m study.measures --input …` — classifiers (M1, M3, M3b, M5b, topic)
4. `python -m study.judge rate|identify --set …` — judges, via the Batch API
5. `python -m study.swap build` — counterfactuals
6. `python -m study.data` — unit table
7. `python -m study.analyze --B 10000`, `python -m study.analyze_extra`, `python -m study.reliability` — analyses
8. `python -m study.figures` — figures
9. `sh study/rl_queue.sh`, `python -m study.rl_eval`, `python -m study.rl_analyze score|report` — RL

The protocol and its change log are in `docs/protocol_v2.md`. `study/audit_barriers.py` verifies the information
barriers. Total API cost was about US$35.

## References

- Argyle, L. et al. (2023). Out of one, many: Using language models to simulate human samples. *Political Analysis*.
- Bakshy, E., Messing, S., & Adamic, L. (2015). Exposure to ideologically diverse news and opinion on Facebook. *Science*.
- Baly, R. et al. (2020). We can detect your bias: Predicting the political ideology of news articles. *EMNLP*.
- Bavaresco, A. et al. (2025). LLMs instead of human judges? *ACL*. arXiv:2406.18403.
- Bisbee, J. et al. (2024). Synthetic replacements for human survey data? The perils of large language models. *Political Analysis*.
- Burnham, M. et al. (2024). Political DEBATE: Efficient zero-shot and few-shot classifiers for political text. arXiv:2409.02078.
- Cheng, M. et al. (2023). CoMPosT: Characterizing and evaluating caricature in LLM simulations. *EMNLP*.
- Egami, N. et al. (2023). Using imperfect surrogates for downstream inference: Design-based supervised learning. *NeurIPS*.
- Gao, L., Schulman, J., & Hilton, J. (2023). Scaling laws for reward model overoptimization. *ICML*.
- Horych, T. et al. (2024). MAGPIE: Multi-task analysis of media-bias generalization. *LREC-COLING*.
- Liu, Z. et al. (2025). Understanding R1-zero-like training: A critical perspective (Dr. GRPO). *COLM*. arXiv:2503.20783.
- Monroe, B., Colaresi, M., & Quinn, K. (2008). Fightin' words. *Political Analysis*.
- Pan, A., Bhatia, K., & Steinhardt, J. (2022). The effects of reward misspecification. *ICLR*.
- Panickssery, A., Bowman, S., & Feng, S. (2024). LLM evaluators recognize and favor their own generations. *NeurIPS*.
- Robertson, R. et al. (2018). Auditing partisan audience bias within Google Search. *CSCW*.
- Santurkar, S. et al. (2023). Whose opinions do language models reflect? *ICML*.
- Wang, A. et al. (2025). Large language models that replace human participants can harmfully misportray and flatten identity groups. *Nature Machine Intelligence*.
- Zheng, L. et al. (2023). Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. *NeurIPS*.
- Source dataset: USC X 2024 US election dataset, arXiv:2411.00376. Leaning classifier: arXiv:2507.13913.
