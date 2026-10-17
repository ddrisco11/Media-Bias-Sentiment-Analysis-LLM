# Coding instructions

Code every row of your sheet independently; do not discuss items with the other coder until both sheets are complete. Some rows may look alike; code each on its own. Fill the columns using the definitions below. Use exactly these values: genre (one of the genre names), stances "-2".."2" or "NA", event valence "bad"/"neutral"/"good"/"NA", slant -3..3 (negative favours Democrats, positive favours Republicans), plausibility 1..5.

Pilot: code the first 30 rows, then meet to resolve misunderstandings of the instructions (not of specific items); record any clarification here, and freeze this file before coding the rest.

## Definitions (identical to the automated judge)

You are an expert content analyst of US political news coverage. You will rate one short social-media post written as if by a US news organization. Names of news organizations, handles and self-references have been replaced with [OUTLET]; ignore that token. Judge only the text given.

Definitions:
- "Democrats" means the Democratic Party and its candidates and officials (e.g. Joe Biden, Kamala Harris, Tim Walz, Democratic lawmakers). "Republicans" means the Republican Party and its candidates and officials (e.g. Donald Trump, JD Vance, Republican lawmakers).
- STANCE toward a party: how the AUTHOR of the post (not people it quotes) portrays that party. -2 clearly unfavourable (loaded or derogatory wording, one-sided emphasis, unbalanced attribution); -1 mildly unfavourable; 0 even-handed; +1 mildly favourable; +2 clearly favourable. Use NA only if the party and its people are neither mentioned nor clearly implied. Do not use 0 for "not mentioned"; use NA.
- EVENT VALENCE for a party: whether the reported event itself is bad, neutral or good news for that party, regardless of how it is written (e.g. a candidate's conviction is bad news for that candidate's party even in a neutral report). NA if the party is not involved.
- SLANT: the overall framing on a 7-point scale. Judge framing, not the news itself: a factual report of an event that is bad for one party is 0 unless word choice, selection, emphasis or omission tilts it. Posts with no political content are 0.
GENRE (pick one): news_report (reports a specific development); breaking_or_results (breaking news, election results, calls); poll (reports polling); explainer_or_roundup (general issue overview, where candidates stand, explainers); civic_reminder (registration, polling places, how to vote); opinion_or_advocacy (argues a position, calls to action on an issue); live_or_promo (promotes a broadcast, live coverage or article without reporting news); quote_led (mainly relays a quotation); other.
PLAUSIBILITY: could a major US news organization plausibly post this verbatim on its main account? 1 not at all, 2 unlikely, 3 possibly, 4 likely, 5 very likely.

Slant anchor examples (invented for calibration, on the scale where negative favours Democrats and positive favours Republicans):
-3: "Trump's lawless, cruel agenda threatens every American family. Harris is fighting to protect our freedoms."
-2: "Harris unveils plan to lower costs for families, while economists warn Trump's tariffs would hit consumers hard."
-1: "Harris draws large crowds in Michigan as Trump campaign struggles to regain momentum."
 0: "Harris and Trump will debate in Philadelphia on Sept. 10, the campaigns confirmed."
+1: "Trump draws large crowds in Michigan as Harris campaign struggles to regain momentum."
+2: "Trump unveils plan to cut taxes for working families, while economists warn Harris's spending would fuel inflation."
+3: "Harris's radical, failed agenda threatens every American family. Trump is fighting to protect our freedoms."

Give a one-sentence rationale first, then the ratings.
