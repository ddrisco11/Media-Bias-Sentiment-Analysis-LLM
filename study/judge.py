"""LLM judge (M2) and outlet identification (M6), run through the Message Batches API.

Every item is masked first (study/mask.py), so the judge never sees an outlet name, handle or self-reference.
Half of the items (chosen by a hash of the text) get the "flipped" instrument: Republican fields come first and the
slant scale is reversed (+3 = strongly favours Democrats). Flipped answers are mapped back before analysis, which
lets the analysis test for order and anchor effects.

Usage:
  python -m study.judge rate --set <name> --input <jsonl> [...] [--model claude-sonnet-5-5] [--retest]
  python -m study.judge identify --set <name> --input <jsonl> [...]
Results: results/v2/judge/<model>/<set>/{items.jsonl,results.jsonl,batches.json}
"""
import argparse
import hashlib

from study.common import RESULTS, read_jsonl, write_jsonl
from study.llm import params, run_batch
from study.mask import mask_outlets

RUBRIC = """You are an expert content analyst of US political news coverage. You will rate one short social-media post written as if by a US news organization. Names of news organizations, handles and self-references have been replaced with [OUTLET]; ignore that token. Judge only the text given.

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

Give a one-sentence rationale first, then the ratings."""

FLIP_NOTE = ("\n\nIMPORTANT for this item: the SLANT scale is reversed. Use -3 for posts that strongly favour REPUBLICANS and +3 "
             "for posts that strongly favour DEMOCRATS (the anchors above apply with their signs flipped).")
STANCE = {'type': 'string', 'enum': ['-2', '-1', '0', '1', '2', 'NA']}
VALENCE = {'type': 'string', 'enum': ['bad', 'neutral', 'good', 'NA']}
GENRES = ['news_report', 'breaking_or_results', 'poll', 'explainer_or_roundup', 'civic_reminder', 'opinion_or_advocacy',
          'live_or_promo', 'quote_led', 'other']


def schema(flipped):
    parties = ['republicans', 'democrats'] if flipped else ['democrats', 'republicans']
    props = {'rationale': {'type': 'string'}, 'genre': {'type': 'string', 'enum': GENRES}}
    for p in parties:
        props[f'stance_{p}'] = STANCE
    for p in parties:
        props[f'event_valence_{p}'] = VALENCE
    props['slant'] = {'type': 'integer', 'enum': [-3, -2, -1, 0, 1, 2, 3]}
    props['plausibility'] = {'type': 'integer', 'enum': [1, 2, 3, 4, 5]}
    return {'type': 'object', 'additionalProperties': False, 'required': list(props), 'properties': props}


def is_flipped(text):
    return int(hashlib.sha256(('flip:' + text).encode()).hexdigest(), 16) % 2 == 1


def rate_request(model, masked, flipped):
    prompt = f'Post to rate:\n"""{masked}"""' + (FLIP_NOTE if flipped else '')
    p = params(model, prompt, schema(flipped), max_tokens=1024, system=[{'type': 'text', 'text': RUBRIC,
                                                                        'cache_control': {'type': 'ephemeral'}}])
    return p


def unflip(parsed, flipped):
    if parsed is None:
        return None
    out = dict(parsed)
    if flipped:
        out['slant'] = -parsed['slant']
    return out


IDENTIFY = """Below is a short social-media post from the main X account of one of five US news organizations. Names of news organizations, handles and self-references have been replaced with [OUTLET].

Post:
\"\"\"{text}\"\"\"

Which organization most likely posted it? Choose from: nytimes (The New York Times), foxnews (Fox News), reuters (Reuters), ap (Associated Press), abc (ABC News). Give the single most likely answer and your confidence (0-100) that it is correct."""
ID_SCHEMA = {'type': 'object', 'additionalProperties': False, 'required': ['answer', 'confidence'],
             'properties': {'answer': {'type': 'string', 'enum': ['nytimes', 'foxnews', 'reuters', 'ap', 'abc']},
                            'confidence': {'type': 'integer'}}}


def load_items(inputs):
    items = {}
    for path in inputs:
        for r in read_jsonl(path):
            text = r.get('text') or ''
            if not text.strip():
                continue
            masked = mask_outlets(text)
            if masked:
                items[hashlib.sha256(masked.encode()).hexdigest()[:24]] = masked
    return items


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('task', choices=['rate', 'identify'])
    parser.add_argument('--set', required=True)
    parser.add_argument('--input', nargs='+', required=True)
    parser.add_argument('--model', default='claude-sonnet-5-5')
    parser.add_argument('--retest', action='store_true', help='Separate results directory for test-retest')
    args = parser.parse_args()
    items = load_items(args.input)
    out = RESULTS / 'judge' / args.model / (args.set + ('-retest' if args.retest else '')) / args.task
    write_jsonl(out / 'items.jsonl', [{'item': k, 'masked_text': v, 'flipped': is_flipped(v) if args.task == 'rate' else None}
                                      for k, v in items.items()])
    if args.task == 'rate':
        reqs = {k: rate_request(args.model, v, is_flipped(v)) for k, v in items.items()}
    else:
        reqs = {k: params(args.model, IDENTIFY.format(text=v), ID_SCHEMA, max_tokens=256) for k, v in items.items()}
    print(f'{args.task} {args.set}: {len(reqs)} unique masked items on {args.model}')
    run_batch(out, reqs)


if __name__ == '__main__':
    main()
