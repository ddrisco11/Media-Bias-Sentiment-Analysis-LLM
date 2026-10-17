"""Party-swap test (protocol H3): exchange Democratic and Republican names and see how much each measure moves.

A measure that tracks framing should move little on neutral items, where swapping names leaves the framing
unchanged. A measure that tracks names will move regardless. Two item sources:
  1. Real and generated tweets naming a party, with names swapped by fixed rules. Items are skipped when the swap is
     ambiguous (all of Biden, Harris and Trump present) or involves family members.
  2. 30 built neutral templates with a party slot, each filled both ways.
Usage: python -m study.swap build   (writes results/v2/swap/items.jsonl and the template items)
"""
import argparse
import re

from study.common import RESULTS, ROOT, read_jsonl, scoring_text, write_jsonl

SKIP = re.compile(r'\b(hunter|melania|jill biden|emhoff|jr\.?|ivanka|barron|eric trump|lara trump|maga)\b', re.I)
TOKEN = '⟦{}⟧'


def swap_text(text):
    """Return the party-swapped text, or None if the swap is ambiguous."""
    if SKIP.search(text):
        return None
    has = {n: bool(re.search(rf'\b{n}\b', text, re.I)) for n in ('trump', 'harris', 'biden', 'vance', 'walz')}
    if has['harris'] and has['biden']:
        return None  # two Democratic candidates: no clean one-to-one swap (audit, 2026-10-17)
    dem = 'biden' if has['biden'] and not has['harris'] else 'harris'
    dem_first, dem_full = ('Joe', 'Joe Biden') if dem == 'biden' else ('Kamala', 'Kamala Harris')
    pairs = [(r'\bDonald Trump\b', dem_full), (rf'\b{dem_full}\b', 'Donald Trump'), (r'\bDonald\b', dem_first),
             (rf'\b{dem_first}\b', 'Donald'), (r'\bTrump\b', dem.capitalize()), (rf'\b{dem.capitalize()}\b', 'Trump'),
             (r'\bTrump\'s\b', dem.capitalize() + "'s"), (r'\bJD Vance\b|\bJ\.D\. Vance\b', 'Tim Walz'), (r'\bTim Walz\b', 'JD Vance'),
             (r'\bVance\b', 'Walz'), (r'\bWalz\b', 'Vance'), (r'\bRepublicans\b', 'Democrats'), (r'\bDemocrats\b', 'Republicans'),
             (r'\bRepublican\b', 'Democratic'), (r'\bDemocratic\b', 'Republican'), (r'\bGOP\b', 'Democratic Party'),
             (r'\bDemocrat\b', 'Republican'), (r'\bRNC\b', 'DNC'), (r'\bDNC\b', 'RNC')]
    out, slots = text, []

    def put_case(m, replacement):
        slots.append(replacement.upper() if m.group(0).isupper() and len(m.group(0)) > 3 else replacement)
        return TOKEN.format(len(slots) - 1)
    for i, (pattern, replacement) in enumerate(pairs):
        out = re.sub(pattern, lambda m, i=i, replacement=replacement: put_case(m, replacement), out, flags=re.I)
    for i, replacement in enumerate(slots):
        out = out.replace(TOKEN.format(i), replacement)
    return out if out != text else None


TEMPLATES = [
    '{C} will hold a campaign rally in Pennsylvania on Saturday, the campaign said.',
    '{C} is scheduled to speak in Atlanta on Thursday evening.',
    '{C} met with campaign staff in Philadelphia on Monday ahead of the debate.',
    'The {P} campaign announced a new round of television ads in Arizona and Nevada.',
    '{C} released a list of upcoming campaign stops for the final week of October.',
    '{C} will appear at a town hall in Michigan next week, organizers said.',
    'A spokesperson for {C} confirmed the candidate will attend the event on Friday.',
    '{P} lawmakers returned to Washington on Tuesday after the August recess.',
    '{C} arrived in Wisconsin on Wednesday for a two-day campaign swing.',
    'The {P} National Committee reported its monthly fundraising totals on Tuesday.',
    '{C} spoke with reporters briefly before boarding a flight to North Carolina.',
    'Campaign officials said {C} will release a detailed schedule for debate week.',
    '{C} toured a manufacturing plant in Ohio on Monday afternoon.',
    'The {P} convention will be held in August, party officials confirmed.',
    '{C} is expected to address supporters in Las Vegas this weekend.',
    '{P} officials in Georgia announced new voter outreach offices on Friday.',
    '{C} participated in a radio interview on Thursday morning.',
    'A new poll shows {C} with 46% support among likely voters in Pennsylvania, within the margin of error.',
    '{C} named a new campaign manager on Wednesday, the campaign said in a statement.',
    'The {P} Party filed paperwork for its slate of electors in Arizona on Monday.',
    '{C} will visit three swing states over the next five days.',
    '{P} leaders scheduled a press conference for 11 a.m. on Capitol Hill.',
    '{C} received a briefing from campaign advisers on Sunday, aides said.',
    '{C} is set to sit for a televised interview airing Sunday night.',
    'The {P} campaign said it raised $50 million in the past month.',
    '{C} attended a church service in Detroit on Sunday morning.',
    '{C} will deliver remarks on the economy in Pittsburgh on Tuesday.',
    '{P} officials in Nevada said early voting sites will open on Saturday.',
    '{C} spent Monday preparing for the upcoming debate, according to the campaign.',
    '{C} thanked volunteers at a field office in Milwaukee on Saturday.',
]
FILLS = {'D': {'C': 'Kamala Harris', 'P': 'Democratic'}, 'R': {'C': 'Donald Trump', 'P': 'Republican'}}


def build():
    sources = {'real': (RESULTS / 'real' / 'real_tweets.jsonl', lambda r: r['included'], 'outlet'),
               'main-v1': (ROOT / 'dataset' / 'synthetic' / 'main-v1' / 'generations.jsonl', lambda r: True, 'voice'),
               'sens-ref': (RESULTS / 'generations' / 'sens-ref' / 'generations.jsonl', lambda r: True, 'voice'),
               'event-v1': (RESULTS / 'generations' / 'event-v1' / 'generations.jsonl', lambda r: True, 'voice')}
    items = []
    for source, (path, keep, group) in sources.items():
        for r in read_jsonl(path):
            if not keep(r):
                continue
            text = scoring_text(r['text'])
            swapped = swap_text(text)
            if swapped:
                items.append({'source': source, 'group': r[group], 'original': text, 'swapped': swapped, 'kind': 'tweet'})
    for i, t in enumerate(TEMPLATES):
        items.append({'source': 'template', 'group': f't{i:02d}', 'original': t.format(**FILLS['D']),
                      'swapped': t.format(**FILLS['R']), 'kind': 'template'})
    write_jsonl(RESULTS / 'swap' / 'items.jsonl', items)
    write_jsonl(RESULTS / 'swap' / 'texts_for_judge.jsonl', [{'text': i['swapped']} for i in items] +
                [{'text': i['original']} for i in items if i['kind'] == 'template'])
    print(f"{len(items)} swap pairs: " + str({s: sum(i['source'] == s for i in items) for s in {i['source'] for i in items}}))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('step', choices=['build'])
    parser.parse_args()
    build()


if __name__ == '__main__':
    main()
