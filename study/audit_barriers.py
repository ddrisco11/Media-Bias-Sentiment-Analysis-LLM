"""Information-barrier audit (protocol section 10): evaluated systems never see evaluation criteria, and evaluators
never see source or condition labels.

Checks:
1. Every generation prompt (all runs, including RL prompts) is free of evaluation vocabulary. Matches are listed for
   manual review, because common words ("civil rights", "right ear") trigger the pattern harmlessly.
2. Judge and identification inputs contain only an item ID, masked text and the flip flag: no source, voice,
   condition, outlet, model or score fields, and no unmasked outlet names.
3. Human-labelling sheets contain no source, condition, classifier or judge fields.
Usage: python -m study.audit_barriers   (writes results/v2/validity/barrier_audit.json)
"""
from pathlib import Path
import re

from study.common import RESULTS, ROOT, read_jsonl, write_json, now
from study.mask import _outlets

EVAL_TERMS = re.compile(r'\b(slant|stance|left|right|center|centre|bias(?:ed)?|partisan|rubric|classif\w*|judge|evaluat\w*|'
                        r'score|leaning|liberal|conservative|ideolog\w*|reward)\b', re.I)
# Phrases that contain evaluation words but are topic or event content, not criteria.
BENIGN = re.compile(r'civil rights|right ear|judge [A-Z]|federal judge|centers for|conservative political action|CPAC', re.I)
LABEL_FIELDS = {'voice', 'outlet', 'condition', 'source', 'model', 'leaning', 'label', 'score', 'run', 'arm'}


def prompt_hits(prompt):
    body = prompt.split('about this news:')[0]
    return [m.group(0) for m in EVAL_TERMS.finditer(BENIGN.sub(' ', body))]


def main():
    report = {'created_at': now(), 'generation': {}, 'judge_inputs': {}, 'human_sheets': {}}
    files = list((RESULTS / 'generations').glob('*/generations.jsonl')) + \
        [ROOT / 'dataset' / 'synthetic' / r / 'generations.jsonl' for r in ('main-v1', 'pilot-v1')]
    prompts_by_file = {str(p.relative_to(ROOT)): {r['prompt'] for r in read_jsonl(p)} for p in files if p.exists()}
    for name, prompts in prompts_by_file.items():
        hits = {p[:120]: prompt_hits(p) for p in prompts if prompt_hits(p)}
        report['generation'][name] = {'unique_prompts': len(prompts), 'prompts_with_terms': len(hits), 'examples': dict(list(hits.items())[:5])}
    for items in RESULTS.glob('judge/*/*/*/items.jsonl'):
        rows = read_jsonl(items)
        fields = set().union(*(r.keys() for r in rows)) if rows else set()
        unmasked = sum(bool(_outlets.search(r['masked_text'])) for r in rows)
        report['judge_inputs'][str(items.relative_to(ROOT))] = {'items': len(rows), 'fields': sorted(fields),
                                                                'label_fields_present': sorted(fields & LABEL_FIELDS),
                                                                'unmasked_outlet_mentions': unmasked}
    for sheet in (RESULTS / 'human').glob('*sheet*.csv'):
        header = sheet.read_text(encoding='utf-8').splitlines()[0].lower()
        report['human_sheets'][sheet.name] = {'header': header,
                                              'label_fields_present': [f for f in LABEL_FIELDS if f in header.split(',')]}
    report['passed'] = (all(v['prompts_with_terms'] == 0 for v in report['generation'].values()) and
                        all(not v['label_fields_present'] and v['unmasked_outlet_mentions'] == 0 for v in report['judge_inputs'].values()) and
                        all(not v['label_fields_present'] for v in report['human_sheets'].values()))
    write_json(RESULTS / 'validity' / 'barrier_audit.json', report)
    print('passed' if report['passed'] else 'NEEDS REVIEW')
    for section in ('generation', 'judge_inputs', 'human_sheets'):
        for k, v in report[section].items():
            print(section, k, {x: y for x, y in v.items() if x not in ('examples', 'fields', 'header')})


if __name__ == '__main__':
    main()
