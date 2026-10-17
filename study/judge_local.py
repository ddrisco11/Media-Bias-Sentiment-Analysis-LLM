"""Non-Claude judge (M2c): Qwen2.5-7B-Instruct run locally with the same rubric and masking as M2.

Unlike M2, every item uses the normal (unflipped) instrument: in a first pass the 7B model mishandled the reversed
scale (12 of 40 replies invalid), and the flip exists only to detect order effects in the primary judge.

The JSON reply is validated against the same schema rules. Invalid replies get one retry, then count as missing.
Usage: python -m study.judge_local --items results/v2/human/human_set.jsonl
Output: results/v2/judge/qwen2.5-7b/<set>/rate/results.jsonl (same format as the batch judge)
"""
import argparse
import json
import re

from study.common import RESULTS, read_jsonl, write_jsonl
from study.judge import FLIP_NOTE, GENRES, RUBRIC

MODEL = 'Qwen/Qwen2.5-7B-Instruct'
FORMAT = ('\n\nReply with a single JSON object and nothing else, with keys in this order: "rationale" (one sentence), '
          '"genre" (one of: {genres}), {stance_keys}, {valence_keys}, "slant" (integer -3..3), "plausibility" (integer 1..5). '
          'Stance values are strings: "-2","-1","0","1","2" or "NA". Event valence values: "bad","neutral","good" or "NA".')


def keys(flipped):
    parties = ['republicans', 'democrats'] if flipped else ['democrats', 'republicans']
    return (', '.join(f'"stance_{p}"' for p in parties), ', '.join(f'"event_valence_{p}"' for p in parties))


def valid(d):
    try:
        return (d['genre'] in GENRES and d['stance_democrats'] in ('-2', '-1', '0', '1', '2', 'NA') and
                d['stance_republicans'] in ('-2', '-1', '0', '1', '2', 'NA') and int(d['slant']) in range(-3, 4) and
                int(d['plausibility']) in range(1, 6))
    except (KeyError, TypeError, ValueError):
        return False


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--items', required=True, help='JSONL with "item" and "masked_text" fields')
    parser.add_argument('--set', default='human-set')
    args = parser.parse_args()
    items = read_jsonl(args.items)
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to('mps').eval()
    out_dir = RESULTS / 'judge' / 'qwen2.5-7b' / args.set / 'rate'
    done = {r['custom_id'] for r in read_jsonl(out_dir / 'results.jsonl')} if (out_dir / 'results.jsonl').exists() else set()
    rows = read_jsonl(out_dir / 'results.jsonl') if done else []
    todo = [it for it in items if it['item'] not in done]
    write_jsonl(out_dir / 'items.jsonl', [{'item': it['item'], 'masked_text': it['masked_text'], 'flipped': False} for it in items])
    for start in range(0, len(todo), 8):
        batch = todo[start:start + 8]
        prompts = []
        for it in batch:
            flipped = False
            sk, vk = keys(flipped)
            user = f'Post to rate:\n"""{it["masked_text"]}"""' + (FLIP_NOTE if flipped else '') + \
                FORMAT.format(genres=', '.join(GENRES), stance_keys=sk, valence_keys=vk)
            prompts.append(tok.apply_chat_template([{'role': 'system', 'content': RUBRIC}, {'role': 'user', 'content': user}],
                                                   tokenize=False, add_generation_prompt=True))
        for attempt in range(2):
            enc = tok(prompts, return_tensors='pt', padding=True).to('mps')
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=300, do_sample=False, pad_token_id=tok.pad_token_id)
            texts = tok.batch_decode(gen[:, enc['input_ids'].shape[1]:], skip_special_tokens=True)
            parsed = []
            for t in texts:
                m = re.search(r'\{.*\}', t, re.S)
                try:
                    d = json.loads(m.group(0)) if m else None
                except json.JSONDecodeError:
                    d = None
                parsed.append(d if d and valid(d) else None)
            if all(parsed) or attempt == 1:
                break
        for it, d, t in zip(batch, parsed, texts):
            if d:
                d['slant'], d['plausibility'] = int(d['slant']), int(d['plausibility'])
            rows.append({'custom_id': it['item'], 'type': 'succeeded' if d else 'invalid', 'parsed': d, 'text': t, 'model': MODEL})
        write_jsonl(out_dir / 'results.jsonl', rows)
        print(f'{len(rows)}/{len(items)}', flush=True)


if __name__ == '__main__':
    main()
