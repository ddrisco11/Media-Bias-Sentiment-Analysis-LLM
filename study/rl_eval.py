"""Evaluate RL checkpoints (protocol section 7): sample, then score with reward and held-out measures.

For a run and step (or the base model), samples:
  - training topics: n_train per outlet x topic (default 20); held-out topics: n_held per outlet x topic (default 50);
  - the final step also under reworded template t1 (10 per outlet x topic, all 14 topics);
  - the base model additionally samples the neutral voice (model-qwen condition for RQ4).
Sampling matches training: temperature 1.0, top_p 1, top_k 0, no repetition penalty, 96 new tokens, chat template.
Each sample records its format status and its mean token negative log-likelihood under the base (reference) model.
Usage: python -m study.rl_eval --base
       python -m study.rl_eval --run calibrated-s1 --steps 50 100 200 300
Output: results/v2/rl_eval/<run>/step-<n>.jsonl
"""
import argparse
import random
import zlib

from scripts.generate_synthetic_tweets import TOPIC_WORDING, VOICES
from study.common import OUTLETS, RESULTS, ROOT, now, write_jsonl
from study.generate import TEMPLATES
from study.rl import HELD_OUT_TOPICS, TEMPLATE, TRAIN_TOPICS, CONFIG, format_fail

BASE_MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'


def jobs(n_train, n_held, final, base):
    out = []
    voices = OUTLETS + (['neutral'] if base else [])
    for v in voices:
        for t in TRAIN_TOPICS:
            out += [{'voice': v, 'topic': t, 'split': 'train', 'template': 'v1', 'prompt': TEMPLATE.format(voice=VOICES[v], topic=TOPIC_WORDING[t])}] * n_train
        for t in HELD_OUT_TOPICS:
            out += [{'voice': v, 'topic': t, 'split': 'heldout', 'template': 'v1', 'prompt': TEMPLATE.format(voice=VOICES[v], topic=TOPIC_WORDING[t])}] * n_held
        if final:
            for t in TOPIC_WORDING:
                out += [{'voice': v, 'topic': t, 'split': 'train' if t in TRAIN_TOPICS else 'heldout', 'template': 't1',
                         'prompt': TEMPLATES['t1'].format(voice=VOICES[v], topic=TOPIC_WORDING[t])}] * 10
    return [dict(j) for j in out]


def evaluate(run, step, n_train=20, n_held=50, final=False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    base = run == 'base'
    seed = zlib.crc32(f"{run}:{step}".encode()) if not base else 20241016
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16).to('mps')
    if not base:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, ROOT / 'artifacts' / 'rl' / run / f'step-{step}')
    model.eval()
    js = jobs(n_train, n_held, final or base, base)
    random.Random(seed).shuffle(js)
    stop_ids = {tok.eos_token_id, tok.convert_tokens_to_ids('<|im_end|>')}
    rows = []
    for i in range(0, len(js), 64):
        chunk = js[i:i + 64]
        chats = [tok.apply_chat_template([{'role': 'user', 'content': j['prompt']}], tokenize=False, add_generation_prompt=True) for j in chunk]
        enc = tok(chats, return_tensors='pt', padding=True).to('mps')
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=CONFIG['max_new_tokens'], do_sample=True, temperature=1.0, top_p=1.0,
                                 top_k=0, repetition_penalty=1.0, pad_token_id=tok.pad_token_id)
            plen = enc['input_ids'].shape[1]
            comp = out[:, plen:]
            mask = torch.ones_like(comp)
            finished = []
            for r in range(comp.shape[0]):
                ids = comp[r].tolist()
                stop = next((k for k, t in enumerate(ids) if t in stop_ids), None)
                finished.append(stop is not None)
                if stop is not None:
                    mask[r, stop + 1:] = 0
            attn = torch.cat([enc['attention_mask'], mask], 1)
            # Mean token NLL of the sampled completion under the reference (base) model.
            ctx = model.disable_adapter() if not base else torch.no_grad()
            with ctx:
                logits = model(input_ids=out, attention_mask=attn).logits[:, plen - 1:-1].float()
            lp = torch.log_softmax(logits, -1).gather(-1, comp.unsqueeze(-1)).squeeze(-1)
            nll = (-(lp * mask).sum(1) / mask.sum(1)).cpu().tolist()
        for r, j in enumerate(chunk):
            text = tok.decode(comp[r][mask[r].bool()], skip_special_tokens=True).strip()
            rows.append({**j, 'run': run, 'step': step, 'text': text, 'finished': finished[r],
                         'format_fail': format_fail(text, finished[r]), 'ref_nll': nll[r]})
    path = RESULTS / 'rl_eval' / run / f'step-{step}.jsonl'
    write_jsonl(path, rows)
    print(f'{run} step {step}: {len(rows)} samples -> {path} ({now()})', flush=True)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run')
    parser.add_argument('--base', action='store_true')
    parser.add_argument('--steps', nargs='+', type=int, default=[50, 100, 200, 300])
    args = parser.parse_args()
    if args.base:
        evaluate('base', 0)
        return
    for s in args.steps:
        # Intermediate checkpoints get a smaller sample; the pre-registered final checkpoint gets the full one.
        if s == 300:
            evaluate(args.run, s, final=True)
        else:
            evaluate(args.run, s, n_train=5, n_held=10)


if __name__ == '__main__':
    main()
