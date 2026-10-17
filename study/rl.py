"""RL study (protocol section 7): optimise an open model's outlet voices against the leaning classifier (M1).

Reward:  r(x) = sum_l P_M1(l|x) * (p_o(l) - q_o(l)) - 1[format fail] - beta * sum_t sg(log pi - log pi_ref)
  p_o  outlet o's smoothed label shares on the cleaned real tweets (the condition decides which outlet's target)
  q_o  the policy's own label marginal for o: this step's batch estimate shrunk toward the previous one (kappa=16)
Advantages are group-mean-centred (no std division). One on-policy update per batch. LoRA on all linear layers;
reference log-probabilities come from the same model with the adapter disabled.

Conditions: calibrated (own outlet target), format (no leaning reward), shuffled (cyclic derangement of targets),
pooled (every outlet gets the pooled real target).

Usage: python -m study.rl --condition calibrated --seed 1 --model Qwen/Qwen2.5-1.5B-Instruct
       python -m study.rl --condition calibrated --seed 1 --steps 10 --timing   (feasibility pilot)
Adapters are written under artifacts/rl/ (git-ignored, never released); logs and manifests under results/v2/rl/.
"""
import argparse
from collections import Counter
import json
import math
import random
import time

from scripts.generate_synthetic_tweets import TOPIC_WORDING, VOICES, flags
from study.common import LABELS, OUTLETS, RESULTS, ROOT, now, read_jsonl, scoring_text, write_json
from study.measures import load_cache, text_hash

TEMPLATE = ('It is 2024, during the US presidential campaign. Write one tweet that {voice} might post on its main X '
            'account about {topic}. Reply with only the tweet text: 280 characters or fewer, no links, no hashtags.')
HELD_OUT_TOPICS = ['immigration', 'economy', 'gender rights', 'environment']
TRAIN_TOPICS = [t for t in TOPIC_WORDING if t not in HELD_OUT_TOPICS]
DERANGEMENT = {'nytimes': 'foxnews', 'foxnews': 'nytimes', 'abc': 'ap', 'ap': 'reuters', 'reuters': 'abc'}
CONFIG = dict(group_size=8, prompts_per_step=10, lr=2e-5, beta=0.05, kappa=16, steps=300, warmup=10, clip=1.0,
              lora_r=16, lora_alpha=32, max_new_tokens=96, temperature=1.0, init_samples_per_outlet=200,
              checkpoints=[50, 100, 200, 300])


def real_targets():
    """Smoothed M1 label shares and predicted-topic mix per outlet on the cleaned real tweets."""
    real = [r for r in read_jsonl(RESULTS / 'real' / 'real_tweets.jsonl') if r['included']]
    leaning, topics = load_cache('leaning'), load_cache('topic')
    targets, topic_mix, counts = {}, {}, {}
    for o in OUTLETS:
        rows = [r for r in real if r['outlet'] == o]
        c = Counter(leaning[text_hash(r['text'])]['label'] for r in rows)
        counts[o] = dict(c)
        targets[o] = {l: (c[l] + 1) / (len(rows) + 3) for l in LABELS}
        t = Counter(topics[text_hash(r['text'])]['topic'] for r in rows)
        total = sum(t[x] for x in TRAIN_TOPICS)
        topic_mix[o] = {x: t[x] / total for x in TRAIN_TOPICS}
    pooled_counts = Counter()
    for o in OUTLETS:
        pooled_counts.update(counts[o])
    n = sum(pooled_counts.values())
    pooled = {l: (pooled_counts[l] + 1) / (n + 3) for l in LABELS}
    return targets, pooled, topic_mix, counts


def condition_targets(condition, targets, pooled):
    if condition == 'calibrated':
        return {o: targets[o] for o in OUTLETS}
    if condition == 'shuffled':
        return {o: targets[DERANGEMENT[o]] for o in OUTLETS}
    if condition == 'pooled':
        return {o: pooled for o in OUTLETS}
    if condition == 'format':
        return None
    raise ValueError(condition)


def update_q(q, probs, outlets, kappa):
    """Batch estimate of each outlet's label marginal, shrunk toward the previous estimate."""
    new = {}
    for o in q:
        idx = [i for i, x in enumerate(outlets) if x == o]
        batch_sum = {l: sum(probs[i][l] for i in idx) for l in LABELS}
        new[o] = {l: (batch_sum[l] + kappa * q[o][l]) / (len(idx) + kappa) for l in LABELS}
    return new


def leaning_reward(p, target, q):
    """sum_l P(l|x) (p_o(l) - q_o(l)): positive for labels the policy under-produces relative to the target."""
    return sum(p[l] * (target[l] - q[l]) for l in LABELS)


def group_advantages(rewards, group_size):
    """Group-mean-centred advantages without std division (Dr. GRPO)."""
    out = []
    for i in range(0, len(rewards), group_size):
        g = rewards[i:i + group_size]
        m = sum(g) / len(g)
        out += [r - m for r in g]
    return out


def format_fail(text, finished):
    f = flags(text)
    return (not finished) or f['empty'] or f['over_280_chars'] or f['has_link'] or f['has_hashtag'] or \
        f['possible_preface'] or f['possible_decline']


class Policy:
    def __init__(self, model_id, seed, lora=True):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model
        self.torch, self.device = torch, 'mps' if torch.backends.mps.is_available() else 'cpu'
        torch.manual_seed(seed)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
        # fp32 throughout: bf16 sampling on MPS produced NaN probabilities once the policy drifted (first launch,
        # 2026-10-16; those runs were aborted and restarted in fp32 for every condition).
        base = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32).to(self.device)
        base.generation_config.update(do_sample=True, temperature=CONFIG['temperature'], top_p=1.0, top_k=0,
                                      repetition_penalty=1.0)
        if lora:
            cfg = LoraConfig(r=CONFIG['lora_r'], lora_alpha=CONFIG['lora_alpha'], lora_dropout=0.0,
                             target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
                             task_type='CAUSAL_LM')
            self.model = get_peft_model(base, cfg)
            for p in self.model.parameters():
                if p.requires_grad:
                    p.data = p.data.float()
        else:
            self.model = base

    def prompt_ids(self, prompts):
        chats = [self.tokenizer.apply_chat_template([{'role': 'user', 'content': p}], tokenize=False,
                                                    add_generation_prompt=True) for p in prompts]
        return self.tokenizer(chats, return_tensors='pt', padding=True).to(self.device)

    def sample(self, prompts, n):
        """n completions per prompt. Returns texts, finished flags, and tensors for log-prob computation."""
        torch = self.torch
        enc = self.prompt_ids(prompts)
        self.model.eval()
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=CONFIG['max_new_tokens'], num_return_sequences=n,
                                      do_sample=True, temperature=CONFIG['temperature'], top_p=1.0, top_k=0,
                                      repetition_penalty=1.0, pad_token_id=self.tokenizer.pad_token_id)
        prompt_len = enc['input_ids'].shape[1]
        completion = out[:, prompt_len:]
        eos = self.tokenizer.eos_token_id
        stop_ids = {eos, self.tokenizer.convert_tokens_to_ids('<|im_end|>')}
        mask = torch.ones_like(completion)
        finished = []
        for i in range(completion.shape[0]):
            ids = completion[i].tolist()
            stop = next((j for j, t in enumerate(ids) if t in stop_ids), None)
            finished.append(stop is not None)
            if stop is not None:
                mask[i, stop + 1:] = 0  # keep the stop token itself, drop padding after it
        texts = [self.tokenizer.decode(completion[i][mask[i].bool()], skip_special_tokens=True).strip()
                 for i in range(completion.shape[0])]
        attn = torch.cat([enc['attention_mask'].repeat_interleave(n, 0), mask], dim=1)
        return texts, finished, {'sequences': out, 'attention': attn, 'prompt_len': prompt_len, 'mask': mask}

    def token_logprobs(self, batch, rows, use_adapter=True, grad=False):
        """Per-token log-probs of the sampled completion for the given row indices (chunked to bound memory:
        full-vocabulary logits for 80 rollouts would not fit)."""
        torch = self.torch
        seq, attn, plen = batch['sequences'][rows], batch['attention'][rows], batch['prompt_len']
        def forward():
            logits = self.model(input_ids=seq, attention_mask=attn).logits[:, plen - 1:-1].float()
            return torch.log_softmax(logits, -1).gather(-1, seq[:, plen:].unsqueeze(-1)).squeeze(-1)
        if grad:
            return forward()
        with torch.no_grad():
            if use_adapter:
                return forward()
            with self.model.disable_adapter():
                return forward()


class RewardModel:
    def __init__(self):
        from study.measures import SequenceClassifier
        self.classifier = SequenceClassifier('leaning')

    def probabilities(self, texts):
        scored = [scoring_text(t) or '.' for t in texts]
        return self.classifier.probabilities(scored, batch_size=32)


def build_prompts(rng, topic_mix):
    jobs = []
    for o in OUTLETS:
        for _ in range(CONFIG['prompts_per_step'] // len(OUTLETS)):
            topic = rng.choices(TRAIN_TOPICS, weights=[topic_mix[o][t] for t in TRAIN_TOPICS])[0]
            jobs.append({'outlet': o, 'topic': topic, 'prompt': TEMPLATE.format(voice=VOICES[o], topic=TOPIC_WORDING[topic])})
    return jobs


def train(args):
    import torch
    rng = random.Random(args.seed)
    random.seed(args.seed)
    targets, pooled, topic_mix, counts = real_targets()
    ctargets = condition_targets(args.condition, targets, pooled)
    run = f'{args.condition}-s{args.seed}' + (f'-{args.tag}' if args.tag else '')
    out = RESULTS / 'rl' / run
    out.mkdir(parents=True, exist_ok=True)
    adapters = ROOT / 'artifacts' / 'rl' / run
    cfg = {**CONFIG, 'steps': args.steps, 'beta': args.beta, 'lr': args.lr}
    write_json(out / 'manifest.json', {'run': run, 'condition': args.condition, 'seed': args.seed, 'model': args.model,
                                       'config': cfg, 'targets_used': ctargets, 'real_targets': targets, 'real_counts': counts,
                                       'pooled_target': pooled, 'topic_mix': topic_mix, 'train_topics': TRAIN_TOPICS,
                                       'held_out_topics': HELD_OUT_TOPICS, 'template': TEMPLATE, 'started_at': now(),
                                       'derangement': DERANGEMENT if args.condition == 'shuffled' else None,
                                       'note': 'Qwen chat template inserts its default system prompt; MPS is not bit-deterministic.'})
    policy, reward_model = Policy(args.model, args.seed), RewardModel()
    params = [p for p in policy.model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / CONFIG['warmup']))

    # Initialise q from base-policy samples (same for every condition, so conditions start identically).
    q = {}
    init_n = 8 if args.timing else CONFIG['init_samples_per_outlet']
    for o in OUTLETS:
        prompts = [TEMPLATE.format(voice=VOICES[o], topic=TOPIC_WORDING[rng.choices(TRAIN_TOPICS, weights=[topic_mix[o][t] for t in TRAIN_TOPICS])[0]])
                   for _ in range(init_n // 8)]
        texts, _, _ = policy.sample(prompts, 8)
        probs = reward_model.probabilities(texts)
        q[o] = {l: sum(p[l] for p in probs) / len(probs) for l in LABELS}
    write_json(out / 'q_init.json', q)

    log = open(out / 'train_log.jsonl', 'a')
    for step in range(1, args.steps + 1):
        t0 = time.time()
        jobs = build_prompts(rng, topic_mix)
        try:
            texts, finished, batch = policy.sample([j['prompt'] for j in jobs], CONFIG['group_size'])
        except RuntimeError as error:  # includes torch.AcceleratorError (NaN/inf sampling probabilities)
            write_json(out / 'failed.json', {'step': step, 'error': f'{type(error).__name__}: {str(error)[:300]}', 'at': now()})
            raise
        G = CONFIG['group_size']
        outlets = [j['outlet'] for j in jobs for _ in range(G)]
        probs = reward_model.probabilities(texts)
        mask = batch['mask'].float()
        chunks = [list(range(i, min(i + args.chunk, len(texts)))) for i in range(0, len(texts), args.chunk)]
        logp_old = torch.cat([policy.token_logprobs(batch, c, use_adapter=True) for c in chunks])
        ref = torch.cat([policy.token_logprobs(batch, c, use_adapter=False) for c in chunks])
        kl_seq = ((logp_old - ref) * mask).sum(-1)
        fails = [format_fail(t, f) for t, f in zip(texts, finished)]
        # Update q with this batch (shrunk toward the previous estimate), then reward against it.
        q = update_q(q, probs, outlets, CONFIG['kappa'])
        rewards = []
        for i, (o, p) in enumerate(zip(outlets, probs)):
            lean = 0.0 if ctargets is None else leaning_reward(p, ctargets[o], q[o])
            rewards.append(lean - float(fails[i]) - args.beta * kl_seq[i].item())
        adv = torch.tensor(group_advantages(rewards, G), device=mask.device)
        # On-policy: one update per batch, so the importance ratio is 1 and no clipping is needed.
        policy.model.train()
        opt.zero_grad()
        total_tokens, loss_value = mask.sum(), 0.0
        for c in chunks:
            logp = policy.token_logprobs(batch, c, grad=True)
            chunk_loss = -(adv[c].unsqueeze(1) * logp * mask[c]).sum() / total_tokens
            chunk_loss.backward()
            loss_value += chunk_loss.item()
        loss = torch.tensor(loss_value)
        grad_norm = torch.nn.utils.clip_grad_norm_(params, CONFIG['clip']).item()
        opt.step()
        sched.step()
        labels = [max(p, key=p.get) for p in probs]
        record = {'step': step, 'seconds': round(time.time() - t0, 2), 'loss': loss.item(), 'grad_norm': grad_norm,
                  'reward_mean': sum(rewards) / len(rewards), 'kl_seq_mean': kl_seq.mean().item(),
                  'format_fail_rate': sum(fails) / len(fails), 'mean_chars': sum(map(len, texts)) / len(texts),
                  'q': q, 'batch_label_shares': {o: {l: sum(1 for i, x in enumerate(outlets) if x == o and labels[i] == l) / outlets.count(o)
                                                     for l in LABELS} for o in OUTLETS},
                  'tv_q_to_real': {o: 0.5 * sum(abs(q[o][l] - targets[o][l]) for l in LABELS) for o in OUTLETS},
                  'example': {'outlet': outlets[0], 'text': texts[0]}}
        log.write(json.dumps(record) + '\n')
        log.flush()
        print(f"step {step} {record['seconds']}s reward {record['reward_mean']:+.3f} kl {record['kl_seq_mean']:.2f} "
              f"fail {record['format_fail_rate']:.2f} tv {sum(record['tv_q_to_real'].values())/5:.3f}", flush=True)
        if step in CONFIG['checkpoints'] or step == args.steps:
            policy.model.save_pretrained(adapters / f'step-{step}')
        # Release cached MPS memory every step; two fp32 runs in parallel exhausted unified memory (v2.6 note).
        del batch, logp_old, ref, kl_seq, adv
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    log.close()
    write_json(out / 'finished.json', {'finished_at': now(), 'steps': args.steps})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--condition', required=True, choices=['calibrated', 'format', 'shuffled', 'pooled'])
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--model', default='Qwen/Qwen2.5-1.5B-Instruct')
    parser.add_argument('--steps', type=int, default=CONFIG['steps'])
    parser.add_argument('--lr', type=float, default=CONFIG['lr'])
    parser.add_argument('--beta', type=float, default=CONFIG['beta'])
    parser.add_argument('--tag', default='')
    parser.add_argument('--chunk', type=int, default=4, help='Rollouts per forward/backward chunk (memory only)')
    parser.add_argument('--timing', action='store_true', help='Small initialisation for a feasibility/timing pilot')
    train(parser.parse_args())


if __name__ == '__main__':
    main()
