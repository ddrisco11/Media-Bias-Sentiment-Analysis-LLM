"""Paper figures from results/v2/analysis/results.json (and rl_results.json when present).

Design rules (validated light palette; one y-scale per panel, so measures on different scales get separate panels;
fixed series colours with marker shapes as secondary encoding; legends always shown; recessive grid).
Usage: python -m study.figures   (writes docs/paper/figures/*.png)
"""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from study.common import OUTLETS, OUTLET_NAMES, RESULTS, ROOT

OUT = ROOT / 'docs' / 'paper' / 'figures'
SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
S1, S2, S3 = '#2a78d6', '#eb6834', '#1baf7a'   # validated categorical slots 1-3 (light)
SHORT = {'nytimes': 'NYT', 'foxnews': 'Fox', 'reuters': 'Reuters', 'ap': 'AP', 'abc': 'ABC'}
MEASURE_LABEL = {'m1_net': 'Classifier (M1) net score\n100 x [P(right) - P(left)]',
                 'slant': 'Blind judge (M2) slant\n-3 favours Democrats ... +3 favours Republicans'}

plt.rcParams.update({'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK, 'font.size': 9, 'axes.grid': True,
                     'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True, 'axes.spines.top': False,
                     'axes.spines.right': False, 'legend.frameon': False, 'patch.edgecolor': SURFACE, 'patch.linewidth': 1.0, 'savefig.dpi': 200, 'savefig.bbox': 'tight'})


def ci_points(ax, ys, est, lo, hi, color, marker, label, offset=0.0):
    ys = np.asarray(ys) + offset
    ax.hlines(ys, lo, hi, color=color, linewidth=2)
    ax.plot(est, ys, marker, color=color, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.2, label=label, linestyle='none')


def zero_line(ax):
    ax.axvline(0, color=INK2, linewidth=0.8)


def fig_voice_minus_neutral(r):
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2))
    ys = np.arange(len(OUTLETS))[::-1]
    for ax, m in zip(axes, ['m1_net', 'slant']):
        d = r['rq1_main']['minus_neutral']
        ci_points(ax, ys, [d[o][m]['estimate'] for o in OUTLETS], [d[o][m]['ci95'][0] for o in OUTLETS],
                  [d[o][m]['ci95'][1] for o in OUTLETS], S1, 'o', 'main-v1 (Haiku, prompt v1)', 0.12)
        p = r['h1']['pooled']
        ci_points(ax, ys, [p[f'{o}:{m}']['estimate'] for o in OUTLETS], [p[f'{o}:{m}']['ci95'][0] for o in OUTLETS],
                  [p[f'{o}:{m}']['ci95'][1] for o in OUTLETS], S2, 'D', 'pooled over 5 pre-named variants', -0.12)
        zero_line(ax)
        ax.set_yticks(ys, [OUTLET_NAMES[o] for o in OUTLETS])
        ax.set_xlabel(MEASURE_LABEL[m] + '\n(voice minus neutral voice)')
    axes[1].set_yticklabels([])
    axes[0].legend(loc='lower left', bbox_to_anchor=(0, 1.02), ncol=2)
    fig.savefig(OUT / 'fig1_voice_minus_neutral.png')
    plt.close(fig)


def fig_real_vs_generated(r):
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2))
    ys = np.arange(len(OUTLETS))[::-1]
    for ax, m in zip(axes, ['m1_net', 'slant']):
        rq = r['rq2_all']
        real = [rq[o]['real_level'][m] for o in OUTLETS]
        gen = [rq[o]['generated_level'][m] for o in OUTLETS]
        ev = [r['rq2_events'][o]['generated_level'][m] for o in OUTLETS]
        ci_points(ax, ys, [x['estimate'] for x in real], [x['ci95'][0] for x in real], [x['ci95'][1] for x in real], S1, 'o', 'real tweets (cleaned)', 0.15)
        ci_points(ax, ys, [x['estimate'] for x in gen], [x['ci95'][0] for x in gen], [x['ci95'][1] for x in gen], S2, 'D', 'Haiku voice, topic only (main-v1)', 0.0)
        ax.plot(ev, ys - 0.15, 's', color=S3, markersize=6, markeredgecolor=SURFACE, label='Haiku voice, same events (event-v1)', linestyle='none')
        zero_line(ax)
        ax.set_yticks(ys, [OUTLET_NAMES[o] for o in OUTLETS])
        ax.set_xlabel(MEASURE_LABEL[m])
    axes[1].set_yticklabels([])
    axes[0].legend(loc='lower left', bbox_to_anchor=(0, 1.02), ncol=3)
    fig.savefig(OUT / 'fig2_real_vs_generated.png')
    plt.close(fig)


def fig_swap(r):
    s = r['swap']
    groups = [('neutral_templates', 'Neutral\ntemplates'), ('neutral_tweets_real', 'Neutral\nreal'),
              ('neutral_tweets_generated', 'Neutral\ngenerated'), ('partisan_all', 'Partisan\n(should flip)')]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.0), gridspec_kw={'wspace': 0.45})
    x = np.arange(len(groups))
    axes[0].bar(x, [s[g]['m1_mean_abs_delta'] for g, _ in groups], color=S1, width=0.55)
    axes[0].set_ylabel('Mean |change| after swapping\nparty names (M1 points)')
    axes[1].bar(x, [s[g]['judge_mean_abs_delta'] for g, _ in groups], color=S2, width=0.55)
    axes[1].set_ylabel('Mean |change| after swapping\nparty names (judge slant)')
    for ax, key in zip(axes, ['m1_mean_abs_delta', 'judge_mean_abs_delta']):
        ax.set_xticks(x, [lab for _, lab in groups])
        for i, (g, _) in enumerate(groups):
            ax.annotate(f"{s[g][key]:.2f}" if key == 'judge_mean_abs_delta' else f"{s[g][key]:.0f}", (i, s[g][key]),
                        ha='center', va='bottom', fontsize=8, color=INK2, xytext=(0, 2), textcoords='offset points')
        ax.grid(axis='x', visible=False)
    axes[0].set_title('Classifier (M1)', loc='left', fontsize=9)
    axes[1].set_title('Blind judge (M2)', loc='left', fontsize=9)
    fig.savefig(OUT / 'fig3_party_swap.png')
    plt.close(fig)


def fig_link():
    d = json.loads((RESULTS / 'validity' / 'link_artifact.json').read_text())
    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    x = np.arange(2)
    a = [100 * d['real_without_link']['share_center'], 100 * d['generated_as_written']['share_center']]
    b = [100 * d['real_with_link']['share_center'], 100 * d['generated_with_link']['share_center']]
    ax.bar(x - 0.15, a, 0.3, color=S1, label='no link')
    ax.bar(x + 0.15, b, 0.3, color=S2, label='trailing t.co link')
    for xs, vals in [(x - 0.15, a), (x + 0.15, b)]:
        for xi, v in zip(xs, vals):
            ax.annotate(f'{v:.0f}%', (xi, v), ha='center', va='bottom', fontsize=8, color=INK2, xytext=(0, 2), textcoords='offset points')
    ax.set_xticks(x, ['Real tweets', 'Generated tweets'])
    ax.set_ylabel('Share labelled "center" by M1 (%)')
    ax.legend(loc='upper right')
    ax.grid(axis='x', visible=False)
    fig.savefig(OUT / 'fig4_link_artifact.png')
    plt.close(fig)


def fig_spread_identification(r):
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2))
    x = np.arange(len(OUTLETS))
    c = r['caricature']['slant']
    axes[0].bar(x - 0.15, [c[o]['sd_real'] for o in OUTLETS], 0.3, color=S1, label='real tweets')
    axes[0].bar(x + 0.15, [c[o]['sd_generated'] for o in OUTLETS], 0.3, color=S2, label='Haiku voice (main-v1)')
    axes[0].set_ylabel('Within-outlet SD of judge slant')
    ident = r['identification']
    for i, (k, col, lab) in enumerate([('real', S1, 'real tweets'), ('main-v1', S2, 'Haiku, topic only'), ('event-v1', S3, 'Haiku, same events')]):
        axes[1].bar(x + (i - 1) * 0.25, [100 * ident[k]['per_outlet_accuracy'][o] for o in OUTLETS], 0.25, color=col, label=lab)
    axes[1].axhline(20, color=INK2, linewidth=0.8, linestyle='--')
    axes[1].annotate('chance 20%', (-0.55, 21), ha='left', va='bottom', fontsize=7.5, color=INK2)
    axes[1].set_ylabel('Judge picks the right outlet (%)')
    for ax, ncol in zip(axes, (1, 1)):
        ax.set_xticks(x, [SHORT[o] for o in OUTLETS])
        ax.grid(axis='x', visible=False)
        ax.legend(loc='lower left', bbox_to_anchor=(0, 1.02), ncol=ncol, fontsize=8)
    fig.savefig(OUT / 'fig5_spread_identification.png')
    plt.close(fig)


def fig_variants(r):
    variants = ['sens-ref', 'sens-t1', 'sens-t2', 'sens-t3', 'sens-month', 'model-sonnet', 'model-opus']
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6))
    for ax, m in zip(axes, ['m1_net', 'slant']):
        for k, o in enumerate(OUTLETS):
            vals = [r['rq4_variants'][v]['minus_neutral'][o][m] for v in variants]
            ys = len(OUTLETS) - 1 - k + np.linspace(-0.3, 0.3, len(variants))
            ax.hlines(ys, [x['ci95'][0] for x in vals], [x['ci95'][1] for x in vals], color=S1, linewidth=1.2, alpha=0.8)
            ax.plot([x['estimate'] for x in vals], ys, 'o', color=S1, markersize=4, linestyle='none',
                    label='one variant (95% CI)' if k == 0 else None)
        zero_line(ax)
        ax.set_yticks(range(len(OUTLETS))[::-1], [OUTLET_NAMES[o] for o in OUTLETS])
        ax.set_xlabel(MEASURE_LABEL[m] + '\n(voice minus neutral, each variant)')
    axes[1].set_yticklabels([])
    axes[0].legend(loc='lower left', bbox_to_anchor=(0, 1.02))
    fig.text(0.5, -0.14, 'Variants top to bottom within each outlet: ' + ', '.join(variants), ha='center', fontsize=7.5, color=INK2)
    fig.savefig(OUT / 'fig6_variants.png')
    plt.close(fig)


def fig_rl():
    path = RESULTS / 'analysis' / 'rl_results.json'
    if not path.exists():
        return
    r = json.loads(path.read_text())
    conds = {'calibrated': (S1, 'o'), 'format': (S2, 'D'), 'shuffled': (S3, 's'), 'pooled': (INK2, '^')}
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0), sharey=True)
    for ax, (key, title) in zip(axes, [('tv_m1', 'Reward classifier (M1)'), ('tv_m3', 'Held-out classifier (M3)'), ('tv_judge', 'Blind judge (M2)')]):
        for run, steps in r['evals'].items():
            if run == 'base':
                continue
            cond = run.rsplit('-s', 1)[0]
            if cond not in conds:
                continue
            pts = sorted((int(s), v['train'][key]) for s, v in steps.items() if 'train' in v and v['train'].get(key) is not None)
            base = r['evals'].get('base', {}).get('0', {}).get('train', {}).get(key)
            if base is not None:
                pts = [(0, base)] + pts
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=conds[cond][1], color=conds[cond][0], linewidth=1.5,
                        markersize=4, label=cond if run.endswith('-s1') else None)
        ax.set_title(title, loc='left', fontsize=9)
        ax.set_xlabel('RL step (training topics)')
    axes[0].set_ylabel('Distance to real outlet\n(total variation, mean over outlets)')
    axes[0].legend(loc='lower left', bbox_to_anchor=(0, 1.08), ncol=4)
    fig.savefig(OUT / 'fig7_rl_curves.png')
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    r = json.loads((RESULTS / 'analysis' / 'results.json').read_text())
    fig_voice_minus_neutral(r)
    fig_real_vs_generated(r)
    fig_swap(r)
    fig_link()
    fig_spread_identification(r)
    fig_variants(r)
    fig_rl()
    print('figures written to', OUT)


if __name__ == '__main__':
    main()
