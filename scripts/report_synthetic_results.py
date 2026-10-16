"""Build V1Results.md, machine-readable metrics, and plots from scored Haiku runs.

Run after scripts.score_synthetic_dataset. Matplotlib is required only for plots.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

from scripts.score_media_dataset import read_jsonl, write_json
from scripts.summarize_synthetic_tweets import declined
from topic_classification import DEFAULT_TOPICS
from topic_classification.classifier import TOPIC_POLICY_VERSION, GUN_EVIDENCE

VOICES = ['nytimes','foxnews','reuters','ap','abc','neutral']
NAMES = {'nytimes':'NYT','foxnews':'Fox News','reuters':'Reuters','ap':'AP','abc':'ABC','neutral':'Neutral voice'}
LABELS = ['left','center','right']
COLORS = ['#3569ad','#a5acb5','#d05a54']
TOPICS = [t for t in DEFAULT_TOPICS if t != 'other']


def signed(row):
    p = row['political_leaning']['probabilities']
    return p['right'] - p['left']


def agreement(rows):
    """Compare predicted topic with requested topic, not human truth."""
    matrix = {t: {p: 0 for p in DEFAULT_TOPICS} for t in TOPICS}
    for r in rows: matrix[r['topic']][r['topic_classification']['topic']] += 1
    per_topic = {}
    for topic in TOPICS:
        tp = matrix[topic][topic]
        support = sum(matrix[topic].values())
        predicted = sum(matrix[t][topic] for t in TOPICS)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_topic[topic] = dict(n=support, matched=tp, precision=precision, recall=recall, f1=f1)
    supported = [v for v in per_topic.values() if v['n']]
    return {'n': len(rows), 'match_rate': sum(matrix[t][t] for t in TOPICS) / len(rows) if rows else None,
            'macro_f1_requested_labels': statistics.mean(v['f1'] for v in supported) if supported else None,
            'per_topic': per_topic, 'confusion_counts': matrix}


def summary(rows):
    if not rows: return {'n':0}
    counts = Counter(r['political_leaning']['label'] for r in rows)
    conf = [r['political_leaning']['confidence'] for r in rows]
    return {'n':len(rows), 'leaning_counts':{l:counts[l] for l in LABELS},
            'mean_signed_score':statistics.mean(map(signed, rows)),
            'mean_leaning_confidence':statistics.mean(conf),
            'median_leaning_confidence':statistics.median(conf),
            'leaning_confidence_over_99pct':sum(c>.99 for c in conf)/len(conf),
            'mean_topic_confidence':statistics.mean(r['topic_classification']['confidence'] for r in rows),
            'topic_reranked':sum(r['topic_classification'].get('reranked',False) for r in rows),
            'agreement':agreement(rows)}


def weighted(rows, target_counts):
    """Standardize over requested topics; fail closed if a cell is missing."""
    total = sum(target_counts[t] for t in TOPICS)
    weights = {t:target_counts[t]/total for t in TOPICS}
    cells = {t:[r for r in rows if r['topic']==t] for t in TOPICS}
    coverage = sum(weights[t] for t in TOPICS if cells[t])
    if any(not cells[t] for t in TOPICS):
        return {'weight_coverage':coverage,'mean_signed_score':None,'match_rate':None}
    return {'weight_coverage':coverage,
            'mean_signed_score':sum(weights[t]*statistics.mean(map(signed,cells[t])) for t in TOPICS),
            'match_rate':sum(weights[t]*statistics.mean(r['topic_classification']['topic']==t for r in cells[t]) for t in TOPICS),
            'leaning_shares':{l:sum(weights[t]*statistics.mean(r['political_leaning']['label']==l for r in cells[t]) for t in TOPICS) for l in LABELS}}


def validate(root, rows):
    manifest = json.loads((root/'run_manifest.json').read_text())
    def require(condition, message):
        if not condition: raise ValueError(message)
    require(manifest['topic_policy_version']==TOPIC_POLICY_VERSION and manifest['topics']==DEFAULT_TOPICS,'Stale topic policy')
    for filename, checksum in manifest['classifier_code_sha256'].items():
        require(hashlib.sha256(Path(filename).read_bytes()).hexdigest()==checksum, 'Scoring code changed since inference')
    sources = {}
    for filename, info in manifest['sources'].items():
        path=Path(filename)
        require(hashlib.sha256(path.read_bytes()).hexdigest()==info['sha256'],'Source hash changed')
        source_rows=read_jsonl(path)
        require(len(source_rows)==info['rows'],'Source count differs')
        for r in source_rows: sources[f'{path.parent.name}/{r["id"]}']=r
    require(len(rows)==len(sources)==len({r['key'] for r in rows}),'Coverage/duplicate error')
    require({r['key'] for r in rows}==set(sources),'Source IDs differ')
    for stage in ('political','topic'):
        checkpoint=read_jsonl(root/f'{stage}_scores.jsonl')
        require(len(checkpoint)==len(rows) and len({r['key'] for r in checkpoint})==len(rows),'Invalid checkpoint coverage')
        by_key={r['key']:r for r in checkpoint}
        field='political_leaning' if stage=='political' else 'topic_classification'
        require(all(r[field]==by_key[r['key']] for r in rows),'Combined output differs from checkpoint')
    for r in rows:
        source=sources[r['key']]
        require(all(r[k]==source[k] for k in ('id','text','voice','arm','topic','flags','stop_reason')),'Original metadata or text changed')
        p=r['political_leaning']; probs=p['probabilities']; topic=r['topic_classification']
        require(set(probs)==set(LABELS) and all(math.isfinite(v) and 0<=v<=1 for v in probs.values()) and abs(sum(probs.values())-1)<1e-5,'Bad leaning probabilities')
        require(p['model'] in manifest['models'] and topic['model'] in manifest['models'], 'Unexpected model identifier')
        require(p['label']==max(probs,key=probs.get) and p['confidence']==probs[p['label']],'Bad leaning label/confidence')
        require(topic['topic'] in DEFAULT_TOPICS and math.isfinite(topic['confidence']) and 0<=topic['confidence']<=1,'Bad topic result')
        require(topic['topic']!='gun issues' or GUN_EVIDENCE.search(r['text']),'Unsupported gun prediction')
        require(topic['input_characters']==len(r['text']) and topic['truncated']==(len(r['text'])>1000),'Topic truncation metadata mismatch')
        require(p['truncated']==(p['input_tokens']>512),'Political truncation metadata mismatch')
    return {'status':'passed','scored_responses':len(rows),'source_text_preserved':True,
            'political_truncated':sum(r['political_leaning']['truncated'] for r in rows),
            'topic_truncated':sum(r['topic_classification']['truncated'] for r in rows),
            'sources':manifest['sources']}


def plots(root, main, pilot, real, metrics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
    dest=root/'figures';dest.mkdir(exist_ok=True)
    def save(fig,name):
        fig.savefig(dest/name,dpi=180,bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4.8),layout='constrained')
    left=np.zeros(len(VOICES))
    for label,color in zip(LABELS,COLORS):
        values=np.array([metrics['main_by_voice'][v]['leaning_counts'][label]/metrics['main_by_voice'][v]['n']*100 for v in VOICES])
        ax.barh([NAMES[v] for v in VOICES],values,left=left,label=label.title(),color=color)
        for i,val in enumerate(values):
            if val>=5: ax.text(left[i]+val/2,i,f'{val:.1f}%',ha='center',va='center',color='white' if label!='center' else '#17202a',fontsize=9)
        left+=values
    ax.invert_yaxis();ax.set(xlim=(0,100),xlabel='Share of predicted labels (%)',title='Haiku main run: political-leaning predictions by requested voice')
    ax.legend(ncol=3,loc='upper center',bbox_to_anchor=(.5,-.13));save(fig,'main_leaning.png')
    matrix=metrics['main']['agreement']['confusion_counts']; predicted=list(DEFAULT_TOPICS)
    arr=np.array([[matrix[t][p] for p in predicted] for t in TOPICS])
    frac=arr/arr.sum(axis=1,keepdims=True)*100
    fig,ax=plt.subplots(figsize=(12,9),layout='constrained');im=ax.imshow(frac,cmap='Blues',vmin=0,vmax=100)
    ax.set_xticks(range(len(predicted)),predicted,rotation=50,ha='right')
    ax.set_yticks(range(len(TOPICS)),[f'{t} (n={sum(matrix[t].values())})' for t in TOPICS])
    for i in range(len(TOPICS)):
        for j in range(len(predicted)):
            if arr[i,j]: ax.text(j,i,f'{frac[i,j]:.0f}',ha='center',va='center',fontsize=8,color='white' if frac[i,j]>55 else '#17202a')
    ax.set(xlabel='Predicted topic',ylabel='Requested generation topic',title='Main run: topic prompt agreement\nRow percentages; requested topics are not human-verified truth')
    fig.colorbar(im,ax=ax,shrink=.7,label='Row share (%)');save(fig,'topic_confusion.png')
    fig,ax=plt.subplots(figsize=(10,6),layout='constrained')
    vals=metrics['main']['agreement']['per_topic']; order=sorted(TOPICS,key=lambda t:vals[t]['recall'])
    bars=ax.barh(order,[100*vals[t]['recall'] for t in order],color='#247f87')
    for bar,t in zip(bars,order):
        ax.text(min(bar.get_width()+1,96),bar.get_y()+bar.get_height()/2,f"{100*vals[t]['recall']:.1f}%  (n={vals[t]['n']})",va='center',ha='left' if bar.get_width()<85 else 'right',fontsize=9)
    ax.set(xlim=(0,115),xlabel='Predicted topic equals requested topic (%)',title='Topic agreement varies by requested subject');save(fig,'topic_agreement.png')
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained'); y=np.arange(len(VOICES))
    ax.scatter([100*metrics['main_by_voice'][v]['mean_signed_score'] for v in VOICES],y-.12,label='Synthetic: unweighted',marker='o',color='#247f87',s=55)
    ax.scatter([100*metrics['main_weighted_by_voice'][v]['mean_signed_score'] for v in VOICES],y,label='Synthetic: pooled real-topic weights',marker='D',color='#a76ac4',s=45)
    ax.scatter([100*metrics['real_by_outlet'][v]['mean_signed_score'] for v in VOICES[:-1]],y[:-1]+.14,label='Real tweets: unweighted context',marker='x',color='#303b49',s=65)
    ax.axvline(0,color='#b0b6bd',lw=1);ax.set_yticks(y,[NAMES[v] for v in VOICES]);ax.invert_yaxis()
    ax.set(xlim=(-105,105),xlabel='100 × mean[P(right) − P(left)]  |  left ← 0 → right',title='Descriptive model scores, not validated outlet bias ratings')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.18),ncol=1);save(fig,'leaning_comparison.png')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    x=np.arange(len(VOICES));width=.36
    for idx,arm in enumerate(['thinking-off','thinking-on']):
        data=[metrics['pilot_by_arm_voice'][f'{arm}/{v}'] for v in VOICES]
        axes[0].bar(x+(idx-.5)*width,[100*d['written']['agreement']['match_rate'] for d in data],width,label=arm,color=['#247f87','#a76ac4'][idx])
        axes[1].bar(x+(idx-.5)*width,[100*d['declined']/d['all']['n'] for d in data],width,label=arm,color=['#247f87','#a76ac4'][idx])
    for ax in axes:
        ax.set_xticks(x,[NAMES[v] for v in VOICES],rotation=30,ha='right');ax.set_ylim(0,105);ax.legend()
    axes[0].set(title='Pilot: topic agreement, declines excluded',ylabel='Prompt agreement (%)')
    axes[1].set(title='Pilot: flagged decline rate',ylabel='Responses flagged as declines (%)')
    save(fig,'pilot_comparison.png')
    fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained')
    for rows,label,color in [(main,'Haiku main','#247f87'),(real,'Real news tweets','#a76ac4')]:
        values=[r['political_leaning']['confidence'] for r in rows]
        ax.hist(values,bins=np.linspace(1/3,1,15),weights=np.ones(len(values))*100/len(values),histtype='step',lw=2,label=label,color=color)
    ax.set(xlabel='Winning political-leaning probability',ylabel='Share of dataset (%)',title='Confidence distribution — confidence is not measured correctness');ax.legend();save(fig,'confidence.png')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('reports/v1'))
    args=parser.parse_args();root=args.root
    rows=read_jsonl(root/'scored_generations.jsonl')
    validation=validate(root,rows);write_json(root/'validation.json',validation)
    main_rows=[r for r in rows if r['run']=='main-v1'];pilot=[r for r in rows if r['run']=='pilot-v1']
    real=read_jsonl(Path('dataset/annotated_tweets.jsonl'))
    gen_manifest=json.loads(Path('dataset/synthetic/main-v1/run_manifest.json').read_text())
    target=gen_manifest['topic_mix']['source_counts']
    target_hash=hashlib.sha256(Path(gen_manifest['topic_mix']['source']).read_bytes()).hexdigest()
    if target_hash!=gen_manifest['topic_mix']['source_sha256']: raise ValueError('Real topic weighting source changed')
    clean=[r for r in main_rows if not any(r['flags'].values()) and not declined(r)]
    seen=set();unique=[]
    for r in main_rows:
        key=(r['voice'],r['text'].strip().lower())
        if key not in seen: unique.append(r);seen.add(key)
    metrics={'main':summary(main_rows),'main_written':summary([r for r in main_rows if not declined(r)]),
             'main_strict_clean':summary(clean),'main_deduplicated_within_voice':summary(unique),
             'main_by_voice':{},'main_weighted_by_voice':{},'pilot_by_arm_voice':{},'real_by_outlet':{},
             'data_quality':{},'pilot_by_arm':{},'weighting_target':target}
    for v in VOICES:
        group=[r for r in main_rows if r['voice']==v]
        metrics['main_by_voice'][v]=summary(group);metrics['main_weighted_by_voice'][v]=weighted(group,target)
        if v!='neutral':
            q=[r for r in real if r['analysis_account']==v];c=Counter(r['political_leaning']['label'] for r in q)
            metrics['real_by_outlet'][v]={'n':len(q),'mean_signed_score':statistics.mean(map(signed,q)),'leaning_counts':dict(c)}
        for arm in ['thinking-off','thinking-on']:
            group=[r for r in pilot if r['voice']==v and r['arm']==arm]
            metrics['pilot_by_arm_voice'][f'{arm}/{v}']={'all':summary(group),'written':summary([r for r in group if not declined(r)]),'declined':sum(map(declined,group))}
    for arm in ['thinking-off','thinking-on']:
        group=[r for r in pilot if r['arm']==arm]
        metrics['pilot_by_arm'][arm]={'all':summary(group),'written':summary([r for r in group if not declined(r)]),'declined':sum(map(declined,group))}
    for run,group in [('main-v1',main_rows),('pilot-v1',pilot)]:
        metrics['data_quality'][run]={'n':len(group),'declines':sum(map(declined,group)),
            'flags':dict(Counter(k for r in group for k,v in r['flags'].items() if v)),
            'exact_text_duplicates':len(group)-len({r['text'] for r in group}),
            'max_characters':max(len(r['text']) for r in group)}
    metrics['notes']=['Topic reference is the requested generation topic, not human truth.',
        'No leaning reference labels: leaning accuracy, F1 and calibration error are not estimable.',
        'Macro F1 covers the 14 requested labels; predictions of other count as false negatives.',
        'Signed score is mean P(right) minus P(left), not confidence or bias intensity.',
        'Main weighting uses original pooled real predicted topic counts, excluding other, over requested synthetic topics.',
        'No significance claims or independent-sample confidence intervals: templates and duplicates create dependence.']
    write_json(root/'metrics.json',metrics)
    import importlib.metadata
    import platform
    write_json(root/'report_manifest.json', {
        'inputs_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [root/'scored_generations.jsonl', Path('dataset/annotated_tweets.jsonl'), Path('dataset/synthetic/main-v1/run_manifest.json')]},
        'report_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'python':platform.python_version(),
        'packages':{p:importlib.metadata.version(p) for p in ('matplotlib','numpy')},
    })
    plots(root,main_rows,pilot,real,metrics)
    pct=lambda x:f'{100*x:.1f}%'
    score=lambda x:f'{100*x:+.1f}'
    lines=['# V1 Results: Haiku Synthetic News Classification','',
        'Analysis date: October 15, 2026 (America/New_York). Generation manifests use October 16 UTC.', '',
        '**These are invented Haiku responses in requested outlet voices, not actual posts by those organizations.**', '',
        f"Both existing classifiers scored **{len(rows):,} responses**: {len(main_rows):,} main-run responses and {len(pilot):,} pilot responses. Main-run topic agreement with the requested topic is **{pct(metrics['main']['agreement']['match_rate'])}**; macro F1 against those requested labels is **{metrics['main']['agreement']['macro_f1_requested_labels']:.3f}**. These measure prompt agreement, not independently verified classification accuracy.", '',
        f"The main-run leaning labels are {pct(metrics['main']['leaning_counts']['left']/len(main_rows))} left, {pct(metrics['main']['leaning_counts']['center']/len(main_rows))} center, and {pct(metrics['main']['leaning_counts']['right']/len(main_rows))} right. For context, the real sample receives {pct(sum(r['political_leaning']['label']=='center' for r in real)/len(real))} center labels. This is a distribution shift, not evidence that synthetic writing is objectively more neutral.", '',
        '**Political-leaning accuracy cannot be measured:** no human leaning labels exist for these examples. The left/center/right outputs below describe classifier behavior, not verified ideological bias or positive/negative sentiment.', '',
        '## Scope and methods','',
        '- Main: 525 responses per voice, six voices, thinking off. Pilot: 70 per voice per arm, thinking on/off, 840 total. Pilot and main are reported separately because their requested-topic allocations differ.',
        '- Inputs are the unedited `text` fields. Generation prompts, requested voice, requested topic, and thinking summaries are not passed to either classifier. Original generation records are unchanged.',
        '- Political leaning: `matous-volf/political-leaning-deberta-large`; topics: `fastino/gliner2-base-v1`, with the existing compact taxonomy and gun-evidence reranking guard. Model revisions, code hashes, source hashes, and environment are in [the scoring manifest](reports/v1/run_manifest.json).',
        f"- All responses were scored, including declines and format violations. Political truncations: {validation['political_truncated']}; topic truncations: {validation['topic_truncated']}. Flagged declines are excluded from the pilot's written-response comparisons, with all-response metrics also retained.",
        '- Topic exact match compares the predicted topic to the requested topic. Precision, recall, F1 and confusion counts use that same proxy reference. A mismatch may reflect generator topic drift, classifier error, or a valid overlapping topic.',
        '- The signed score is `100 × mean[P(right) − P(left)]`, ranging from −100 to +100. It is not confidence, bias intensity, or an outlet rating.',
        '- Weighted synthetic results average each requested-topic cell using the main-generation manifest’s pooled real-topic frequencies, excluding “other.” This corrects the minimum-10-per-topic oversampling. The weighting source hash was checked. It does not correct differences in genre, events, language, or framing.', '',
        '## Main-run topic results','',
        '![Main topic prompt agreement](reports/v1/figures/topic_agreement.png)','',
        '| Requested topic | N | Precision* | Recall / match* | F1* |','|---|---:|---:|---:|---:|']
    for t,d in metrics['main']['agreement']['per_topic'].items():
        lines.append(f"| {t} | {d['n']} | {pct(d['precision'])} | {pct(d['recall'])} | {d['f1']:.3f} |")
    lines+=['','*All metrics use the generation request as the reference; no human topic labels are available. Macro F1 averages the 14 requested categories. “Other” predictions count as errors against the requested category, but “other” is not included as a fifteenth reference class.','',
        '![Requested versus predicted topics](reports/v1/figures/topic_confusion.png)','',
        '## Main-run political-leaning outputs','',
        '![Predicted political-leaning shares](reports/v1/figures/main_leaning.png)','',
        '| Requested voice | N | Left | Center | Right | Topic match | Signed score | Weighted signed score |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for v in VOICES:
        d=metrics['main_by_voice'][v];c=d['leaning_counts'];w=metrics['main_weighted_by_voice'][v]
        lines.append(f"| {NAMES[v]} | {d['n']} | {pct(c['left']/d['n'])} | {pct(c['center']/d['n'])} | {pct(c['right']/d['n'])} | {pct(d['agreement']['match_rate'])} | {score(d['mean_signed_score'])} | {score(w['mean_signed_score'])} |")
    lines+=['','![Synthetic and real score comparison](reports/v1/figures/leaning_comparison.png)','',
        'Real-tweet points are unweighted descriptive context from the existing 500-per-outlet dataset. Synthetic weighted points use a common pooled requested-topic mix; real points retain each outlet’s observed mix. This is not a controlled estimate of an outlet-voice effect. The neutral voice is a generation instruction, not a ground-truth center label.','',
        '![Confidence distributions](reports/v1/figures/confidence.png)','',
        f"Main mean winning leaning confidence is {pct(metrics['main']['mean_leaning_confidence'])}; {pct(metrics['main']['leaning_confidence_over_99pct'])} of predictions exceed 99%. These probabilities have not been calibrated against human labels here; high confidence does not establish correctness.", '',
        '## Pilot: thinking on versus off','',
        '![Pilot topic agreement and declines](reports/v1/figures/pilot_comparison.png)','',
        '| Arm | All responses | Flagged declines | Written responses | Topic match, all | Topic match, written | Signed score, written |','|---|---:|---:|---:|---:|---:|---:|']
    for arm,d in metrics['pilot_by_arm'].items():
        lines.append(f"| {arm} | {d['all']['n']} | {d['declined']} | {d['written']['n']} | {pct(d['all']['agreement']['match_rate'])} | {pct(d['written']['agreement']['match_rate'])} | {score(d['written']['mean_signed_score'])} |")
    lines+=['','Declines are detected using the existing API stop reason and possible-decline flag. Filtering them changes the topic/voice composition, so a written-only difference is descriptive rather than a clean causal thinking effect. Per-arm/per-voice metrics are in `reports/v1/metrics.json`.','',
        '## Data quality and sensitivity','',
        '| Main-run inclusion rule | N | Topic match | Macro F1* | Signed score |','|---|---:|---:|---:|---:|']
    for name,key in [('All responses','main'),('No format/decline flags','main_strict_clean'),('Deduplicated within voice','main_deduplicated_within_voice')]:
        d=metrics[key];lines.append(f"| {name} | {d['n']} | {pct(d['agreement']['match_rate'])} | {d['agreement']['macro_f1_requested_labels']:.3f} | {score(d['mean_signed_score'])} |")
    lines+=['','Strict-clean excludes any stored flag, including overlength, links, hashtags, quotation wrapping and prefaces. Deduplication uses trimmed, lowercased text within each voice and retains the first record. It does not detect paraphrases; filtered comparisons are unweighted and their topic mix can change.','',
        '| Run | Responses | Declines | Over 280 chars | Hashtag | Link | Preface | Quote-wrapped | Exact duplicate texts* |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for run,d in metrics['data_quality'].items():
        f=d['flags'];lines.append(f"| {run} | {d['n']} | {d['declines']} | {f.get('over_280_chars',0)} | {f.get('has_hashtag',0)} | {f.get('has_link',0)} | {f.get('possible_preface',0)} | {f.get('quoted',0)} | {d['exact_text_duplicates']} |")
    lines+=['','*Exact duplicate count is excess rows with byte-identical text across the whole run, including across voices; this differs from the within-voice normalized-text sensitivity analysis above. Flags can overlap.','',
        '## Diagnostic mismatches','',
        'Examples below are selected for high topic confidence from distinct requested→predicted pairs; they are diagnostic, not a representative error-rate sample. Text is quoted as generated (line breaks escaped for display).','',
        '| ID | Requested | Predicted | Topic confidence | Generated text |','|---|---|---|---:|---|']
    seen_pairs=set()
    for r in sorted(main_rows,key=lambda r:r['topic_classification']['confidence'],reverse=True):
        pred=r['topic_classification'];pair=(r['topic'],pred['topic'])
        if pair[0]==pair[1] or pair in seen_pairs: continue
        seen_pairs.add(pair);text=r['text'].replace('|','\\|').replace('\n','<br>')
        lines.append(f"| {r['id']} | {r['topic']} | {pred['topic']} | {pct(pred['confidence'])} | {text} |")
        if len(seen_pairs)==6:break
    lines+=['','## Interpretation and limitations','',
        '- This is an inference-only baseline. No model was trained or fine-tuned on these synthetic responses.',
        '- Requested topics are known, but the actual generated text may discuss several subjects. Human review is needed to separate prompt adherence from classifier quality.',
        '- Requested outlet voices do not constitute leaning labels. The generator can reproduce stereotypes or converge on generic prose; the classifier can amplify those artifacts.',
        '- The leaning classifier previously assigned near-certain left/right labels to routine factual real tweets. Its confidence is not evidence of ideological certainty.',
        '- Synthetic content differs from the real election-keyword dataset in topics, candidates mentioned, writing style, links and event specificity. Real-versus-synthetic differences are not causal or validated ratings of the organizations.',
        '- The topic weights come from this same topic classifier on real tweets, so weighting inherits its errors. The gun-evidence guard can miss implicit references and makes reranked confidence conditional on a reduced candidate set.',
        '- Repeated templates and duplicates reduce independence. No statistical significance claims or population confidence intervals are made.',
        '- Before training, establish a human-reviewed held-out set of real tweets. Do not use these model predictions as unquestioned ground truth.', '',
        '## Artifacts and reproduction','',
        '- [Combined scored responses](reports/v1/scored_generations.jsonl): original generated text and metadata plus both classifier outputs.',
        '- [Political scores](reports/v1/political_scores.jsonl) and [topic scores](reports/v1/topic_scores.jsonl): resumable ID-keyed checkpoints.',
        '- [Metrics](reports/v1/metrics.json), [validation](reports/v1/validation.json), and [run manifest](reports/v1/run_manifest.json).',
        '- Source data: [main run](dataset/synthetic/main-v1/generations.jsonl), [pilot](dataset/synthetic/pilot-v1/generations.jsonl).', '',
        '```sh','python -m pip install -r requirements-report.txt',
        'python -m scripts.score_synthetic_dataset --device mps',
        'python -m scripts.report_synthetic_results','```','',
        'Use `--device cpu` or an available CUDA device in a fresh output directory for a different environment. The scoring manifest prevents incompatible checkpoint reuse. Plotting is deterministic for fixed result files.','']
    Path('V1Results.md').write_text('\n'.join(lines))
    print(json.dumps({'main_n':len(main_rows),'pilot_n':len(pilot),'topic_agreement':metrics['main']['agreement']['match_rate'],'validation':validation['status']},indent=2))


if __name__=='__main__':main()
