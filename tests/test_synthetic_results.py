import json
from pathlib import Path
import tempfile
import unittest
from scripts.report_synthetic_results import agreement, weighted, summary, TOPICS
from scripts.score_synthetic_dataset import checked_checkpoint


def row(topic='president', predicted='president', left=.2, center=.1, right=.7):
    p={'left':left,'center':center,'right':right}
    label=max(p,key=p.get)
    return {'topic':topic,'political_leaning':{'probabilities':p,'label':label,'confidence':p[label]},
            'topic_classification':{'topic':predicted,'confidence':.8}}


class SyntheticResultTests(unittest.TestCase):
    def test_confusion_and_f1_include_other_as_false_negative(self):
        result=agreement([row(),row(predicted='other'),row(topic='elections',predicted='president')])
        self.assertAlmostEqual(result['match_rate'],1/3)
        self.assertAlmostEqual(result['per_topic']['president']['precision'],.5)
        self.assertAlmostEqual(result['per_topic']['president']['recall'],.5)
        self.assertAlmostEqual(result['macro_f1_requested_labels'],.25)
        self.assertEqual(result['confusion_counts']['president']['other'],1)

    def test_signed_score_uses_both_probabilities(self):
        result=summary([row(left=.4,center=.1,right=.5)])
        self.assertAlmostEqual(result['mean_signed_score'],.1)
        self.assertAlmostEqual(result['mean_leaning_confidence'],.5)

    def test_topic_weighting_uses_cells_not_oversampled_rows(self):
        rows=[row(t,t,left=.1,center=.1,right=.8) for t in TOPICS]
        target={t:1 for t in TOPICS}
        original=weighted(rows,target)
        rows.extend([row('president','president',left=.1,center=.1,right=.8)]*50)
        self.assertAlmostEqual(weighted(rows,target)['mean_signed_score'],original['mean_signed_score'])
        self.assertAlmostEqual(weighted(rows,target)['match_rate'],1)
        missing=[r for r in rows if r['topic']!='elections']
        self.assertIsNone(weighted(missing,target)['mean_signed_score'])

    def test_checkpoint_rejects_duplicate_or_unknown_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'checkpoint.jsonl'
            for keys in [('a','a'),('a','b')]:
                path.write_text(''.join(json.dumps({'key':k})+'\n' for k in keys))
                with self.assertRaises(ValueError):checked_checkpoint(path,{'a'})
            path.write_text(json.dumps({'key':'a'})+'\n')
            self.assertEqual(set(checked_checkpoint(path,{'a'})),{'a'})
