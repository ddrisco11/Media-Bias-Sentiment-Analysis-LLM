"""Unit tests for the v2 study code that do not need models or the network."""
import unittest

from study import rl
from study.common import party_mentioned, scoring_text, near_duplicate_clusters
from study.mask import mask_outlets
from study.stats import holm, random_effects, tost_equivalent, total_variation
from study.swap import swap_text
from study.data import keyword_match


class RewardTests(unittest.TestCase):
    def test_reward_is_zero_at_fixed_point_and_pushes_toward_target(self):
        target = {'left': 0.5, 'center': 0.0, 'right': 0.5}
        q_at_target = dict(target)
        self.assertAlmostEqual(rl.leaning_reward({'left': 1, 'center': 0, 'right': 0}, target, q_at_target), 0.0)
        q_too_left = {'left': 0.9, 'center': 0.0, 'right': 0.1}
        right = rl.leaning_reward({'left': 0, 'center': 0, 'right': 1}, target, q_too_left)
        left = rl.leaning_reward({'left': 1, 'center': 0, 'right': 0}, target, q_too_left)
        self.assertGreater(right, 0)
        self.assertLess(left, 0)
        self.assertLessEqual(abs(right), 1)

    def test_q_update_shrinks_toward_previous(self):
        q = {'a': {'left': 1.0, 'center': 0.0, 'right': 0.0}}
        probs = [{'left': 0.0, 'center': 0.0, 'right': 1.0}] * 16
        new = rl.update_q(q, probs, ['a'] * 16, kappa=16)
        self.assertAlmostEqual(new['a']['right'], 0.5)
        self.assertAlmostEqual(sum(new['a'].values()), 1.0)

    def test_group_advantages_are_mean_centred_without_scaling(self):
        adv = rl.group_advantages([1.0, 3.0, 0.0, 0.0], 2)
        self.assertEqual(adv, [-1.0, 1.0, 0.0, 0.0])

    def test_shuffled_targets_are_a_derangement(self):
        self.assertTrue(all(k != v for k, v in rl.DERANGEMENT.items()))
        self.assertEqual(sorted(rl.DERANGEMENT.values()), sorted(rl.DERANGEMENT))
        self.assertEqual(set(rl.TRAIN_TOPICS) & set(rl.HELD_OUT_TOPICS), set())


class TextTests(unittest.TestCase):
    def test_scoring_text_strips_links_and_wrapping_quotes(self):
        self.assertEqual(scoring_text('"Vote today" https://t.co/abc'), 'Vote today')

    def test_party_mention_ignores_place_names(self):
        self.assertEqual(party_mentioned('Police in Harris County say crime fell'), 'neither')
        self.assertEqual(party_mentioned('Harris and Trump debate'), 'both')

    def test_masking_hides_outlets_but_keeps_ordinary_words(self):
        self.assertEqual(mask_outlets('Tune in to Fox News. #FoxNews'), 'Tune in to [OUTLET]. [OUTLET]')
        self.assertIn('Fox hunting', mask_outlets('Fox hunting season opens'))

    def test_swap_exchanges_parties_once(self):
        self.assertEqual(swap_text('Trump attacks Democrats'), 'Harris attacks Republicans')
        self.assertEqual(swap_text('Biden and Trump trade barbs'), 'Trump and Biden trade barbs')
        self.assertIsNone(swap_text('Hunter Biden convicted'))
        self.assertIsNone(swap_text('Biden, Harris and Trump all spoke'))

    def test_keyword_filter_uses_and_semantics(self):
        self.assertIn('Donald Trump', keyword_match('Trump spoke; Donald later left'))
        self.assertEqual(keyword_match('Inflation cooled last month'), [])

    def test_near_duplicates_cluster_together(self):
        c = near_duplicate_clusters(['a', 'b', 'c'], ['polls are open check your registration today',
                                                      'polls are open check your registration now', 'tariffs on china rise'], 0.5)
        self.assertEqual(c['a'], c['b'])
        self.assertNotEqual(c['a'], c['c'])


class StatsTests(unittest.TestCase):
    def test_holm(self):
        self.assertEqual(list(holm([0.01, 0.04, 0.03])), [0.03, 0.06, 0.06])

    def test_tost_and_tv(self):
        self.assertTrue(tost_equivalent((-5, 5), 15))
        self.assertFalse(tost_equivalent((-20, 5), 15))
        self.assertAlmostEqual(total_variation([1, 0, 0], [0, 0, 1]), 1.0)

    def test_random_effects_pools_identical_estimates(self):
        r = random_effects([10, 10, 10], [1, 1, 1])
        self.assertAlmostEqual(r['estimate'], 10)
        self.assertAlmostEqual(r['tau2'], 0)


if __name__ == '__main__':
    unittest.main()
