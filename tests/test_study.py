"""Unit tests for the v2 study code that do not need models or the network."""
import unittest

from study.common import party_mentioned, scoring_text, near_duplicate_clusters
from study.mask import mask_outlets
from study.stats import holm, random_effects, tost_equivalent, total_variation
from study.swap import swap_text
from study.data import keyword_match


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
