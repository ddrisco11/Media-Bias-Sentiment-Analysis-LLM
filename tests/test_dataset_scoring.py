import unittest
from scripts.score_media_dataset import input_text
from scripts.media_dataset import URL_PATTERN


class DatasetTests(unittest.TestCase):
    def test_author_matching_excludes_mentions_and_affiliates(self):
        match = URL_PATTERN.match('https://twitter.com/FoxNews/status/1234567890123456789')
        self.assertEqual(match.groups(), ('FoxNews', '1234567890123456789'))
        self.assertIsNone(URL_PATTERN.match('Mentioning @FoxNews in a post'))
        match = URL_PATTERN.match('https://x.com/foxnewsradio/status/123')
        self.assertNotEqual(match[1].lower(), 'foxnews')

    def test_uses_text_then_rawcontent_without_altering_content(self):
        self.assertEqual(input_text({'id': 1, 'text': ' Text ', 'rawContent': 'Other'}), (' Text ', 'text'))
        self.assertEqual(input_text({'id': 1, 'text': None, 'rawContent': 'Fallback'}), ('Fallback', 'rawContent'))
        with self.assertRaises(ValueError): input_text({'id': 1, 'text': ' '})
