import unittest
from unittest.mock import Mock
from topic_classification import TopicClassifier
from topic_classification.classifier import TopicResult


class TopicGuardTests(unittest.TestCase):
    def test_unsupported_gun_label_is_reranked_by_model(self):
        classifier = TopicClassifier()
        backend = Mock()
        classifier._model = backend
        backend.batch_extract.return_value = [{'topic': {'label': 'president', 'confidence': 0.8}}]
        result = classifier._resolve_gun_evidence(
            ['A candidate discussed rally attendance.'], [TopicResult('gun issues', 0.9)], 8)
        self.assertEqual(result[0].topic, 'president')
        self.assertEqual(result[0].confidence, 0.8)
        self.assertTrue(result[0].reranked)
        self.assertNotIn('gun issues', backend.create_schema.return_value.classification.call_args.args[1])

    def test_explicit_gun_content_and_custom_taxonomy_are_preserved(self):
        for classifier, text in [(TopicClassifier(), 'Congress debates firearms laws.'),
                                 (TopicClassifier({'gun issues': 'Custom', 'other': 'Other'}), 'Metaphor')]:
            classifier._model = Mock()
            result = classifier._resolve_gun_evidence([text], [TopicResult('gun issues', 0.9)], 8)
            self.assertEqual(result[0].topic, 'gun issues')
            classifier._model.batch_extract.assert_not_called()
