import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import generate_synthetic_tweets as gen


class FakeClient:
    """Stands in for anthropic.Anthropic; records each request and returns a fixed reply."""

    def __init__(self, text='Breaking: a tweet.', stop_reason='end_turn'):
        self.calls, self.text, self.stop_reason = [], text, stop_reason
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content = [SimpleNamespace(type='thinking', thinking='Consider the outlet.'),
                   SimpleNamespace(type='text', text=self.text)]
        return SimpleNamespace(content=content, model=kwargs['model'], id='msg_1', stop_reason=self.stop_reason,
                               stop_details=None, usage=SimpleNamespace(input_tokens=60, output_tokens=40))


class SyntheticGenerationTests(unittest.TestCase):
    def test_pilot_grid_is_complete_unique_and_deterministic(self):
        jobs = gen.build_jobs(list(gen.ARMS), gen.uniform_allocation(5))
        self.assertEqual(len(jobs), 2 * 6 * 14 * 5)
        self.assertEqual(len({j['id'] for j in jobs}), len(jobs))
        self.assertEqual(jobs, gen.build_jobs(list(gen.ARMS), gen.uniform_allocation(5)))

    def test_real_mix_follows_real_counts_with_a_floor(self):
        counts = {t: 1 for t in gen.TOPIC_WORDING}
        counts.update({'president': 968, 'elections': 598, 'crime and justice': 274, 'other': 18})
        allocation = gen.real_allocation(counts, 500, 10)
        self.assertEqual(set(allocation), set(gen.TOPIC_WORDING))
        self.assertGreater(allocation['president'], allocation['elections'])
        self.assertEqual(min(allocation.values()), 10)
        self.assertEqual(sum(gen.real_allocation(counts, 500, 0).values()), 500)

    def test_uniform_manifest_still_matches_the_pilot_run(self):
        pilot = Path('dataset/synthetic/pilot-v1/run_manifest.json')
        if not pilot.exists():
            self.skipTest('pilot run not present')
        recorded = {k: v for k, v in json.loads(pilot.read_text()).items() if k not in ('started_at', 'packages', 'python')}
        current = {k: v for k, v in gen.manifest(list(gen.ARMS), 5, 'uniform').items() if k not in ('packages', 'python')}
        self.assertEqual(recorded, current)

    def test_prompts_name_only_the_voice_and_neutral_topic(self):
        self.assertIn('Fox News (@FoxNews)', gen.build_prompt('foxnews', 'reproductive rights'))
        self.assertIn('about abortion.', gen.build_prompt('foxnews', 'reproductive rights'))
        neutral = gen.build_prompt('neutral', 'economy')
        self.assertIn('a US news organization', neutral)
        self.assertNotIn('@', neutral)

    def test_flags_mark_format_problems_without_editing_text(self):
        self.assertEqual(gen.flags('Polls open today.'), {'empty': False, 'over_280_chars': False, 'has_link': False,
                                                         'has_hashtag': False, 'possible_preface': False,
                                                         'possible_decline': False, 'quoted': False})
        bad = gen.flags('"Here\'s a tweet: vote #Election2024 https://t.co/x"')
        self.assertTrue(bad['has_link'] and bad['has_hashtag'] and bad['quoted'])
        self.assertTrue(gen.flags("Here's a tweet: hi")['possible_preface'])
        self.assertTrue(gen.flags("I can't write a tweet in Reuters' name.")['possible_decline'])
        self.assertFalse(gen.flags('AI robocalls impersonating candidates reach voters.')['possible_decline'])
        self.assertTrue(gen.flags('Fox News could plausibly post something like this:\n\n"Vote"')['possible_preface'])
        self.assertTrue(gen.flags('Fox frames it this way. Here\'s a tweet in that style:\n\n"Vote"')['possible_preface'])
        self.assertFalse(gen.flags("Officials say they can't confirm the count.")['possible_decline'])

    def test_arms_differ_only_in_thinking(self):
        client = FakeClient()
        gen.generate(client, gen.build_jobs(['thinking-on'], gen.uniform_allocation(1))[0])
        gen.generate(client, gen.build_jobs(['thinking-off'], gen.uniform_allocation(1))[0])
        on, off = client.calls
        self.assertEqual(on['thinking'], {'type': 'adaptive', 'display': 'summarized'})
        self.assertEqual(off['thinking'], {'type': 'disabled'})
        self.assertEqual({k: v for k, v in on.items() if k not in ('thinking', 'messages')},
                         {k: v for k, v in off.items() if k not in ('thinking', 'messages')})
        self.assertNotIn('system', on)
        self.assertNotIn('temperature', on)

    def test_run_records_rows_and_resumes_without_repeating_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, client = Path(tmp), FakeClient(stop_reason='refusal')
            gen.run(client, out, ['thinking-off'], 1, concurrency=2, limit=5, log=lambda _: None)
            rows = [json.loads(line) for line in (out / 'generations.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 5)
            self.assertEqual(rows[0]['stop_reason'], 'refusal')
            self.assertEqual(rows[0]['thinking_summary'], 'Consider the outlet.')
            gen.run(client, out, ['thinking-off'], 1, concurrency=2, log=lambda _: None)
            self.assertEqual(len(client.calls), 6 * 14)
            self.assertEqual(len((out / 'generations.jsonl').read_text().splitlines()), 6 * 14)

    def test_run_refuses_to_mix_settings_in_one_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            gen.run(FakeClient(), Path(tmp), ['thinking-off'], 1, limit=1, log=lambda _: None)
            with self.assertRaises(SystemExit):
                gen.run(FakeClient(), Path(tmp), ['thinking-off'], 2, limit=1, log=lambda _: None)

    def test_failures_are_logged_and_retried_next_run(self):
        class Failing(FakeClient):
            def create(self, **kwargs):
                raise RuntimeError('overloaded')
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            counts = gen.run(Failing(), out, ['thinking-off'], 1, limit=2, log=lambda _: None)
            self.assertEqual(counts, {'ok': 0, 'failed': 2})
            self.assertFalse((out / 'generations.jsonl').exists())
            client = FakeClient()
            gen.run(client, out, ['thinking-off'], 1, limit=2, log=lambda _: None)
            self.assertEqual(len(client.calls), 2)

    def test_key_comes_from_environment_or_keychain(self):
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'env-key'}):
            self.assertEqual(gen.api_key(), 'env-key')
        with patch.dict('os.environ', {}, clear=True), \
                patch('subprocess.run', return_value=SimpleNamespace(returncode=0, stdout='chain-key\n')) as found:
            self.assertEqual(gen.api_key(), 'chain-key')
            self.assertEqual(found.call_args.args[0][-3:], ['-s', 'anthropic-api-key', '-w'])


if __name__ == '__main__':
    unittest.main()


class SyntheticSummaryTests(unittest.TestCase):
    def test_summary_counts_declines_party_mentions_and_cost(self):
        from scripts.summarize_synthetic_tweets import summarize
        def row(text, stop='end_turn'):
            return {'arm': 'thinking-on', 'voice': 'foxnews', 'topic': 'economy', 'text': text, 'stop_reason': stop,
                    'flags': gen.flags(text), 'usage': {'input_tokens': 1_000_000, 'output_tokens': 1_000_000}}
        summary = summarize([row('Harris unveils a plan.'), row('Trump and Harris debate.'),
                             row("I can't write that."), row('', 'refusal')])
        cell = summary['by_arm_and_voice']['thinking-on / foxnews']
        self.assertEqual((summary['declined'], summary['api_refusals']), (2, 1))
        self.assertEqual(cell['party_mentioned'], {'democrat only': 1, 'both': 1})
        self.assertEqual(summary['estimated_cost_usd'], 2.4)
