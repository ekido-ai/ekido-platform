import json
import unittest
from unittest.mock import patch

from ekido_core.job_spec import spec_problems
from ekido_platform import assistant


class AssistantPreview(unittest.TestCase):
    def setUp(self):
        self.dataset = {'recipe_id': 'ticket-sorter-hinglish', 'sha256': 'a' * 64,
                        'bytes': 12345, 'train_rows': 100, 'test_rows': 25}

    def test_every_recipe_preview_has_a_valid_pinned_spec(self):
        for recipe_id in assistant.RECIPES:
            with self.subTest(recipe=recipe_id):
                data = {**self.dataset, 'recipe_id': recipe_id}
                proposal, spec = assistant.build_preview(recipe_id, 'ds_test', data, 'qwen2.5-0.5b-instruct')
                self.assertEqual(proposal['recipe_id'], recipe_id)
                self.assertEqual(spec_problems(spec), [])
                self.assertEqual(spec['dataset']['sha256'], data['sha256'])
                self.assertEqual(len(spec['model']['revision']), 40)

    def test_untrusted_dataset_or_unlicensed_model_is_refused(self):
        for changed in ({'recipe_id': 'invoice-fields'}, {'sha256': '../x'}, {'train_rows': True},
                        {'test_rows': 19}, {'bytes': -1}):
            with self.subTest(metadata=changed), self.assertRaises(assistant.AssistantError):
                assistant.build_preview('ticket-sorter-hinglish', 'ds_test', {**self.dataset, **changed},
                                        'qwen2.5-0.5b-instruct')
        for model in ('unknown', [], None):
            with self.subTest(model=model), self.assertRaises(assistant.AssistantError):
                assistant.build_preview('ticket-sorter-hinglish', 'ds_test', self.dataset, model)
        with patch.object(assistant, 'spec_problems', return_value=['broken']):
            with self.assertRaises(assistant.AssistantError):
                assistant.build_preview('ticket-sorter-hinglish', 'ds_test', self.dataset, 'qwen2.5-0.5b-instruct')

    def test_structured_output_refuses_ambiguous_or_extra_fields(self):
        self.assertEqual(assistant.policy_decision('{"decision":"refuse"}'), 'refuse')
        for raw in ('not json', '{"decision":"allow","decision":"refuse"}',
                    '{"decision":"allow","extra":"x"}', '{"decision":null}'):
            with self.subTest(raw=raw), self.assertRaises(assistant.AssistantError):
                assistant.policy_decision(raw)
        good = {'action': 'propose', 'recipe_id': 'review-sentiment', 'language': 'en', 'reply': 'Use sentiment.'}
        self.assertEqual(assistant.task_decision(json.dumps(good)), good)
        for changes in ({'recipe_id': '../invalid'}, {'language': []}, {'reply': ''}, {'action': 'unknown'}):
            with self.subTest(changes=changes), self.assertRaises(assistant.AssistantError):
                assistant.task_decision(json.dumps({**good, **changes}))


if __name__ == '__main__':
    unittest.main()
