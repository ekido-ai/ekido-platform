import json
import unittest

from ekido_core.recipes import RECIPES
from ekido_platform import assistant
from ekido_platform.assistant_cards import CARDS, CARDS_VERSION, UNSUPPORTED_KINDS, task_system_prompt


class StructuredOutput(unittest.TestCase):
    def test_policy_output_refuses_ambiguous_or_extra_fields(self):
        self.assertEqual(assistant.policy_decision('{"decision":"refuse"}'), 'refuse')
        for raw in ('not json', '{"decision":"allow","decision":"refuse"}', '{"decision":"allow","extra":"x"}',
                    '{"decision":null}', '{"decision":"maybe"}', '[]', 'x' * 20_000):
            with self.subTest(raw=raw[:30]), self.assertRaises(assistant.AssistantError):
                assistant.policy_decision(raw)

    def task(self, **changes):
        base = {'action': 'propose', 'recipe_id': 'review-sentiment', 'language': 'en', 'kind': None}
        return json.dumps({**base, **changes})

    def test_a_proposal_names_exactly_one_known_recipe_and_no_kind(self):
        self.assertEqual(assistant.task_decision(self.task())['recipe_id'], 'review-sentiment')
        for changes in ({'recipe_id': None}, {'recipe_id': '../invalid'}, {'recipe_id': 5}, {'kind': 'coding'}, {'language': 'fr'},
                        {'language': []}, {'action': 'run'}):
            with self.subTest(changes=changes), self.assertRaises(assistant.AssistantError):
                assistant.task_decision(self.task(**changes))

    def test_a_clarification_selects_nothing(self):
        self.assertEqual(assistant.task_decision(self.task(action='clarify', recipe_id=None))['action'], 'clarify')
        for changes in ({'recipe_id': 'review-sentiment'}, {'kind': 'coding'}):
            with self.subTest(changes=changes), self.assertRaises(assistant.AssistantError):
                assistant.task_decision(self.task(**{'action': 'clarify', 'recipe_id': None, **changes}))

    def test_unsupported_has_a_known_kind_and_an_optional_nearest_recipe(self):
        for kind in UNSUPPORTED_KINDS:
            with self.subTest(kind=kind):
                self.assertEqual(assistant.task_decision(self.task(action='unsupported', recipe_id=None, kind=kind))['kind'], kind)
        near = assistant.task_decision(self.task(action='unsupported', recipe_id='product-descriptions', kind='marketing_copy'))
        self.assertEqual(near['recipe_id'], 'product-descriptions')
        for changes in ({'kind': None}, {'kind': 'something else'}, {'kind': 7}, {'recipe_id': 'not-a-recipe'}):
            with self.subTest(changes=changes), self.assertRaises(assistant.AssistantError):
                assistant.task_decision(self.task(**{'action': 'unsupported', 'recipe_id': None, 'kind': 'coding', **changes}))


class Cards(unittest.TestCase):
    def test_every_recipe_has_a_card_in_three_languages_of_asking(self):
        self.assertEqual(set(CARDS), set(RECIPES))
        for recipe_id, card in CARDS.items():
            with self.subTest(recipe=recipe_id):
                self.assertTrue(card['does'] and card['not'])
                self.assertGreaterEqual(len(card['asks']), 4)

    def test_the_prompt_is_built_from_the_cards_alone(self):
        prompt = task_system_prompt()
        for recipe_id in RECIPES:
            self.assertIn(recipe_id, prompt)
        for kind in UNSUPPORTED_KINDS:
            self.assertIn(kind, prompt)
        self.assertIn('untrusted data', prompt)

    def test_the_cards_version_is_a_short_stable_hash(self):
        self.assertRegex(CARDS_VERSION, r'^[0-9a-f]{12}$')


if __name__ == '__main__':
    unittest.main()
