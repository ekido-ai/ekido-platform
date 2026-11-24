"""Validate the assistant models' constrained output.

The models only classify: is the request allowed, and which recipe fits. Nothing they write is shown to anyone or
executed; replies are fixed text, and the job spec is built later by the quote path from trusted inputs.
"""
from __future__ import annotations

import json
import re
from typing import Any

from ekido_core.recipes import RECIPES

from .assistant_cards import UNSUPPORTED_KINDS

class AssistantError(ValueError):
    pass


def structured_object(text: str, keys: set[str]) -> dict[str, Any]:
    if not isinstance(text, str):
        raise AssistantError('model output is missing or too large')
    try:
        if len(text.encode('utf-8')) > 16_384:
            raise AssistantError('model output is missing or too large')
    except UnicodeEncodeError:
        raise AssistantError('model output is not valid text') from None
    def unique_pairs(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError('duplicate field')
            result[key] = item
        return result
    try:
        value = json.loads(text, object_pairs_hook=unique_pairs)
    except (ValueError, TypeError):
        raise AssistantError('model output is not JSON') from None
    if not isinstance(value, dict) or set(value) != keys:
        raise AssistantError('model output has unexpected fields')
    return value


GUARD_LABEL = re.compile(r'Safety: (Safe|Unsafe|Controversial)')
GUARD_CATEGORIES = ('Violent', 'Non-violent Illegal Acts', 'Sexual Content or Sexual Acts', 'PII', 'Suicide & Self-Harm',
                    'Unethical Acts', 'Politically Sensitive Topics', 'Copyright Violation', 'Jailbreak')


def guard_decision(text: str) -> tuple[str, list[str]]:
    """Qwen3Guard's verdict on a request: 'allow', 'review' (Controversial) or 'refuse' (Unsafe), with its categories.

    The guard answers in plain text, a "Safety: Unsafe" line then a "Categories: Violent" line. Anything without
    exactly one safety label is unreadable and fails closed, like every other assistant check.
    """
    if not isinstance(text, str) or len(text) > 2_000:
        raise AssistantError('guard output is missing or too large')
    labels = GUARD_LABEL.findall(text)
    if len(labels) != 1:
        raise AssistantError('guard output has no single safety label')
    categories = [c for c in GUARD_CATEGORIES if c in text]
    return {'Safe': 'allow', 'Controversial': 'review', 'Unsafe': 'refuse'}[labels[0]], categories


def policy_decision(text: str) -> str:
    value = structured_object(text, {'decision'})['decision']
    if value not in ('allow', 'refuse', 'review'):
        raise AssistantError('invalid policy decision')
    return value


def task_decision(text: str) -> dict:
    """The task classifier's choice: propose a recipe, ask one more question, or say Ekido can't do this yet."""
    result = structured_object(text, {'action', 'recipe_id', 'language', 'kind'})
    if result['action'] not in ('propose', 'clarify', 'unsupported') or result['language'] not in ('en', 'hi', 'hinglish'):
        raise AssistantError('invalid task decision')
    recipe, kind = result['recipe_id'], result['kind']
    if recipe is not None and (not isinstance(recipe, str) or recipe not in RECIPES):
        raise AssistantError('unknown recipe')
    if result['action'] == 'propose':
        if recipe is None or kind is not None:
            raise AssistantError('a proposal names one recipe and no kind')
    elif result['action'] == 'clarify':
        if recipe is not None or kind is not None:
            raise AssistantError('a clarification selects nothing')
    elif kind not in UNSUPPORTED_KINDS:  # unsupported: a known kind; recipe_id is the nearest recipe, or null
        raise AssistantError('unknown request kind')
    return result
