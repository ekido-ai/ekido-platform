"""Validate constrained assistant output and construct a recipe job preview.

The model only classifies the user's task; dataset identity and execution
settings come from trusted server-side inputs, never from generated text.
"""
from __future__ import annotations

import json
import re
from typing import Any

from ekido_core.job_spec import MODEL_ALIASES as MODELS, spec_problems
from ekido_core.recipes import RECIPES

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
    result = structured_object(text, {'action', 'recipe_id', 'language', 'reply'})
    if result['action'] not in ('propose', 'clarify') or result['language'] not in ('en', 'hi', 'hinglish'):
        raise AssistantError('invalid task decision')
    if not isinstance(result['reply'], str) or not 1 <= len(result['reply'].strip()) <= 500:
        raise AssistantError('invalid reply')
    try:
        result['reply'].encode('utf-8')
    except UnicodeEncodeError:
        raise AssistantError('invalid reply') from None
    if result['action'] == 'propose':
        if not isinstance(result['recipe_id'], str) or result['recipe_id'] not in RECIPES:
            raise AssistantError('unknown recipe')
    elif result['recipe_id'] is not None:
        raise AssistantError('a clarification cannot select a recipe')
    return result


def build_preview(recipe_id: str, dataset_id: str, metadata: dict, base_model: str) -> tuple[dict, dict]:
    if not isinstance(recipe_id, str) or recipe_id not in RECIPES or not isinstance(base_model, str) or base_model not in MODELS:
        raise AssistantError('unknown recipe or model')
    if not isinstance(dataset_id, str) or not re.fullmatch(r'ds_[A-Za-z0-9_-]{1,100}', dataset_id):
        raise AssistantError('invalid dataset id')
    if not isinstance(metadata, dict) or metadata.get('recipe_id') != recipe_id:
        raise AssistantError('dataset does not match recipe')
    digest = metadata.get('sha256')
    if not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest):
        raise AssistantError('dataset fingerprint is invalid')
    for field, minimum in (('bytes', 1), ('train_rows', 1), ('test_rows', 20)):
        value = metadata.get(field)
        if type(value) is not int or not minimum <= value <= 2_147_483_647:
            raise AssistantError('dataset metadata is invalid')
    model_id, revision = MODELS[base_model]
    spec = {'template': 'llm-lora/v1', 'recipe_id': recipe_id,
            'model': {'id': model_id, 'revision': revision},
            'dataset': {field: metadata[field] for field in ('sha256', 'bytes', 'train_rows', 'test_rows')},
            'lora': {'r': 16, 'alpha': 32, 'dropout': 0.05, 'target_modules': ['q_proj', 'v_proj']},
            'max_seq_length': 512, 'logical_batch_size': min(16, metadata['train_rows']),
            'total_logical_steps': 200, 'seed': 7,
            'optimizer': {'name': 'adamw', 'lr': 2e-4, 'weight_decay': 0.0, 'betas': [0.9, 0.999],
                          'eps': 1e-8, 'max_grad_norm': 1.0},
            'lr_schedule': {'name': 'linear_warmup_linear_decay', 'warmup_steps': 10}}
    if spec_problems(spec):
        raise AssistantError('generated job spec failed validation')
    proposal = {'recipe_id': recipe_id, 'dataset_id': dataset_id, 'base_model': base_model, 'distill': False}
    if RECIPES[recipe_id].labels:
        proposal['labels'] = list(RECIPES[recipe_id].labels)
    return proposal, spec
