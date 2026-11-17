"""Authenticated, non-executing assistant previews with fail-closed policy screening."""
from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import re
import threading
import time
from pathlib import Path
from typing import Callable, Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from . import accounts
from .accounts import AccountError, Principal
from .assistant import AssistantError, build_preview, policy_decision, task_decision
from .assistant_model import AssistantModel, POLICY_SCHEMA, POLICY_SYSTEM, TASK_SCHEMA, TASK_SYSTEM


class Message(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    message: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = Field(default=None, max_length=2048)
    dataset_id: str | None = Field(default=None, max_length=103)
    base_model: Literal['qwen2.5-0.5b-instruct', 'qwen2.5-1.5b-instruct'] = 'qwen2.5-0.5b-instruct'


REPLIES = {
    'en': {'clarify': 'Which task do you need: ticket sorting, invoice fields, product descriptions, clinic FAQs, or review sentiment?',
           'dataset': 'Select an authorized dataset for {recipe} and provide its dataset ID.',
           'preview': 'Review this suggested job before continuing.'},
    'hi': {'clarify': 'आपको कौन सा काम चाहिए: टिकट छाँटना, इनवॉइस पढ़ना, उत्पाद विवरण, क्लिनिक FAQ या समीक्षा भावना?',
           'dataset': '{recipe} के लिए अपना अधिकृत डेटासेट चुनें और उसकी डेटासेट ID दें।',
           'preview': 'आगे बढ़ने से पहले इस सुझाए गए काम को जाँचें।'},
    'hinglish': {'clarify': 'Aapko kaunsa kaam chahiye: ticket sorting, invoice fields, product descriptions, clinic FAQ ya review sentiment?',
                 'dataset': '{recipe} ke liye apna authorized dataset select karke uski dataset ID dein.',
                 'preview': 'Aage badhne se pehle is suggested job ko check karein.'},
}


def _subject(who: Principal) -> str:
    return f'{who.via}:{who.user_id if who.via == "session" else who.key_id}'


def _encode(state: dict, secret: str) -> str:
    payload = json.dumps(state, separators=(',', ':'), sort_keys=True).encode()
    sig = hmac.new(secret.encode(), payload, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(payload + sig).decode().rstrip('=')


def _decode(token: str, secret: str, subject: str) -> dict:
    try:
        raw = base64.b64decode(token + '=' * (-len(token) % 4), altchars=b'-_', validate=True)
        if base64.urlsafe_b64encode(raw).decode().rstrip('=') != token:
            raise ValueError('non-canonical token')
        payload, sig = raw[:-32], raw[-32:]
        if not hmac.compare_digest(sig, hmac.new(secret.encode(), payload, hashlib.sha256).digest()):
            raise ValueError('signature')
        state = json.loads(payload)
        if (not isinstance(state, dict) or set(state) != {'subject', 'expires', 'recipe_id', 'language'}
                or state['subject'] != subject or type(state['expires']) is not int
                or state['expires'] < int(time.time()) or state['expires'] > int(time.time()) + 1800
                or state['language'] not in ('en', 'hi', 'hinglish')
                or (state['recipe_id'] is not None and state['recipe_id'] not in accounts_recipes())):
            raise ValueError('state')
        return state
    except (ValueError, TypeError, KeyError, UnicodeDecodeError, binascii.Error):
        raise AccountError('INVALID_CONVERSATION', 'Start a new assistant conversation.', 400) from None


def accounts_recipes():
    from ekido_core.recipes import RECIPES
    return RECIPES


def _dataset(root: Path | None, dataset_id: str, engine: Engine, who: Principal) -> dict:
    if root is None:
        raise AccountError('ASSISTANT_UNAVAILABLE', 'Dataset storage is not configured.', 503)
    if not re.fullmatch(r'ds_[A-Za-z0-9_-]{1,100}', dataset_id):
        raise AccountError('NOT_FOUND', 'No such dataset.', 404)
    try:
        path = (root / f'{dataset_id}.json').resolve()
        path.relative_to(root)
        with path.open('rb') as handle:
            raw = handle.read(16_385)
        if len(raw) > 16_384:
            raise ValueError('too large')
        value = json.loads(raw)
        if not isinstance(value, dict) or not isinstance(value.get('org_id'), str) or not value['org_id']:
            raise ValueError('invalid owner')
    except (ValueError, OSError, UnicodeError):
        raise AccountError('NOT_FOUND', 'No such dataset.', 404) from None
    with Session(engine) as session:
        accounts.require_role(session, who, value['org_id'], *accounts.ROLES)
        if who.via == 'api_key' and 'jobs' not in who.scopes:
            raise AccountError('FORBIDDEN', 'The API key needs jobs scope.', 403)
    return value


def build_assistant_router(engine: Engine, principal: Callable, engine_url: str, state_secret: str,
                           datasets_root: str, *, transport=None) -> APIRouter:
    router = APIRouter(prefix='/v1')
    root = Path(datasets_root).resolve() if datasets_root else None
    try:
        model = AssistantModel(engine_url, transport=transport) if engine_url else None
    except AssistantError:
        model = None
    permits = threading.BoundedSemaphore(2)

    @router.post('/assistant/messages')
    def message(body: Message, who: Principal = Depends(principal)) -> dict:
        if model is None or len(state_secret) < 32:
            raise AccountError('ASSISTANT_UNAVAILABLE', 'Assistant is not configured.', 503)
        if who.via == 'api_key' and 'jobs' not in who.scopes:
            raise AccountError('FORBIDDEN', 'The API key needs jobs scope.', 403)
        subject = _subject(who)
        state = _decode(body.conversation_id, state_secret, subject) if body.conversation_id else {
            'subject': subject, 'expires': int(time.time()) + 1800, 'recipe_id': None, 'language': 'en'}
        dataset = _dataset(root, body.dataset_id, engine, who) if body.dataset_id else None
        if not permits.acquire(blocking=False):
            raise AccountError('ASSISTANT_BUSY', 'Assistant is busy. Try again.', 503)
        try:
            decision = policy_decision(model.decide(POLICY_SYSTEM, body.message, POLICY_SCHEMA))
            if decision != 'allow':
                return {'conversation_id': _encode(state, state_secret), 'refused': decision,
                        'reply': 'This request cannot proceed without policy review.' if decision == 'review'
                                 else 'I cannot help create a model for that request.'}
            context = f"Previously selected recipe: {state['recipe_id'] or 'none'}.\nCurrent message:\n{body.message}"
            task = task_decision(model.decide(TASK_SYSTEM, context, TASK_SCHEMA))
        except AssistantError:
            raise AccountError('ASSISTANT_UNAVAILABLE', 'Assistant could not safely process this request.', 503) from None
        finally:
            permits.release()
        if task['action'] == 'propose':
            state['recipe_id'] = task['recipe_id']
        state['language'] = task['language']
        conversation_id = _encode(state, state_secret)
        if task['action'] == 'clarify' or state['recipe_id'] is None:
            return {'conversation_id': conversation_id,
                    'reply': REPLIES[state['language']]['clarify']}
        if dataset is None:
            return {'conversation_id': conversation_id,
                    'reply': REPLIES[state['language']]['dataset'].format(recipe=state['recipe_id'])}
        try:
            proposal, spec = build_preview(state['recipe_id'], body.dataset_id, dataset, body.base_model)
        except AssistantError:
            raise AccountError('INVALID_DATASET', 'Dataset does not match this recipe or has invalid metadata.', 422) from None
        return {'conversation_id': conversation_id, 'reply': REPLIES[state['language']]['preview'],
                'proposal': proposal, 'job_spec': spec, 'policy_status': 'model_screened_needs_review'}

    return router
