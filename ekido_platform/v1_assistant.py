"""The assistant: the front door to the recipes (decisions A1, A3, G1, G2 and U4, 24 Nov).

A person describes the job in their own words. The assistant screens it twice (Qwen3Guard, then our own policy check),
picks one of the recipes, and hands over to the normal upload, quote and pay flow. It never builds a job spec or starts
anything: the quote path builds the only spec, from trusted inputs.

What the models write is never shown or executed. They only classify, and every reply is fixed text here. Message text
is never stored: a conversation is a short-lived signed token, and a request Ekido can't do yet is counted by kind only.
It's off until switched on per team in the admin area, while it is evaluated.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import re
import threading
import time
from collections import deque
from datetime import date, datetime
from typing import Callable

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import Boolean, Date, DateTime, ForeignKey, Integer, String, select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Mapped, Session, mapped_column

from ekido_core.recipes import RECIPES

from . import accounts
from .accounts import AccountError, Principal
from .assistant_language import detect_language
from .assistant import AssistantError, guard_decision, policy_decision, task_decision
from .assistant_cards import CARDS_VERSION
from .assistant_model import AssistantModel, GuardModel, POLICY_SCHEMA, POLICY_SYSTEM, TASK_SCHEMA, TASK_SYSTEM
from .models import Base, utcnow
from .v1_datasets import Dataset

UUID = re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}')
RATE_LIMIT, RATE_WINDOW = 20, 600  # messages per person per 10 minutes


class AssistantTeam(Base):
    """Whether a team may use the assistant. Off by default; switched in the admin area (audited)."""

    __tablename__ = 'assistant_teams'
    org_id: Mapped[str] = mapped_column(String(32), ForeignKey('orgs.id'), primary_key=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class AssistantUnsupported(Base):
    """Requests Ekido can't train yet, counted by day and kind only. No text, no person, no team."""

    __tablename__ = 'assistant_unsupported'
    day: Mapped[date] = mapped_column(Date, primary_key=True)
    kind: Mapped[str] = mapped_column(String(32), primary_key=True)
    count: Mapped[int] = mapped_column(Integer, default=0)


class Message(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    message: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = Field(default=None, max_length=2048)
    # A signed-in person names the team they're working in; an API key already belongs to one.
    org_id: str | None = Field(default=None, max_length=32)
    dataset_id: str | None = Field(default=None, max_length=36)


TASKS = {'en': 'ticket sorting, invoice fields, product descriptions, customer-question answers, or review sentiment',
         'hi': 'टिकट छाँटना, इनवॉइस पढ़ना, उत्पाद विवरण, ग्राहकों के सवालों के जवाब, या समीक्षा भावना',
         'hinglish': 'ticket sorting, invoice fields, product descriptions, customer questions ke jawab, ya review sentiment'}
REPLIES = {
    'en': {'clarify': 'Which of these fits best: {tasks}?',
           'upload': 'This fits "{title}". Next, upload your examples as a CSV with the columns {columns}.',
           'quote': 'This fits "{title}", and your dataset matches. Next, get a quote.',
           'unsupported': "Ekido can't train a model for that yet. Today it can do {tasks}.",
           'closest': ' The closest is "{title}".',
           'refuse': 'I can\'t help create a model for that request.',
           'review': 'This request needs a policy review before it can go ahead.'},
    'hi': {'clarify': 'इनमें से कौन सा काम सबसे सही है: {tasks}?',
           'upload': 'यह "{title}" के लिए सही है। अब अपने उदाहरण CSV में अपलोड करें, कॉलम: {columns}।',
           'quote': 'यह "{title}" के लिए सही है, और आपका डेटासेट मेल खाता है। अब कोटेशन लें।',
           'unsupported': 'Ekido अभी इसके लिए मॉडल नहीं बना सकता। आज यह कर सकता है: {tasks}।',
           'closest': ' सबसे नज़दीकी: "{title}"।',
           'refuse': 'मैं इस अनुरोध के लिए मॉडल बनाने में मदद नहीं कर सकता।',
           'review': 'आगे बढ़ने से पहले इस अनुरोध की नीति समीक्षा ज़रूरी है।'},
    'hinglish': {'clarify': 'Inmein se kaunsa kaam sahi hai: {tasks}?',
                 'upload': 'Ye "{title}" ke liye sahi hai. Ab apne examples CSV mein upload karein, columns: {columns}.',
                 'quote': 'Ye "{title}" ke liye sahi hai, aur aapka dataset match karta hai. Ab quote lijiye.',
                 'unsupported': 'Ekido abhi iske liye model nahi bana sakta. Aaj ye kar sakta hai: {tasks}.',
                 'closest': ' Sabse kareeb: "{title}".',
                 'refuse': 'Main is request ke liye model banane mein madad nahi kar sakta.',
                 'review': 'Aage badhne se pehle is request ka policy review zaroori hai.'},
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
                or (state['recipe_id'] is not None and state['recipe_id'] not in RECIPES)):
            raise ValueError('state')
        return state
    except (ValueError, TypeError, KeyError, UnicodeDecodeError, binascii.Error):
        raise AccountError('INVALID_CONVERSATION', 'Start a new assistant conversation.', 400) from None


def count_unsupported(engine: Engine, kind: str) -> None:
    """Add one to today's count for this kind. Two requests at once both count (the second retries as an update)."""
    today = utcnow().date()
    for _ in range(2):
        with Session(engine) as session:
            row = session.get(AssistantUnsupported, (today, kind))
            if row is None:
                session.add(AssistantUnsupported(day=today, kind=kind, count=1))
            else:
                row.count += 1
            try:
                session.commit()
                return
            except IntegrityError:
                session.rollback()


class RateLimit:
    """At most `limit` messages per person per `window` seconds, in this process (Valkey later, decision 5)."""

    def __init__(self, limit: int = RATE_LIMIT, window: int = RATE_WINDOW, max_people: int = 10_000):
        self.limit, self.window, self.max_people = limit, window, max_people
        self.seen: dict[str, deque] = {}
        self.lock = threading.Lock()

    def allow(self, subject: str, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        with self.lock:
            if subject not in self.seen and len(self.seen) >= self.max_people:
                for key in [k for k, q in self.seen.items() if not q or now - q[-1] > self.window]:
                    del self.seen[key]
                if len(self.seen) >= self.max_people:
                    return False
            times = self.seen.setdefault(subject, deque())
            while times and now - times[0] > self.window:
                times.popleft()
            if len(times) >= self.limit:
                return False
            times.append(now)
            return True


class RefusalBreaker:
    """After `limit` refused requests from one person within `window` seconds, stop answering them for the rest of the window, so the
    screening models cannot be probed over and over. In this process, like RateLimit."""

    def __init__(self, limit: int = 5, window: int = 3600, max_people: int = 10_000):
        self.limit, self.window, self.max_people = limit, window, max_people
        self.refused: dict[str, deque] = {}
        self.lock = threading.Lock()

    def _recent(self, subject: str, now: float) -> deque:
        times = self.refused.get(subject, deque())
        while times and now - times[0] > self.window:
            times.popleft()
        return times

    def open(self, subject: str, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        with self.lock:
            return len(self._recent(subject, now)) >= self.limit

    def record(self, subject: str, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        with self.lock:
            if subject not in self.refused and len(self.refused) >= self.max_people:
                for key in [k for k, q in self.refused.items() if not q or now - q[-1] > self.window]:
                    del self.refused[key]
                if len(self.refused) >= self.max_people:
                    return
            times = self._recent(subject, now)
            times.append(now)
            self.refused[subject] = times


def build_assistant_router(engine: Engine, principal: Callable, engine_url: str, state_secret: str, *,
                           guard_url: str = '', transport=None, limiter: RateLimit | None = None,
                           breaker: RefusalBreaker | None = None) -> APIRouter:
    router = APIRouter(prefix='/v1')
    try:
        model = AssistantModel(engine_url, transport=transport) if engine_url else None
    except AssistantError:
        model = None
    try:
        guard = GuardModel(guard_url, transport=transport) if guard_url else None
    except AssistantError:
        guard = None
    permits = threading.BoundedSemaphore(2)
    limiter = limiter or RateLimit()
    breaker = breaker or RefusalBreaker()

    def team_of(who: Principal, org_id: str | None) -> str:
        if who.via == 'api_key':
            if 'jobs' not in who.scopes:
                raise AccountError('FORBIDDEN', 'The API key needs jobs scope.', 403)
            return who.org_id
        if not org_id:
            raise AccountError('ORG_REQUIRED', 'Say which team this is for (org_id).', 400)
        with Session(engine) as session:
            accounts.require_role(session, who, org_id, *accounts.ROLES)
        return org_id

    @router.post('/assistant/messages')
    def message(body: Message, who: Principal = Depends(principal)) -> dict:
        if model is None or guard is None or len(state_secret) < 32:
            # The guard is required too: without it the assistant doesn't answer (fail closed, decision G1).
            raise AccountError('ASSISTANT_UNAVAILABLE', 'Assistant is not configured.', 503)
        org_id = team_of(who, body.org_id)
        with Session(engine) as session:
            switch = session.get(AssistantTeam, org_id)
            if switch is None or not switch.enabled:
                raise AccountError('ASSISTANT_NOT_ENABLED', 'The assistant is not switched on for this team yet.', 403)
            dataset = None
            if body.dataset_id is not None:
                found = session.get(Dataset, body.dataset_id) if UUID.fullmatch(body.dataset_id) else None
                if found is None or found.org_id != org_id:
                    raise AccountError('NOT_FOUND', 'No such dataset.', 404)
                dataset = {'id': found.id, 'recipe_id': found.recipe_id}
        subject = _subject(who)
        if not limiter.allow(subject):
            raise AccountError('RATE_LIMITED', 'Too many assistant messages. Try again in a few minutes.', 429)
        if breaker.open(subject):
            raise AccountError('ASSISTANT_PAUSED', 'Several requests were refused, so the assistant is paused for you for a while.', 429)
        state = _decode(body.conversation_id, state_secret, subject) if body.conversation_id else {
            'subject': subject, 'expires': int(time.time()) + 1800, 'recipe_id': None, 'language': 'en'}
        if not permits.acquire(blocking=False):
            raise AccountError('ASSISTANT_BUSY', 'Assistant is busy. Try again.', 503)
        spoken = detect_language(body.message)  # by script and common words: the model often mislabels Hinglish as Hindi
        try:
            # 1. Qwen3Guard, a separate model trained for safety (G1). 2. Our own policy check. "Unsafe" stops here. "Unsure" goes on:
            # the policy check decides (Piyush, 25 Nov: the guard blocked ordinary invoice requests as unsure).
            screened, _ = guard_decision(guard.classify(body.message))
            decision = 'refuse' if screened == 'refuse' else policy_decision(model.decide(POLICY_SYSTEM, body.message, POLICY_SCHEMA))
            if decision != 'allow':
                if decision == 'refuse':
                    breaker.record(subject)
                state['language'] = spoken
                return {'conversation_id': _encode(state, state_secret), 'refused': decision,
                        'reply': REPLIES[spoken][decision], 'cards_version': CARDS_VERSION}
            context = f"Previously selected recipe: {state['recipe_id'] or 'none'}.\nCurrent message:\n{body.message}"
            task = task_decision(model.decide(TASK_SYSTEM, context, TASK_SCHEMA))
        except AssistantError:
            raise AccountError('ASSISTANT_UNAVAILABLE', 'Assistant could not safely process this request.', 503) from None
        finally:
            permits.release()
        language = state['language'] = spoken
        replies = REPLIES[language]
        answer = {'language': language, 'cards_version': CARDS_VERSION}
        if task['action'] == 'unsupported':
            count_unsupported(engine, task['kind'])
            reply = replies['unsupported'].format(tasks=TASKS[language])
            if task['recipe_id']:
                reply += replies['closest'].format(title=RECIPES[task['recipe_id']].title)
                answer['closest_recipe_id'] = task['recipe_id']
            return {**answer, 'conversation_id': _encode(state, state_secret), 'unsupported': True, 'reply': reply}
        if task['action'] == 'propose':
            state['recipe_id'] = task['recipe_id']
        conversation_id = _encode(state, state_secret)
        if task['action'] == 'clarify' or state['recipe_id'] is None:
            return {**answer, 'conversation_id': conversation_id, 'reply': replies['clarify'].format(tasks=TASKS[language])}
        recipe = RECIPES[state['recipe_id']]
        proposal = {'recipe_id': recipe.recipe_id, 'next': 'upload'}
        if recipe.labels:
            proposal['labels'] = list(recipe.labels)
        if dataset is not None and dataset['recipe_id'] == recipe.recipe_id:
            proposal.update(dataset_id=dataset['id'], next='quote')
        key = 'quote' if proposal['next'] == 'quote' else 'upload'
        return {**answer, 'conversation_id': conversation_id, 'proposal': proposal,
                'policy_status': 'model_screened_needs_review',
                'reply': replies[key].format(title=recipe.title, columns=' and '.join(recipe.columns))}

    return router
