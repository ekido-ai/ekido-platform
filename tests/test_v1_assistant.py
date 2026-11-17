import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from ekido_core.job_spec import spec_problems
from ekido_platform import accounts
from ekido_platform.assistant_model import POLICY_SYSTEM, loopback_origin
from ekido_platform.assistant import AssistantError
from ekido_platform.main import Settings, create_app


class AssistantRoute(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.policy = 'allow'
        self.task = {'action': 'propose', 'recipe_id': 'ticket-sorter-hinglish',
                     'language': 'hinglish', 'reply': 'Untrusted model text'}
        self.calls = []
        def respond(request):
            body = json.loads(request.content)
            self.calls.append(body)
            self.assertEqual(body['response_format']['type'], 'json_object')
            decision = {'decision': self.policy} if body['messages'][0]['content'] == POLICY_SYSTEM else self.task
            return httpx.Response(200, json={'choices': [{'message': {'content': json.dumps(decision)}}]})
        settings = Settings(database_url='sqlite://', _env_file=None, assistant_engine_url='http://127.0.0.1:9000',
            assistant_state_secret='assistant-secret-' + 'x' * 32, assistant_datasets_root=str(self.root))
        self.app = create_app(settings, assistant_transport=httpx.MockTransport(respond))
        with Session(self.app.state.engine) as session:
            user, self.cookie = accounts.finish_login(session, accounts.start_login(session, 'a@example.com'))
            org = accounts.create_org(session, user.id, 'Team')
            _, self.key = accounts.create_api_key(session, org.id, user.id, 'jobs', 'test', ['jobs'])
            _, self.no_scope = accounts.create_api_key(session, org.id, user.id, 'models', 'test', ['models.read'])
            other, self.other_cookie = accounts.finish_login(session, accounts.start_login(session, 'b@example.com'))
            session.commit()
            self.org_id = org.id
        (self.root / 'ds_test.json').write_text(json.dumps({'org_id': self.org_id,
            'recipe_id': 'ticket-sorter-hinglish', 'sha256': 'a' * 64, 'bytes': 1000,
            'train_rows': 30, 'test_rows': 20}), encoding='utf-8')
        self.client = TestClient(self.app)
        self.url = '/v1/assistant/messages'
        self.headers = {'authorization': f'Bearer {self.key}'}

    def test_conversation_builds_valid_spec_from_owned_metadata(self):
        first = self.client.post(self.url, headers=self.headers, json={'message': 'sort support tickets'})
        self.assertEqual(first.status_code, 200, first.text)
        self.assertNotIn('job_spec', first.json())
        self.assertNotIn('Untrusted model text', first.text)
        token = first.json()['conversation_id']
        second = self.client.post(self.url, headers=self.headers, json={'message': 'use this dataset',
            'conversation_id': token, 'dataset_id': 'ds_test'})
        self.assertEqual(second.status_code, 200, second.text)
        result = second.json()
        self.assertEqual(result['proposal']['dataset_id'], 'ds_test')
        self.assertEqual(spec_problems(result['job_spec']), [])
        self.assertEqual(result['job_spec']['dataset']['sha256'], 'a' * 64)
        self.assertEqual(result['policy_status'], 'model_screened_needs_review')
        self.assertEqual(len(self.calls), 4)

    def test_refusal_and_review_never_generate_a_proposal(self):
        for decision in ('refuse', 'review'):
            self.policy = decision
            response = self.client.post(self.url, headers=self.headers, json={'message': 'policy test'})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()['refused'], decision)
            self.assertNotIn('job_spec', response.json())
        self.assertEqual(len(self.calls), 2)  # Task model was never invoked.

    def test_ambiguous_followup_does_not_reuse_a_stale_recipe_as_a_proposal(self):
        first = self.client.post(self.url, headers=self.headers, json={'message': 'sort tickets'})
        self.task = {'action': 'clarify', 'recipe_id': None, 'language': 'en', 'reply': 'Please clarify'}
        response = self.client.post(self.url, headers=self.headers, json={'message': 'different task',
            'conversation_id': first.json()['conversation_id'], 'dataset_id': 'ds_test'})
        self.assertEqual(response.status_code, 200)
        self.assertNotIn('proposal', response.json())

    def test_untrusted_output_and_wrong_scope_fail_closed(self):
        self.policy = 'nonsense'
        self.assertEqual(self.client.post(self.url, headers=self.headers, json={'message': 'hello'}).status_code, 503)
        self.assertEqual(self.client.post(self.url, headers={'authorization': f'Bearer {self.no_scope}'},
                                           json={'message': 'hello'}).status_code, 403)
        self.assertEqual(self.client.post(self.url, json={'message': 'hello'}).status_code, 401)

    def test_engine_failure_never_returns_a_proposal(self):
        with patch('ekido_platform.assistant_model.AssistantModel.decide', side_effect=AssistantError('offline')):
            response = self.client.post(self.url, headers=self.headers, json={'message': 'private example'})
        self.assertEqual(response.status_code, 503)
        self.assertNotIn('private example', response.text)
        self.assertNotIn('proposal', response.json())

    def test_other_users_cannot_reuse_conversations_or_datasets(self):
        response = self.client.post(self.url, headers=self.headers, json={'message': 'sort tickets'})
        token = response.json()['conversation_id']
        other = TestClient(self.app)
        other.cookies.set('ekido_session', self.other_cookie)
        self.assertEqual(other.post(self.url, json={'message': 'hello', 'conversation_id': token}).status_code, 400)
        self.assertEqual(other.post(self.url, json={'message': 'hello', 'dataset_id': 'ds_test'}).status_code, 404)
        self.assertEqual(self.client.post(self.url, headers=self.headers,
            json={'message': 'hello', 'conversation_id': token[:-1] + ('A' if token[-1] != 'A' else 'B')}).status_code, 400)

    def test_loopback_engine_only_and_unconfigured_endpoint(self):
        for origin in ('https://127.0.0.1:9000', 'http://example.com:9000', 'http://127.0.0.1:9000/path'):
            with self.subTest(origin=origin), self.assertRaises(AssistantError):
                loopback_origin(origin)
        app = create_app(Settings(database_url='sqlite://', _env_file=None))
        # Identity is still required before the unavailable response.
        self.assertEqual(TestClient(app).post(self.url, json={'message': 'hi'}).status_code, 401)


if __name__ == '__main__':
    unittest.main()
