import json
import os
import tempfile
import threading
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, delete, select
from sqlalchemy.orm import Session

from ekido_platform import accounts
from ekido_platform.accounts import InferenceUsage, JobOwner
from ekido_platform.main import Settings, create_app


class InferenceAuthority(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.token = 'internal-test-secret-' + 'x' * 32
        self.app = create_app(Settings(database_url='sqlite://', _env_file=None,
            inference_models_root=str(self.root), inference_internal_token=self.token, inference_rate_per_minute=2))
        with Session(self.app.state.engine) as session:
            user, _ = accounts.finish_login(session, accounts.start_login(session, 'one@example.com'))
            org = accounts.create_org(session, user.id, 'Team One')
            session.add(JobOwner(job_id='job1', org_id=org.id))
            _, self.key = accounts.create_api_key(session, org.id, user.id, 'predict', 'test', ['models.predict'])
            _, self.no_scope = accounts.create_api_key(session, org.id, user.id, 'no scope', 'test', ['jobs'])
            other, _ = accounts.finish_login(session, accounts.start_login(session, 'two@example.com'))
            other_org = accounts.create_org(session, other.id, 'Team Two')
            _, self.other_key = accounts.create_api_key(session, other_org.id, other.id, 'predict', 'test', ['models.predict'])
            session.commit()
        self.manifest = {'job_id': 'job1', 'engine_url': 'http://127.0.0.1:19081', 'engine_model': 'local-model'}
        self.write_manifest()
        self.client = TestClient(self.app)
        self.headers = {'x-ekido-inference-token': self.token}

    def write_manifest(self):
        (self.root / 'model1.json').write_text(json.dumps(self.manifest), encoding='utf-8')

    def reserve(self, key=None, headers=None):
        return self.client.post('/internal/inference/reserve', headers=headers or self.headers,
                                json={'api_key': key or self.key, 'model_id': 'model1'})

    def test_authorization_scope_ownership_and_revocation(self):
        self.assertEqual(self.reserve(headers={'x-ekido-inference-token': 'bad'}).status_code, 401)
        self.assertEqual(self.reserve(key='ek_test_invalid').status_code, 401)
        self.assertEqual(self.reserve(key=self.no_scope).status_code, 403)
        self.assertEqual(self.reserve(key=self.other_key).status_code, 404)
        with Session(self.app.state.engine) as session:
            key_id = session.scalar(select(accounts.ApiKey.id).where(accounts.ApiKey.key_hash == accounts._hash(self.key)))
            key = session.get(accounts.ApiKey, key_id)
            key.revoked_at = accounts.utcnow()
            session.commit()
        self.assertEqual(self.reserve().status_code, 401)

    def test_reservations_limit_and_usage_cannot_be_finalized_twice(self):
        first = self.reserve()
        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(first.json()['engine_url'], self.manifest['engine_url'])
        reservation = first.json()['reservation_id']
        second = self.reserve()
        self.assertEqual(second.status_code, 200)
        self.assertEqual(self.reserve().status_code, 429)
        url = f'/internal/inference/usage/{reservation}'
        self.assertEqual(self.client.post(url, headers=self.headers, json={'status': 'ok', 'latency_ms': 4}).status_code, 422)
        usage = {'status': 'ok', 'latency_ms': 20, 'input_tokens': 3, 'output_tokens': 2}
        self.assertEqual(self.client.post(url, headers=self.headers, json=usage).status_code, 200)
        self.assertEqual(self.client.post(url, headers=self.headers, json=usage).status_code, 409)
        partial = {'status': 'error', 'latency_ms': 30, 'input_tokens': 4, 'output_tokens': 1}
        second_url = f"/internal/inference/usage/{second.json()['reservation_id']}"
        self.assertEqual(self.client.post(second_url, headers=self.headers, json=partial).status_code, 200)
        with Session(self.app.state.engine) as session:
            row = session.get(InferenceUsage, reservation)
            self.assertEqual((row.status, row.input_tokens, row.output_tokens), ('ok', 3, 2))
            self.assertNotIn(self.key, str(row.__dict__))

    def test_untrusted_or_malformed_manifest_is_not_routed(self):
        for url in ('http://example.com:80', 'http://10.0.0.1:1234', 'http://127.0.0.1:1234/x',
                    'http://a:b@127.0.0.1:1234', 'https://127.0.0.1:1234'):
            with self.subTest(url=url):
                self.manifest['engine_url'] = url
                self.write_manifest()
                self.assertEqual(self.reserve().status_code, 404)

    def test_unconfigured_authority_fails_closed(self):
        app = create_app(Settings(database_url='sqlite://', _env_file=None))
        result = TestClient(app).post('/internal/inference/reserve', json={'api_key': self.key, 'model_id': 'model1'})
        self.assertEqual(result.status_code, 503)

    def test_readiness_requires_internal_token_and_a_valid_owned_model(self):
        url = '/internal/inference/ready'
        self.assertEqual(self.client.get(url).status_code, 401)
        ready = self.client.get(url, headers=self.headers)
        self.assertEqual(ready.status_code, 200)
        self.assertEqual(ready.json()['engines'], ['http://127.0.0.1:19081'])
        self.manifest['job_id'] = 'missing'
        self.write_manifest()
        self.assertEqual(self.client.get(url, headers=self.headers).status_code, 503)


@unittest.skipUnless(os.environ.get('TEST_DATABASE_URL', '').startswith('postgresql'), 'requires migrated Postgres')
class PostgresRateLimit(unittest.TestCase):
    def test_concurrent_reservations_obey_limit(self):
        engine = create_engine(os.environ['TEST_DATABASE_URL'])
        self.addCleanup(engine.dispose)
        email = f'{uuid.uuid4().hex}@example.com'
        with Session(engine) as session:
            user, _ = accounts.finish_login(session, accounts.start_login(session, email))
            org = accounts.create_org(session, user.id, 'Inference concurrency test')
            session.add(JobOwner(job_id=str(uuid.uuid4()), org_id=org.id))
            session.flush()
            job_id = session.scalar(select(JobOwner.job_id).where(JobOwner.org_id == org.id))
            key, secret = accounts.create_api_key(session, org.id, user.id, 'test', 'test', ['models.predict'])
            key_id, org_id, user_id = key.id, org.id, user.id
        def cleanup():
            with Session(engine) as session, session.begin():
                session.execute(delete(InferenceUsage).where(InferenceUsage.key_id == key_id))
                session.execute(delete(accounts.ApiKey).where(accounts.ApiKey.id == key_id))
                session.execute(delete(JobOwner).where(JobOwner.org_id == org_id))
                session.execute(delete(accounts.OrgMember).where(accounts.OrgMember.org_id == org_id))
                session.execute(delete(accounts.WebSession).where(accounts.WebSession.user_id == user_id))
                session.execute(delete(accounts.UserEmail).where(accounts.UserEmail.user_id == user_id))
                session.execute(delete(accounts.LoginToken).where(accounts.LoginToken.email == email))
                session.execute(delete(accounts.Org).where(accounts.Org.id == org_id))
                session.execute(delete(accounts.User).where(accounts.User.id == user_id))
        self.addCleanup(cleanup)
        with tempfile.TemporaryDirectory() as temp:
            Path(temp, 'm.json').write_text(json.dumps({'job_id': job_id, 'engine_url': 'http://127.0.0.1:9000',
                                                        'engine_model': 'm'}), encoding='utf-8')
            token = 'concurrency-test-token-' + 'x' * 32
            settings = Settings(database_url=os.environ['TEST_DATABASE_URL'], _env_file=None,
                inference_models_root=temp, inference_internal_token=token, inference_rate_per_minute=2)
            app = create_app(settings, engine=engine)
            barrier = threading.Barrier(6)
            def call():
                with TestClient(app) as client:
                    barrier.wait(timeout=15)
                    return client.post('/internal/inference/reserve', headers={'x-ekido-inference-token': token},
                                       json={'api_key': secret, 'model_id': 'm'}).status_code
            with ThreadPoolExecutor(max_workers=6) as pool:
                statuses = list(pool.map(lambda _: call(), range(6)))
            self.assertEqual(sorted(statuses), [200, 200, 429, 429, 429, 429])


if __name__ == '__main__':
    unittest.main()
