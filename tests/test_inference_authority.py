import json
import os
import tempfile
import threading
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, delete, event, select
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
            _, self.live_key = accounts.create_api_key(session, org.id, user.id, 'predict live', 'live', ['models.predict'])
            _, self.no_scope = accounts.create_api_key(session, org.id, user.id, 'no scope', 'test', ['jobs'])
            other, _ = accounts.finish_login(session, accounts.start_login(session, 'two@example.com'))
            other_org = accounts.create_org(session, other.id, 'Team Two')
            _, self.other_key = accounts.create_api_key(session, other_org.id, other.id, 'predict', 'test', ['models.predict'])
            session.commit()
        self.manifest = {'job_id': 'job1', 'engine_url': 'http://127.0.0.1:19081', 'engine_model': 'local-model', 'mode': 'test'}
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

    def test_test_keys_cannot_reserve_live_or_unmarked_models(self):
        self.assertEqual(self.reserve(key=self.live_key).status_code, 404)
        for mode in ('live', None, 'invalid'):
            with self.subTest(mode=mode):
                if mode is None:
                    self.manifest.pop('mode', None)
                else:
                    self.manifest['mode'] = mode
                self.write_manifest()
                self.assertEqual(self.reserve().status_code, 404)
        with Session(self.app.state.engine) as session:
            self.assertEqual(session.query(InferenceUsage).count(), 0)
        self.manifest.pop('mode', None)
        self.write_manifest()
        self.assertEqual(self.reserve(key=self.live_key).status_code, 200)

    def test_batch_finalization_is_atomic_and_exact_retry_is_idempotent(self):
        first, second = self.reserve().json()['reservation_id'], self.reserve().json()['reservation_id']
        url = '/internal/inference/usage-batch'
        items = [{'reservation_id': first, 'input_tokens': 3, 'output_tokens': 2, 'latency_ms': 5},
                 {'reservation_id': second, 'input_tokens': 6, 'output_tokens': 4, 'latency_ms': 9}]
        self.assertEqual(self.client.post(url, json={'items': items}).status_code, 401)
        missing = [items[0], {**items[1], 'reservation_id': str(uuid.uuid4())}]
        self.assertEqual(self.client.post(url, headers=self.headers, json={'items': missing}).status_code, 404)
        self.assertEqual(self.client.post(url, headers=self.headers, json={'items': [items[0], items[0]]}).status_code, 422)
        self.assertEqual(self.client.post(url, headers=self.headers, json={'items': []}).status_code, 422)
        self.assertEqual(self.client.post(url, headers=self.headers,
            json={'items': [items[0], {**items[1], 'input_tokens': '3'}]}).status_code, 422)
        self.assertEqual(self.client.post(url, headers=self.headers,
            json={'items': [items[0], {**items[1], 'reservation_id': 'z' * 36}]}).status_code, 422)
        with Session(self.app.state.engine) as session:
            self.assertEqual([session.get(InferenceUsage, i).status for i in (first, second)], ['pending', 'pending'])
        self.assertEqual(self.client.post(url, headers=self.headers, json={'items': items}).status_code, 200)
        self.assertEqual(self.client.post(url, headers=self.headers, json={'items': items[::-1]}).status_code, 200)
        different = [items[0], {**items[1], 'input_tokens': 7}]
        self.assertEqual(self.client.post(url, headers=self.headers, json={'items': different}).status_code, 409)
        with Session(self.app.state.engine) as session:
            self.assertEqual([(session.get(InferenceUsage, i).status, session.get(InferenceUsage, i).input_tokens)
                              for i in (first, second)], [('ok', 3), ('ok', 6)])

    def test_batch_rejects_finished_or_mixed_keys_without_partial_success(self):
        first, second = self.reserve().json()['reservation_id'], self.reserve().json()['reservation_id']
        url = '/internal/inference/usage-batch'
        items = [{'reservation_id': i, 'input_tokens': 1, 'output_tokens': 2, 'latency_ms': 3}
                 for i in (first, second)]
        self.assertEqual(self.client.post(f'/internal/inference/usage/{second}', headers=self.headers,
            json={'status': 'error', 'latency_ms': 1}).status_code, 200)
        self.assertEqual(self.client.post(url, headers=self.headers, json={'items': items}).status_code, 409)
        with Session(self.app.state.engine) as session:
            self.assertEqual(session.get(InferenceUsage, first).status, 'pending')
            # Simulate a differently owned reservation without relying on a second model.
            row = session.get(InferenceUsage, second)
            row.status = 'pending'
            row.key_id = session.scalar(select(accounts.ApiKey.id).where(accounts.ApiKey.key_hash == accounts._hash(self.other_key)))
            session.commit()
        self.assertEqual(self.client.post(url, headers=self.headers, json={'items': items}).status_code, 422)
        with Session(self.app.state.engine) as session:
            self.assertEqual([session.get(InferenceUsage, i).status for i in (first, second)], ['pending', 'pending'])

    def test_batch_rolls_back_when_commit_fails(self):
        first, second = self.reserve().json()['reservation_id'], self.reserve().json()['reservation_id']
        items = [{'reservation_id': i, 'input_tokens': 1, 'output_tokens': 2, 'latency_ms': 3}
                 for i in (first, second)]
        def fail_flush(session, _context, _instances):
            if any(isinstance(row, InferenceUsage) and row.status == 'ok' for row in session.dirty):
                raise RuntimeError('simulated database failure')
        event.listen(Session, 'before_flush', fail_flush)
        try:
            with self.assertRaisesRegex(RuntimeError, 'simulated database failure'):
                self.client.post('/internal/inference/usage-batch', headers=self.headers, json={'items': items})
        finally:
            event.remove(Session, 'before_flush', fail_flush)
        with Session(self.app.state.engine) as session:
            self.assertEqual([session.get(InferenceUsage, i).status for i in (first, second)], ['pending', 'pending'])

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
    def test_concurrent_overlapping_batches_commit_without_partial_success(self):
        engine = create_engine(os.environ['TEST_DATABASE_URL'])
        self.addCleanup(engine.dispose)
        email = f'{uuid.uuid4().hex}@example.com'
        ids = [str(uuid.uuid4()) for _ in range(3)]
        with Session(engine) as session:
            user, _ = accounts.finish_login(session, accounts.start_login(session, email))
            org = accounts.create_org(session, user.id, 'Atomic usage test')
            key, _ = accounts.create_api_key(session, org.id, user.id, 'test', 'test', ['models.predict'])
            session.flush()
            key_id, org_id, user_id = key.id, org.id, user.id
            for item in ids:
                session.add(InferenceUsage(id=item, key_id=key_id, model_id='m', created_at=accounts.utcnow(), status='pending'))
            session.commit()
        def cleanup():
            with Session(engine) as session, session.begin():
                session.execute(delete(InferenceUsage).where(InferenceUsage.key_id == key_id))
                session.execute(delete(accounts.ApiKey).where(accounts.ApiKey.id == key_id))
                session.execute(delete(accounts.OrgMember).where(accounts.OrgMember.org_id == org_id))
                session.execute(delete(accounts.WebSession).where(accounts.WebSession.user_id == user_id))
                session.execute(delete(accounts.UserEmail).where(accounts.UserEmail.user_id == user_id))
                session.execute(delete(accounts.LoginToken).where(accounts.LoginToken.email == email))
                session.execute(delete(accounts.Org).where(accounts.Org.id == org_id))
                session.execute(delete(accounts.User).where(accounts.User.id == user_id))
        self.addCleanup(cleanup)
        with tempfile.TemporaryDirectory() as temp:
            token = 'concurrency-test-token-' + 'x' * 32
            app = create_app(Settings(database_url=os.environ['TEST_DATABASE_URL'], _env_file=None,
                inference_models_root=temp, inference_internal_token=token, inference_rate_per_minute=10), engine=engine)
            barrier = threading.Barrier(2)
            def call(batch):
                items = [{'reservation_id': i, 'input_tokens': 1, 'output_tokens': 1, 'latency_ms': 1} for i in batch]
                with TestClient(app) as client:
                    barrier.wait(timeout=15)
                    return client.post('/internal/inference/usage-batch', headers={'x-ekido-inference-token': token},
                                       json={'items': items}).status_code
            with ThreadPoolExecutor(max_workers=2) as pool:
                statuses = list(pool.map(call, ([ids[0], ids[1]], [ids[1], ids[2]])))
            self.assertEqual(sorted(statuses), [200, 409])
            with Session(engine) as session:
                states = [session.get(InferenceUsage, i).status for i in ids]
                self.assertIn(states, (['ok', 'ok', 'pending'], ['pending', 'ok', 'ok']))

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
                                                        'engine_model': 'm', 'mode': 'test'}), encoding='utf-8')
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
