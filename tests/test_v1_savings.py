import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from ekido_platform import accounts
from ekido_platform.accounts import JobOwner
from ekido_platform.main import Settings, create_app


class SavingsEndpoint(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'models').mkdir()
        self.app = create_app(Settings(database_url='sqlite://', _env_file=None,
            savings_evidence_root=str(self.root), session_cookie_secure=False))
        with Session(self.app.state.engine) as session:
            user, self.token = accounts.finish_login(session, accounts.start_login(session, 'owner@example.com'))
            org = accounts.create_org(session, user.id, 'Team')
            self.org_id = org.id
            session.add(JobOwner(job_id='job1', org_id=org.id))
            _, self.key = accounts.create_api_key(session, org.id, user.id, 'reader', 'test', ['models.read'])
            _, self.no_scope = accounts.create_api_key(session, org.id, user.id, 'other', 'test', ['jobs'])
            _, self.other_token = accounts.finish_login(session, accounts.start_login(session, 'other@example.com'))
            session.commit()
        self.write('models/model1.json', {'job_id': 'job1', 'expected_requests': 1000,
            'api_run_file': 'api.json', 'model_run_file': 'model.json', 'quote_file': 'quote.json',
            'price_file': 'price.json'})
        base = {'job_id': 'job1', 'eval_set_hash': 'a' * 64, 'examples': 5, 'metric': 'accuracy', 'score': .8}
        self.write('api.json', {**base, 'requests': 5, 'input_tokens': 100, 'output_tokens': 30})
        self.write('model.json', {**base, 'model_id': 'model1'})
        self.write('quote.json', {'job_id': 'job1', 'model_id': 'model1', 'training_paise': 20,
                                  'hosting_paise_per_1000': 10, 'tax_basis': 'inclusive'})
        self.write('price.json', {'name': 'API', 'date': '2026-11-17', 'source_url': 'https://example.com/prices',
            'input_paise_per_million': '10', 'output_paise_per_million': '20', 'tax_basis': 'inclusive',
            'source_currency': 'INR'})
        self.client = TestClient(self.app)
        self.url = '/v1/models/model1/savings-report'

    def write(self, filename, obj):
        (self.root / filename).write_text(json.dumps(obj), encoding='utf-8')

    def test_authorized_report_links_to_exact_evidence(self):
        auth = {'authorization': f'Bearer {self.key}'}
        response = self.client.get(self.url, headers=auth)
        self.assertEqual(response.status_code, 200, response.text)
        report = response.json()
        self.assertEqual(report['model_id'], 'model1')
        self.assertEqual(len(report['sources']), 4)
        for url in report['sources']:
            evidence = self.client.get(url, headers=auth)
            self.assertEqual(evidence.status_code, 200)
            self.assertEqual(hashlib.sha256(evidence.content).hexdigest(), report['source_sha256'][url])
        self.assertNotIn('api.json', response.text)

    def test_access_scope_and_missing_or_invalid_evidence(self):
        self.assertEqual(self.client.get(self.url).status_code, 401)
        self.assertEqual(self.client.get(self.url, cookies={'ekido_session': self.other_token}).status_code, 404)
        self.assertEqual(self.client.get(self.url, headers={'authorization': f'Bearer {self.no_scope}'}).status_code, 403)
        auth = {'authorization': f'Bearer {self.key}'}
        self.assertEqual(self.client.get(self.url + '/evidence/../../price', headers=auth).status_code, 404)
        self.write('model.json', {'job_id': 'other'})
        self.assertEqual(self.client.get(self.url, headers=auth).status_code, 503)

    def test_incomparable_quality_suppresses_savings_but_keeps_costs(self):
        self.write('model.json', {'job_id': 'job1', 'model_id': 'model1', 'eval_set_hash': 'a' * 64,
                                 'examples': 5, 'metric': 'accuracy', 'score': .5})
        response = self.client.get(self.url, headers={'authorization': f'Bearer {self.key}'})
        self.assertEqual(response.status_code, 200, response.text)
        report = response.json()
        self.assertFalse(report['quality']['comparable'])
        self.assertIsNone(report['first_month_difference_paise'])
        self.assertIsNone(report['later_month_difference_paise'])
        self.assertEqual(report['ekido']['training_paise'], 20)
        self.assertEqual(len(report['sources']), 4)

    def test_signed_in_owner_and_unconfigured_store(self):
        self.assertEqual(self.client.get(self.url, cookies={'ekido_session': self.token}).status_code, 200)
        app = create_app(Settings(database_url='sqlite://', _env_file=None))
        # Authentication is checked even when reports are disabled.
        self.assertEqual(TestClient(app).get(self.url).status_code, 401)


if __name__ == '__main__':
    unittest.main()
