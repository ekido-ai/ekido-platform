import json
import tempfile
import unittest
from pathlib import Path

from ekido_platform.savings import SavingsError, build_savings_report


class SavingsReport(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.files = {
            'api.json': {'job_id': 'job', 'eval_set_hash': 'a' * 64, 'examples': 100,
                         'metric': 'accuracy', 'score': 0.81, 'requests': 10,
                         'input_tokens': 1000, 'output_tokens': 200},
            'model.json': {'job_id': 'job', 'model_id': 'model', 'eval_set_hash': 'a' * 64,
                           'examples': 100, 'metric': 'accuracy', 'score': 0.86},
            'quote.json': {'job_id': 'job', 'model_id': 'model', 'training_paise': 3000,
                           'hosting_paise_per_1000': 50, 'tax_basis': 'inclusive'},
            'price.json': {'name': 'Example API', 'date': '2026-11-17',
                           'source_url': 'https://example.org/pricing',
                           'input_paise_per_million': '100000', 'output_paise_per_million': '200000',
                           'tax_basis': 'inclusive', 'source_currency': 'INR'},
        }
        self.write_files()

    def write_files(self):
        for name, obj in self.files.items():
            (self.root / name).write_text(json.dumps(obj), encoding='utf-8')

    def report(self, **changes):
        args = dict(evidence_root=self.root, model_id='model', expected_requests=10000,
                    api_run_file='api.json', model_run_file='model.json', quote_file='quote.json',
                    price_file='price.json')
        return build_savings_report(**{**args, **changes})

    def test_measured_costs_keep_training_and_monthly_costs_separate(self):
        result = self.report()
        self.assertEqual(result['api_baseline']['monthly_paise'], 140000)
        self.assertEqual(result['ekido'], {'training_paise': 3000, 'hosting_monthly_paise': 500})
        self.assertEqual(result['first_month_difference_paise'], 136500)
        self.assertEqual(result['later_month_difference_paise'], 139500)
        self.assertEqual(set(result['sources']), set(self.files))
        self.assertTrue(all(len(digest) == 64 for digest in result['source_sha256'].values()))
        self.assertEqual(result['quality']['api_score'], 0.81)
        self.assertEqual(self.report(self_hosted=True)['ekido']['hosting_monthly_paise'], 0)

    def test_different_or_missing_task_evidence_is_refused(self):
        for name, field, value in [('api.json', 'eval_set_hash', 'b' * 64),
                                   ('api.json', 'examples', 99), ('api.json', 'metric', 'rouge_l'),
                                   ('api.json', 'job_id', 'other'), ('quote.json', 'model_id', 'other'),
                                   ('api.json', 'requests', 0), ('model.json', 'score', None)]:
            with self.subTest(field=field):
                old = self.files[name][field]
                self.files[name][field] = value
                self.write_files()
                with self.assertRaises(SavingsError):
                    self.report()
                self.files[name][field] = old
        self.write_files()
        with self.assertRaises(SavingsError):
            self.report(api_run_file='../outside.json')
        with self.assertRaises(SavingsError):
            self.report(api_run_file='missing.json')

    def test_fractional_paise_round_up_and_negative_difference_is_preserved(self):
        self.files['api.json'].update(requests=1, input_tokens=1, output_tokens=0)
        self.files['price.json'].update(input_paise_per_million='1', output_paise_per_million='0')
        self.write_files()
        result = self.report(expected_requests=1)
        self.assertEqual(result['api_baseline']['monthly_paise'], 1)
        self.assertEqual(result['ekido']['hosting_monthly_paise'], 1)
        self.assertEqual(result['first_month_difference_paise'], -3000)

    def test_large_inputs_keep_exact_paise_when_global_precision_is_low(self):
        from decimal import localcontext
        self.files['api.json'].update(requests=1, input_tokens=10**15, output_tokens=1)
        self.files['price.json'].update(input_paise_per_million='1000000000000000',
                                        output_paise_per_million='0.000001')
        self.write_files()
        with localcontext() as context:
            context.prec = 6
            result = self.report(expected_requests=10**15)
        self.assertEqual(result['api_baseline']['monthly_paise'], 10**39 + 1000)

    def test_prices_and_assumptions_are_checked(self):
        for field, value in [('input_paise_per_million', -1), ('output_paise_per_million', 'NaN'),
                             ('source_url', 'http://insecure.example'), ('date', 'yesterday'),
                             ('tax_basis', 'exclusive')]:
            with self.subTest(field=field):
                old = self.files['price.json'][field]
                self.files['price.json'][field] = value
                self.write_files()
                with self.assertRaises(SavingsError):
                    self.report()
                self.files['price.json'][field] = old
        self.write_files()
        with self.assertRaises(SavingsError):
            self.report(expected_requests=True)
        with self.assertRaises(SavingsError):
            self.report(expected_requests=-1)

    def test_converted_prices_must_record_exchange_rate_source(self):
        self.files['price.json']['source_currency'] = 'USD'
        self.write_files()
        with self.assertRaises(SavingsError):
            self.report()
        self.files['price.json']['conversion'] = {'inr_per_unit': '94.1', 'date': '2026-11-17',
                                                   'source_url': 'https://example.org/rates'}
        self.write_files()
        self.assertEqual(self.report()['api_baseline']['conversion']['inr_per_unit'], '94.1')


if __name__ == '__main__':
    unittest.main()
