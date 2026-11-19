"""Build a savings report from measured, local evidence, not estimated quality.

This calculation is not an HTTP endpoint. Its caller must obtain the files from a
trusted, customer-authorised artifact store. Amounts are paise; API token prices
are paise per million tokens and hosting is paise per thousand requests.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import date
from decimal import Decimal, InvalidOperation, ROUND_CEILING, localcontext
from pathlib import Path
from urllib.parse import urlparse


# How far below the API's score Ekido's may be and still count as doing the same job. It matches the default
# non-inferiority margin in the job quality policy (0.02).
QUALITY_MARGIN = Decimal('0.02')


class SavingsError(ValueError):
    pass


def _int(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value <= 10**15:
        raise SavingsError(f'{label} must be an integer between {minimum} and 10^15')
    return value


def _amount(value: object, label: str) -> Decimal:
    if isinstance(value, (bool, float)):
        raise SavingsError(f'{label} must use an exact decimal string or integer')
    try:
        amount = Decimal(value)
    except (InvalidOperation, TypeError, ValueError):
        raise SavingsError(f'{label} is not a decimal amount') from None
    if not amount.is_finite() or amount < 0 or amount > 10**15 or amount.as_tuple().exponent < -30:
        raise SavingsError(f'{label} must be finite, non-negative and within supported precision')
    return amount


def _score(value: object, label: str) -> Decimal:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SavingsError(f'{label} must be finite')
        value = str(value)
    score = _amount(value, label)
    if score > 1:
        raise SavingsError(f'{label} must be between zero and one')
    return score


def _evidence(root: Path, filename: str) -> tuple[dict, str, str]:
    root = root.resolve()
    if not isinstance(filename, str) or not filename or Path(filename).is_absolute():
        raise SavingsError('evidence must be a relative JSON path')
    path = (root / filename).resolve()
    try:
        relative = path.relative_to(root)
    except ValueError:
        raise SavingsError('evidence is outside the allowed directory') from None
    if path.suffix.lower() != '.json' or not path.is_file():
        raise SavingsError('evidence must be an existing JSON file')
    with path.open('rb') as handle:
        data = handle.read(1_000_001)
    if len(data) > 1_000_000:
        raise SavingsError('evidence file is too large')
    try:
        obj = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise SavingsError('evidence is not valid JSON') from None
    if not isinstance(obj, dict):
        raise SavingsError('evidence must contain a JSON object')
    return obj, relative.as_posix(), hashlib.sha256(data).hexdigest()


def build_savings_report(*, evidence_root: Path, model_id: str, expected_requests: int,
                         api_run_file: str, model_run_file: str, quote_file: str,
                         price_file: str, self_hosted: bool = False) -> dict:
    """Compare a measured API run and model run on the same customer's test set.

    Files are loaded beneath evidence_root; the returned hashes identify the exact
    bytes used. The request volume is an explicit assumption, not a measurement.
    """
    _int(expected_requests, 'expected_requests')
    if not isinstance(model_id, str) or not model_id.strip() or type(self_hosted) is not bool:
        raise SavingsError('model_id and self_hosted are invalid')
    api, api_path, api_hash = _evidence(evidence_root, api_run_file)
    model, model_path, model_hash = _evidence(evidence_root, model_run_file)
    quote, quote_path, quote_hash = _evidence(evidence_root, quote_file)
    price, price_path, price_hash = _evidence(evidence_root, price_file)
    try:
        test_hash = model['eval_set_hash']
        if (not isinstance(test_hash, str) or not re.fullmatch(r'[0-9a-f]{64}', test_hash)
                or api['eval_set_hash'] != test_hash or not isinstance(model['job_id'], str) or not model['job_id'].strip()
                or api['job_id'] != model['job_id'] or model['model_id'] != model_id
                or not isinstance(model['metric'], str) or not model['metric'].strip() or api['metric'] != model['metric']):
            raise SavingsError('runs must describe the same job, metric and test set')
        examples = _int(model['examples'], 'model examples', minimum=1)
        if _int(api['examples'], 'API examples', minimum=1) != examples:
            raise SavingsError('runs must use the same number of test examples')
        model_score = _score(model['score'], 'model score')
        api_score = _score(api['score'], 'API score')
        request_count = _int(api['requests'], 'measured API requests', minimum=1)
        input_tokens = _int(api['input_tokens'], 'measured input tokens')
        output_tokens = _int(api['output_tokens'], 'measured output tokens')
        input_rate = _amount(price['input_paise_per_million'], 'input rate')
        output_rate = _amount(price['output_paise_per_million'], 'output rate')
        training = _int(quote['training_paise'], 'training cost')
        hosting_rate = _int(quote['hosting_paise_per_1000'], 'hosting rate')
        if price['tax_basis'] not in ('inclusive', 'exclusive') or quote['tax_basis'] != price['tax_basis']:
            raise SavingsError('API and Ekido amounts must use the same tax basis')
        if quote['model_id'] != model_id or quote['job_id'] != model['job_id']:
            raise SavingsError('quote does not match the measured job and model')
        source_url = price['source_url']
        if not isinstance(source_url, str) or urlparse(source_url).scheme != 'https' or not urlparse(source_url).netloc:
            raise SavingsError('API price must have an HTTPS source URL')
        price_date = date.fromisoformat(price['date']).isoformat()
        if not isinstance(price['name'], str) or not price['name'].strip():
            raise SavingsError('API price must name its provider')
        source_currency = price['source_currency']
        if not isinstance(source_currency, str) or not re.fullmatch(r'[A-Z]{3}', source_currency):
            raise SavingsError('API price must state its original currency')
        conversion = None
        if source_currency != 'INR':
            conversion = price['conversion']
            fx_url = conversion['source_url']
            if not isinstance(fx_url, str) or urlparse(fx_url).scheme != 'https' or not urlparse(fx_url).netloc:
                raise SavingsError('currency conversion must have an HTTPS source URL')
            fx_rate = _amount(conversion['inr_per_unit'], 'currency conversion rate')
            if fx_rate <= 0:
                raise SavingsError('currency conversion rate must be positive')
            conversion = {'source_url': fx_url, 'date': date.fromisoformat(conversion['date']).isoformat(),
                          'inr_per_unit': str(fx_rate)}
    except KeyError as exc:
        raise SavingsError(f'missing evidence field: {exc.args[0]}') from None
    except (TypeError, ValueError) as exc:
        if isinstance(exc, SavingsError):
            raise
        raise SavingsError('invalid evidence field') from None

    with localcontext() as context:
        context.prec = 100
        api_monthly = ((Decimal(expected_requests) * (input_tokens * input_rate + output_tokens * output_rate))
                       / (request_count * 1_000_000))
        hosting = Decimal(0) if self_hosted else Decimal(expected_requests) * hosting_rate / 1000
        api_paise = int(api_monthly.to_integral_value(rounding=ROUND_CEILING))
        hosting_paise = int(hosting.to_integral_value(rounding=ROUND_CEILING))
    sources = {api_path: api_hash, model_path: model_hash, quote_path: quote_hash, price_path: price_hash}
    return {
        'model_id': model_id,
        'assumed_requests_per_month': expected_requests,
        'api_baseline': {'name': price['name'], 'monthly_paise': api_paise,
                         'price_source_url': source_url, 'price_date': price_date,
                         'source_currency': source_currency, 'conversion': conversion},
        'ekido': {'training_paise': training, 'hosting_monthly_paise': hosting_paise},
        'quality': {'ekido_score': float(model_score), 'api_score': float(api_score),
                    'metric': model['metric'], 'eval_set_hash': test_hash, 'examples': examples,
                    'difference': float(model_score - api_score), 'margin': float(QUALITY_MARGIN),
                    # Savings only mean something for a model that does the job about as well.
                    'comparable': model_score >= api_score - QUALITY_MARGIN},
        'first_month_difference_paise': api_paise - training - hosting_paise,
        'later_month_difference_paise': api_paise - hosting_paise,
        'sources': list(sources), 'source_sha256': sources,
        'evidence': {'api_run': api_path, 'model_run': model_path, 'quote': quote_path, 'api_price': price_path},
        'request_volume_source': 'customer assumption', 'self_hosted': self_hosted,
        'tax_basis': price['tax_basis'],
    }
