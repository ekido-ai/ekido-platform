# Savings calculation evidence

`ekido_platform.savings.build_savings_report` calculates a report from four JSON
files supplied by a trusted, customer-authorised artifact store. It is a standalone
calculation and is not yet connected to the public savings-report endpoint. Do not
accept arbitrary customer-selected filesystem paths or publish a savings claim until
the artifacts have been authenticated and tied to the customer and job.

All file names passed to the function are relative to its evidence root. The result
contains their relative paths and SHA-256 hashes, so the bytes used can be audited.
The caller supplies expected monthly requests as an assumption.

| Evidence | Required fields |
| --- | --- |
| API run | `job_id`, `eval_set_hash`, `examples`, `metric`, `score`, measured `requests`, `input_tokens`, `output_tokens` |
| Ekido run | `job_id`, `model_id`, `eval_set_hash`, `examples`, `metric`, `score` |
| Quote | `job_id`, `model_id`, `training_paise`, `hosting_paise_per_1000`, `tax_basis` |
| API price | `name`, `source_url` (HTTPS), `date` (ISO date), `input_paise_per_million`, `output_paise_per_million`, `tax_basis`, `source_currency` |

Prices per million tokens use decimal strings or integers in paise. Both artifacts
must use the same `tax_basis` (`inclusive` or `exclusive`) so that the comparison
does not mix tax-inclusive and tax-exclusive amounts. When `source_currency` is
not `INR`, the price artifact must also include a `conversion` object with the
positive `inr_per_unit`, its HTTPS `source_url` and ISO `date`. Rates in paise are
still supplied by the trusted artifact producer. Run scores are
normalised to the interval from zero to one. The calculation refuses mismatched job,
test-set hash, example count, metric or quote model. Input and output token usage
comes from the measured API run. Positive paise amounts are rounded up to whole paise.
The first-month difference includes the one-time training cost; the later-month
difference does not. A negative difference represents a higher Ekido cost. Self-hosting
sets Ekido hosting to zero; it does not measure the customer's own infrastructure cost.

These checks validate and identify the supplied files; they do not independently prove
that measurements or prices inside them are genuine. Artifact provenance, date freshness
and access control belong to the integration that obtains the files.
