# Savings calculation evidence

`ekido_platform.savings.build_savings_report` calculates a report from four JSON
files supplied by a trusted, customer-authorised artifact store. The platform can
serve these through the authenticated savings-report endpoint when the operator
configures `SAVINGS_EVIDENCE_ROOT`. Leave it unset to disable the endpoint.

The root is a trusted operator-managed directory, never a customer-writable upload
directory. Place an operator-verified manifest in `models/<model_id>.json`, with
`job_id`, `expected_requests`, `api_run_file`, `model_run_file`, `quote_file` and
`price_file`, and optional boolean `self_hosted`. File fields are paths relative to
the root. The platform checks the manifest job against its existing `job_owners`
record, the caller's membership or `models.read` API key, and the model and quote
evidence. The report links to authenticated evidence endpoints that return the exact
bytes identified by SHA-256. The operator must establish provenance and keep the
manifest and evidence protected from unauthorized writes. This integration does not
create measurements or provision a production artifact store.

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
`quality.comparable` is true only when Ekido's score is at least the API's score minus 0.02 (the default non-inferiority margin); show savings to a customer only when it is true, because savings from a worse model are not savings. The first-month difference includes the one-time training cost; the later-month
difference does not. A negative difference represents a higher Ekido cost. Self-hosting
sets Ekido hosting to zero; it does not measure the customer's own infrastructure cost.

These checks validate and identify the supplied files; they do not independently prove
that measurements or prices inside them are genuine. Artifact provenance, date freshness
and access control belong to the integration that obtains the files.
