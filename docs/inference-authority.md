# Internal inference authorization and metering

The inference router must call the Platform's internal authority before sending
any customer request to a model engine. The authority is disabled by default.
Configure `INFERENCE_MODELS_ROOT` as a protected operator-managed directory and
`INFERENCE_INTERNAL_TOKEN` as a secret of at least 32 characters. Keep both
services private and use authenticated transport or a protected same-host
loopback channel. Never expose the internal token to customers or logs.

For each served model, an operator places `<model_id>.json` in the root. The
manifest has `job_id`, `engine_url`, and `engine_model`. The job must already
belong to the customer organization in `job_owners`. The engine URL must be an
HTTP origin on a numeric loopback address and a specified port. The directory
must not be writable by customers. Only the operator can assign model ownership
or an engine; these values must not come from request parameters.

Set `"mode": "test"` for a test model. Test keys can reserve only test
models; live keys can reserve only live models. An absent mode defaults to
live, and invalid modes fail closed. Operators must ensure test engines and
their upstream providers do not incur customer charges; billing is not yet
implemented by this service.

The internal `/internal/inference/ready` endpoint returns the loopback origins
of configured, job-owned models to an authenticated router. The router probes
engine health itself; merely having a manifest does not mean an engine is ready.

The router sends the customer's key and model ID to `/internal/inference/reserve`
with the internal token in `X-Ekido-Inference-Token`. The authority checks the
key hash, revocation, `models.predict` scope and job ownership. It reserves one
per-key request under a database row lock and returns an opaque reservation ID
and the trusted engine location. The default limit is 60 requests per minute;
`INFERENCE_RATE_PER_MINUTE` changes it. Use Postgres in deployment: the database
row lock is required for concurrent requests across processes.

After each reservation, the router reports `ok` with actual input/output token
counts, or `error`, and latency to `/internal/inference/usage/{reservation_id}`.
Reservations can be finalized once. Pending rows record incomplete requests
and must be reconciled after router failures; no prompt, response, or key is
stored in usage. Billing and automatic reconciliation are not implemented here.

This authority is only one part of the gateway: a router must still authenticate
its connection to this service, enforce a bounded request size and queue, call the
model engine, finalize all reservations, and expose the customer endpoints. Do
not expose a customer inference endpoint until that integration is verified.
