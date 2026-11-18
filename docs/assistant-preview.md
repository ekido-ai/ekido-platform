# Assistant job previews

This branch requires the shared model table in
[ekido-core #76](https://github.com/ekido-ai/ekido-core/pull/76), pinned to
commit `4655af022b405bcd6066d125803ecafbd487f4f7` in CI. Deployments must
install a core version containing that commit before enabling the assistant.
The policy-screening categories follow the draft acceptable-use page in
[ekido-web #21](https://github.com/ekido-ai/ekido-web/pull/21); that page
itself is pending policy and legal review.

`POST /v1/assistant/messages` is disabled until the operator configures a local
Qwen2.5-1.5B-Instruct llama.cpp engine using `ASSISTANT_ENGINE_URL` and a secret
of at least 32 characters using `ASSISTANT_STATE_SECRET`. The engine must use a
numeric loopback HTTP origin and must be isolated from untrusted local users.
Use a protected operator-managed `ASSISTANT_DATASETS_ROOT` for dataset manifests.
Do not commit secrets, manifests containing customer data, or model prompts.

Authenticated users can submit `message`, optionally `conversation_id`,
`dataset_id` and a permitted `base_model`. The assistant first requests a
schema-constrained policy decision, and refuses or asks for review on anything
other than `allow`. It then requests a schema-constrained recipe classification.
Model responses are parsed and checked again on the server. Generated free-form
reply text and model-provided dataset hashes or training parameters are never
trusted as execution inputs. Conversations use short-lived signed tokens bound
to the calling person or API key; they contain the chosen recipe and language,
but not the user's message.

A dataset manifest `<dataset_id>.json` is created by a trusted artifact process,
not by a customer. It contains `org_id`, `recipe_id`, `sha256`, `bytes`,
`train_rows`, and `test_rows`. Its folder must not be customer-writable. The
server checks the caller's team or scoped key before loading the dataset into a
preview. It builds a spec from pinned, allow-listed models and server defaults,
then runs `spec_problems`. These defaults are a starting plan, not a guarantee
of accuracy, cost or successful training. No training job is started here.

Policy screening is performed by a small language model and can miss or wrongly
refuse requests. A preview has `policy_status: model_screened_needs_review`;
it is **not** policy approval. A separate independent policy gate and human
approval are required before a quote or job can use it. The 50-prompt evaluation
set is not included here; measure both valid-spec rate and each refusal before
enabling this endpoint for real users. The data-upload and artifact process must
also supply authenticated manifests before dataset previews can be used.
