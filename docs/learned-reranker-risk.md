# Learned reranker controls and residual risk

## Production boundary and modes

`RankingService` and `build_ranking_service` remain the public ranking
interface. Learned ranking is an optional layer after the deterministic rule
ranking; it does not replace that interface.

`RANKING_MODEL_MODE` has three bounded values:

- `off` is the production default. It executes the rule baseline exactly and
  does not load a checkpoint.
- `shadow` runs learned scoring but returns the rule order, ranks, scores,
  score breakdowns, and candidate payload unchanged. It adds diagnostics to
  `debug_meta`; `ranking_changed` describes the counterfactual rerank and may
  therefore be true.
- `rerank` is explicit opt-in. It blends learned and rule scores only for the
  configured prefix of each already rule-truncated candidate group. Loading or
  inference failure returns the complete rule result with a diagnostic reason;
  changes are applied transactionally only after all groups score successfully.

The default corrected-checkpoint path is not populated and the default digest
is empty. No corrected checkpoint is shipped as a production default, so
enabling `shadow` or `rerank` also requires an explicit artifact path and
SHA-256. The retained `best.pt` is the known legacy artifact, not that default.

## Artifact and execution controls

The production loader reads a bounded file once, verifies its configured
SHA-256, and deserializes those exact verified in-memory bytes on CPU with
`weights_only=True`. It then validates the two-key schema, strict bounded
metadata, tensor count/size/type/shape/device/finite values, model version, and
exact state-dict keys before constructing an evaluation-mode model. Backend and
scripts production/operational Python code has one centralized `torch.load`
call and no other audited deserialization entry point.

A conservative, order-independent static audit covers ordinary imports,
wildcard imports, simple name/module/object-attribute aliases, literal and
dynamic `getattr`, functions, and control-flow syntax. It rejects alternate
entry points including `torch.serialization.load`, `torch.jit.load`,
`torch.package.PackageImporter`, `torch.hub.load_state_dict_from_url`,
`pickle.load`/`loads`/`Unpickler`, and `joblib.load`. This source audit does not
claim to interpret arbitrary `eval`, monkey-patching, or runtime metaprogramming;
code using such mechanisms requires separate review and must not create model
loading paths.

The known legacy digest is rejected by the production loader before
deserialization. Legacy inspection also identifies only that digest and
returns immutable recorded facts without deserializing `best.pt`; unknown or
modified legacy files are rejected.

Inference uses a bounded wait and a daemon worker plus a per-reranker circuit
breaker. Python threads cannot forcibly cancel a blocked native/PyTorch call:
after timeout the request falls back, but the daemon worker may continue to
consume resources until the call returns, and further calls fall back while it
remains active. Process isolation would be required for hard cancellation.

Training publishes one fixed checkpoint filename and one fixed JSON metadata
sidecar via staged files, replacement, backups, and rollback for caught write
failures. Two filenames cannot be atomically replaced as a pair across a
process or machine crash, so a crash window remains. The checkpoint's embedded
metadata is authoritative; the sidecar is an audit mirror and must not be used
as the loading trust source.

## Legacy inventory

- Model version: `v0.1.0-synthetic`
- Retained checkpoint: `backend/app/services/ranking/checkpoints/best.pt`
- SHA-256: `93850b0e5569b09230f8b9fd6d70efd4db7b1d99bf1551b08dd6979ed43a84fb`
- Classification: compatibility/smoke-only; never production recommendations

The five per-epoch checkpoints are intentionally excluded. The legacy model,
training package, and script that produced the incorrect integration are also
excluded; only the corrected implementation and explicit training entry point
remain. Hotel-scraper and frontend lockfile changes are outside this work.

## Unresolved effectiveness and operational risks

1. There is no real user feedback dataset. The retained legacy model learned
   only synthetic labels derived from rule ranking, so it provides no
   independent evidence of recommendation quality.
2. User-history quality and coverage may be insufficient for stable sequence
   features. Missing, sparse, stale, or mismatched POI identifiers can reduce
   the learned signal without representing user intent.
3. Synthetic smoke validation/test metrics prove only deterministic tensor,
   training, serialization, loading, and inference integration. They are not
   efficacy, fairness, calibration, or generalization evidence.
4. Effectiveness must be re-evaluated on representative shadow traffic and
   agreed offline/online quality and fairness criteria before any `rerank`
   rollout. Shadow diagnostics need monitoring for coverage, fallback rates,
   latency, drift, and counterfactual changes.
5. Ownership and procedures for model-artifact release, signing and
   distribution, version compatibility, retention, promotion, and rollback
   remain to be defined. SHA-256 allowlisting verifies configured bytes but is
   not by itself a complete release-governance or signing system.
6. PyTorch adds packaging, startup, platform, and resource costs. The loader's
   resource checks reduce risk but `weights_only=True` is not a complete sandbox
   for arbitrary hostile input; only trusted, release-approved digests belong
   in configuration.

Until these issues are resolved, production remains `off`; `shadow` is the
evaluation boundary, and `rerank` must not be treated as rollout-ready.
