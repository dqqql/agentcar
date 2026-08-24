# Learned reranker integration boundary

## Current production boundary

The supported ranking implementation remains the deterministic rule-based
`RankingService` and its existing `build_ranking_service` factory. The pipeline
continues to consume that public interface. Learned ranking code must not
replace or rename either interface until it has passed the integration gates
below.

The learned-reranker material is retained only as reference material for a
future, selective migration. Its source snapshot is merge commit `df43570`.
The `model/` and `training/` packages should be reviewed and migrated in small,
testable units rather than enabled through the current service entry point.

## Legacy artifact inventory

- Model version: `v0.1.0-synthetic`
- Retained checkpoint: `backend/app/services/ranking/checkpoints/best.pt`
- SHA-256: `93850B0E5569B09230F8B9FD6D70EFD4DB7B1D99BF1551B08DD6979ED43A84FB`
- Associated vocabulary and training metadata may be retained to reproduce
  compatibility checks.

The old checkpoint was trained on synthetic data. It is for compatibility and
smoke testing only and must never be used to produce production
recommendations. The five duplicate per-epoch checkpoints are intentionally
not retained.

## Known risks

1. **Quality and generalization:** synthetic ranking metrics do not establish
   relevance, fairness, or stability on real users and real candidate pools.
2. **Contract compatibility:** the proposed service changed the public class
   name and ranking behavior, breaking imports, tests, and the pipeline
   contract.
3. **Dependency footprint:** NumPy, Scikit-learn, and PyTorch introduce runtime,
   packaging, platform, and startup costs that are not part of the reliable
   baseline.
4. **Checkpoint safety:** PyTorch checkpoints can contain unsafe serialized
   objects. Only provenance-verified artifacts with a pinned digest may be
   loaded, using the safest supported loading mode; never load user-supplied or
   otherwise untrusted checkpoints.
5. **Fallback ambiguity:** silently falling back between learned and rule-based
   ranking can hide deployment faults and make recommendation behavior
   difficult to audit.
6. **Operational readiness:** the snapshot has no production evaluation,
   calibrated rollout controls, model monitoring, drift detection, or rollback
   evidence.
7. **Unrelated changes:** the PR also touched hotel data collection and the
   frontend lockfile. Those changes are outside the learned-reranker boundary
   and are excluded.

## Gates for a future migration

A future integration must preserve `RankingService` and
`build_ranking_service`, add learned behavior behind an explicit configuration
boundary, and keep the deterministic implementation as a separately tested
rollback path. It must also:

- add contract, unit, artifact-integrity, and end-to-end pipeline tests;
- evaluate against representative held-out data with agreed quality and
  fairness thresholds;
- document data provenance, model versioning, dependency locks, and supported
  platforms;
- fail closed on missing, corrupt, or digest-mismatched artifacts and expose
  useful diagnostics;
- define opt-in rollout, observability, and rollback procedures before serving
  any production traffic.
