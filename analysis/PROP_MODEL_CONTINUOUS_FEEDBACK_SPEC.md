# Player Prop Model: Continuous Feedback + Deployment Spec

## Goal
Ship a production-safe feedback loop for player performance predictions using:
- immutable decision-time logging,
- leakage-safe retraining,
- calibrated probabilities,
- portfolio/risk controls,
- explicit deploy gates.

This spec is aligned to the active phase plan and verification checks already in progress.

## Scope and Constraints
- Keep current architecture centered in `scripts/predict_player_props.py` (monolith-first, refactor later).
- Prefer incremental flags over breaking CLI behavior.
- Any model or policy layer must be evaluated by walk-forward only.
- No phase is considered done without its verification check.

## Current Baseline (as of this spec)
- Canonical history + grading exists (`prop_results_history.csv`).
- Core two-stage + stage-3 residual path exists.
- Market residual/calibration layers exist with coverage thresholds.
- Injury and confirmed starter ingestion exists.
- Signal caps/lineup lock exist.

## Phase Plan (aligned to your checks)

## Phase 1: OOF Audit (No behavior change)
### Objective
Prove every second-stage/residual/calibration path uses leakage-safe inputs.

### Implementation
- Add audit function: `audit_oof_integrity(...)` in `scripts/predict_player_props.py`.
- Emit an OOF report file:
  - `analysis/output/prop_logs/oof_audit_<YYYYMMDD>.json`
- Checks:
  - training rows are not scored by models fitted on same rows,
  - tiny folds are skipped with explicit counts,
  - no in-sample base predictions feed residual models.

### CLI
- `--audit-oof`

### Verification
- `--walk-forward`
- Metrics should be unchanged (except tiny-fold skip side effects).

## Phase 2: Portfolio Controls
### Objective
Reduce concentrated risk and correlated exposure.

### Implementation
- Enforce caps before final signal selection:
  - max signals per game,
  - max signals per team,
  - max 1 correlated same-direction signal per player.
- Add columns to output:
  - `portfolio_blocked_reason`,
  - `correlation_group_id`.

### CLI
- `--portfolio-max-per-game <int>`
- `--portfolio-max-per-team <int>`
- `--portfolio-max-correlated-per-player <int>`

### Verification
- `--date 20260302`
- Confirm game/team caps apply.
- Confirm same-player correlated same-direction signals are reduced to 1.

## Phase 3: Minutes Foundation
### Objective
Make minutes model the primary driver of points/rebounds/assists.

### Implementation
- Expand minutes features:
  - `dnp_prob` (from injury status),
  - `starter_confirmed`,
  - `blowout_risk`,
  - `foul_trouble_proxy`,
  - `minutes_volatility_rolling`.
- Keep stage-2 stat models strictly conditional on predicted minutes.
- Persist minutes diagnostics to:
  - `analysis/output/prop_logs/minutes_model_metrics.csv`

### CLI
- `--minutes-model-v2`

### Verification
- `--walk-forward`
- New minutes features appear.
- Minutes MAE improves or remains flat.

## Phase 4: Player-Stat Uncertainty
### Objective
Replace global residual std with predicted uncertainty per row.

### Implementation
- Add uncertainty model per stat:
  - target: `abs(actual - pred)` or quantile deltas.
- Store predicted std in edge computation:
  - replace `residual_std` fallback when model available.
- Write diagnostics:
  - `analysis/output/prop_logs/uncertainty_metrics.csv`

### CLI
- `--uncertainty-model`

### Verification
- `--walk-forward`
- Uncertainty model trains.
- Predicted std is used in edge computation in place of global std.

## Phase 5: Per-Side Calibration
### Objective
Calibrate `OVER` and `UNDER` separately per stat.

### Implementation
- Train calibrators by `(stat_type, side)` from OOF probabilities only.
- Apply calibration + renormalize:
  - enforce `p_over + p_under ~= 1.0`.
- Save calibration params:
  - `analysis/output/models/prop_calibrators_<version>.joblib`

### CLI
- `--calibration-per-side`

### Verification
- `--walk-forward`
- Over/under calibrators produce different corrections.
- `p_over + p_under` approximately 1.0 after normalization.

## Phase 6: Feature Group Ablation Harness
### Objective
Keep only feature groups that improve actionable OOS outcomes.

### Implementation
- Add grouped ablation runner:
  - groups: market, injury/lineup, minutes, matchup, recency, uncertainty-derived.
- Report paired comparisons + significance test summary.
- Output:
  - `analysis/output/prop_logs/ablation_features_<YYYYMMDD>.csv`

### CLI
- `--ablation-features`

### Verification
- `--ablation-features`
- All feature groups tested.
- Paired comparison report with significance appears.

## Phase 7: Recency-Weighted Training
### Objective
Handle role drift due to injuries/trades/coaching changes.

### Implementation
- Add sample-weight schedules (half-life based) by stat.
- Version the feature cache when weighting changes.
- Output:
  - `analysis/output/prop_logs/recency_weight_metrics.csv`

### CLI
- `--recency-half-life-days <int>`
- `--recency-max-weight <float>`

### Verification
- Cache version bump is visible.
- `--walk-forward` confirms no degradation.

## Phase 8: Matchup Delta Features
### Objective
Add opponent-vs-player profile deltas with controlled complexity.

### Implementation
- Compute and add deltas:
  - player rolling production minus opponent allowed-to-position rolling metrics.
- Ensure all deltas are pregame-safe and shifted.
- Include features in prediction output for auditability.

### CLI
- `--enable-matchup-deltas`

### Verification
- Delta features appear in predictions CSV.
- `--walk-forward` reports impact.

## Phase 9: Monitoring and Daily Report
### Objective
Make model health and data quality observable every day.

### Implementation
- Daily monitoring report includes:
  - hit rate/Accuracy/MES by stat and side,
  - calibration drift,
  - feature missingness by column,
  - coverage vs thresholds.
- Persist to:
  - `analysis/output/prop_logs/daily_metrics.csv`
  - `analysis/output/prop_logs/feature_missingness_daily.csv`

### CLI
- `--daily-report`

### Verification
- `--daily-report`
- Missingness report prints.
- `daily_metrics.csv` is written.

## Phase 10: Deploy Gates
### Objective
Define pass/fail criteria for paper -> production promotion.

### Implementation
- Add deploy-status checker with explicit gates:
  - min matched market rows total/per-stat,
  - min settled actionable bets,
  - positive market efficiency score,
  - calibration drift below threshold,
  - no severe missingness.
- Persist status:
  - `analysis/output/prop_logs/deploy_status_<YYYYMMDD>.json`

### CLI
- `--deploy-status`

### Verification
- `--deploy-status`
- Pass/fail report prints each gate result.

## Continuous Feedback Architecture (non-RL and RL-lite)

## A. Required event logging (immutable)
Add/ensure fields in canonical rows:
- `model_version`, `feature_version`, `policy_version`,
- `decision_ts_utc`,
- `action` (`OVER|UNDER|NO_BET`),
- `action_propensity` (for policy learning),
- `available_actions`,
- executable price fields used for EV.

## B. Daily batch loop
1. ingest lines and predictions (decision-time write),
2. grade settled outcomes,
3. generate daily monitoring report,
4. update calibration and coverage health.

## C. Weekly retrain loop
1. rebuild training frame from canonical history,
2. retrain candidate models with strict OOF,
3. walk-forward benchmark vs champion,
4. promote only on gate pass.

## D. Policy learning (recommended: contextual bandit, not full RL)
- Start with rule policy + epsilon exploration in paper mode.
- Learn action selection from logged outcomes and propensities.
- Offline evaluation with IPS/DR before any promotion.

## Suggested new file split (minimal)
- `scripts/predict_player_props.py`:
  - keep main orchestration + CLI.
- `scripts/prop_feedback.py` (new):
  - training frame builder,
  - OOF audit,
  - deploy gate checker.
- `scripts/prop_policy.py` (new):
  - policy scoring,
  - propensity logging,
  - offline IPS/DR evaluator.

## Cron / automation updates
Use existing daily script and add:
1. post-game grading run,
2. daily report run,
3. weekly retrain + deploy-status run.

Example cadence:
- daily 03:00 ET: grading + daily report,
- weekly Mon 08:00 ET: retrain + deploy-status.

## Definition of Done
- All 10 phase verification checks pass.
- 500+ matched market rows total and >=200 per major stat bucket.
- Multi-week actionable walk-forward shows stable calibration and positive market efficiency score.
- Deploy status returns pass without manual overrides.
