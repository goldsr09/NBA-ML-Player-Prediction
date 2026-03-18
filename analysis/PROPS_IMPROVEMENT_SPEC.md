# Player Performance Predictions Pipeline — Improvement Spec

**Scope**: `predict_player_props.py` (~9235 lines) and its supporting infrastructure.
**Date**: 2026-03-04
**Based on**: Full code review of main branch.

---

## Executive Summary

The props pipeline is already highly sophisticated — multi-stage modeling, OOF residual correction, quantile uncertainty, per-side calibration, market residual models, and a 10-phase feedback loop. The improvements below target the remaining gaps that would yield the highest marginal Accuracy, organized by category.

---

## 1. DATA QUALITY & INGESTION

### 1.1 ESPN Prop Line Coverage Gaps
**Current**: `fetch_espn_player_props()` (lines 1387-1563) relies on ESPN's summary endpoint with pagination. Lines are cached per-date in `prop_cache/prop_lines_YYYYMMDD.csv`.
**Problems**:
- ESPN rate-limits aggressively; the 2-second sleep between pages means a full slate can take 30+ minutes.
- No retry logic for partial fetches — if a game's page times out, that game's lines are silently missing.
- The ESPN athlete ID resolution (`_resolve_espn_athlete_id`) uses a dict-based cache hydrated from disk; lookup itself is fast, but the first-time API fallback per unknown player adds latency on large slates.
- Stat type coverage is limited to what ESPN surfaces (points, rebounds, assists, threes). Steals, blocks, turnovers, and combo props (PRA, PR, PA) are missing entirely.
- No deduplication across sources — if both ESPN and a manual CSV provide lines for the same player/stat, both are used.

**Recommendations**:
1. **Add The Odds API as a second source** — it covers all major books (market data provider, market data provider, market data provider) with consistent JSON. Store raw payloads in `prop_cache/odds_api/` keyed by date/book. Merge using `normalize_player_name()` + stat_type as join key. When both sources have the same prop, prefer the one with tighter vig.
2. **Exponential backoff retry on fetch failures** — wrap the ESPN pagination loop in a retry decorator (max 3 attempts, 5/15/45 second delays). Log partial fetches to `prop_logs/fetch_failures.csv` for monitoring.
3. **Batch pre-warm athlete ID cache** — on first daily run, pre-resolve all expected players for the slate in a single pass rather than one-by-one API fallback. Reduces first-run latency.
4. **Add combo props (PRA, P+R, P+A)** — these are the highest-volume prop markets. Model PRA as `pred_points + pred_rebounds + pred_assists` with a correlation-adjusted std (see §3.5).
5. **Source deduplication** — when merging manual CSV + ESPN + Odds API, keep exactly one line per `(player_name_norm, stat_type, date)` tuple, preferring the source with the lowest vig.

### 1.2 BoxScoreAdvancedV3 Fetch Reliability
**Current**: `load_boxscore_advanced_stats()` fetches from the NBA CDN with a fixed 1-second sleep and 5 retries.
**Problems**:
- The CDN intermittently returns 403s during heavy traffic; the fixed retry cadence doesn't adapt.
- `BOX_ADV_DEFAULT_TIMEOUT = 20s` is too generous — stale connections hang.
- No checksum/validation on the returned JSON; corrupt payloads are cached permanently.

**Recommendations**:
1. Exponential backoff (1s → 2s → 4s) with jitter.
2. Reduce timeout to 10s; add a content-length sanity check (reject payloads < 100 bytes).
3. Write a `_validate_box_adv_payload()` that checks for expected keys (`playerStats`, `teamStats`) before caching.

### 1.3 Historical Odds Backfill Quality
**Current**: `load_game_odds_lookup()` merges ESPN historical odds and current-season odds by game_id. Used to populate `implied_total`, `implied_spread`, `implied_team_total`.
**Problem**: When odds are missing for a game (ESPN didn't cover it, or pre-2022 data), the feature is NaN. XGBoost handles NaN natively, but the imputer fills with median, which blurs the signal for games where odds genuinely inform minutes/scoring context.
**Recommendation**: Replace median imputation for Vegas features with a learned default from team-level rolling stats (e.g., `implied_team_total ≈ team_pre_off_rating_avg5 * team_pre_possessions_avg5 / 100`). This is a better prior than the global median.

---

## 2. FEATURE ENGINEERING

### 2.1 Missing High-Signal Features

| Feature | Rationale | Estimated Impact |
|---------|-----------|-----------------|
| **Pace-adjusted per-minute rates** | Current `pts_per_min` doesn't account for team pace; a 0.7 pts/min on a 95-pace team ≠ 0.7 on a 110-pace team | Medium — improves rebounds, assists models |
| **Opponent 3PA/3P% defense** | Missing from `opp_positional_defense`. Opponents that allow high 3PA inflate fg3m | High for fg3m prop |
| **Player-vs-team historical matchup** | Rolling average of player performance against specific opponent (last 4 meetings) | Medium — captures stylistic matchups |
| **DFS ownership/salary as market sentiment** | High correlation with minutes and usage; available from market data provider | Medium |
| **Game script proxy** | Expected blowout direction × starter status. Currently `blowout_risk` is magnitude-only; direction matters for who plays Q4 | Medium for minutes model |
| **Second-half scoring share** | Some players are Q3/Q4 scorers; if a blowout is expected, their output drops disproportionately | Medium for points |
| **Travel distance** | Available in the monolith via `haversine_miles()` + `TEAM_COORDS` but not wired into player features | Low-Medium |

### 2.2 Feature Staleness / Anchoring Concern
**Current**: Season averages (`pre_X_season`) are deliberately excluded from `get_feature_list()` (line 3319 comment) to prevent anchoring to stale early-season production. Venue splits use expanding averages that are still prone to this.
**Recommendations**:
1. **Cap venue split lookback** — change from `expanding(min_periods=3).mean()` to `rolling(30, min_periods=3).mean()` so the venue signal tracks recent form.
2. **Add a `games_into_season` feature** — let the model learn to trust short-term vs long-term differently early vs late in the season.

### 2.3 Injury Feature Gaps
**Current**: Injury pressure is computed from the backward-looking `injury_proxy_missing_points5` (what was historically missing) and forward-looking `team_injury_pressure_fwd` (what's projected missing tonight).
**Problems**:
- `injury_pressure_delta` (fwd - bwd) is always 0.0 at training time because forward injury pressure can only be computed at prediction time. The model never sees nonzero values during training.
- No opponent injury features — if the opposing team's star defender is out, that's a positive signal for all players on the other side.
- No explicit "teammate X is out" feature beyond aggregate missing-points5. The identity of who is out matters (losing a 30 PPG scorer ≠ losing a 12 PPG role player, even if both average 32 minutes).

**Recommendations**:
1. **Simulate forward injury pressure in training** — for historical games, look up who was actually absent and compute what the forward pressure would have been pre-game. This gives the model real training signal on `injury_pressure_delta`.
2. **Add opponent injury features** — `opp_fwd_injury_missing_points`, `opp_fwd_injury_missing_minutes` at prediction time; at training time, reconstruct from actual absences.
3. **Top-teammate absence encoding** — for each player, track who their top 3 teammates are by usage/minutes. Create binary flags `top1_teammate_out`, `top2_teammate_out`, `top3_teammate_out`. This is more granular than aggregate pressure.

### 2.4 Lineup Confirmation Feature Asymmetry
**Current**: `confirmed_starter` and `lineup_confirmed` are NaN during training (line 2760-2761) and only populated at prediction time. The model never learns their signal during training.
**Recommendation**: For historical games, reconstruct `confirmed_starter` from actual boxscores (started = confirmed_starter=1, bench = 0). This lets the model learn the starter→minutes relationship from data rather than relying purely on the heuristic damping in `predict_two_stage()` (line 4426-4433).

### 2.5 Shrinkage/Role-Change Feature Interactions
**Current**: Phase 12 adds `pre_X_shrunk`, `role_change_flag`, `role_change_direction`, `usage_regime_shift`. The shrinkage is relaxed during role changes (line 2376-2382).
**Problem**: The role-change detection uses a 15% threshold on `avg5 - avg10` which is noisy — a single blowout can trigger it. No persistence tracking (was this change sustained for 3+ games?).
**Recommendation**: Add a `role_change_sustained_3g` flag that requires the minutes shift to persist for 3 consecutive games before triggering. This avoids noise from single-game anomalies.

---

## 3. MODEL ARCHITECTURE

### 3.1 Two-Stage Minutes Dependency Risk
**Current**: All stat predictions flow through `pred_minutes` (Stage 1 → Stage 2). A minutes prediction error propagates and amplifies to all stat predictions.
**Quantified risk**: If minutes MAE is 4 and a player scores 0.65 pts/min, the points error from minutes alone is ±2.6 points — often larger than the edge being signaled.
**Recommendations**:
1. **Blend two-stage and direct predictions** — train both `pred_points_two_stage = f(pred_minutes, ...)` and `pred_points_direct = g(pre_points_avg5, ...)`. Final prediction = `alpha * two_stage + (1 - alpha) * direct` where alpha is learned from OOF performance.
2. **Minutes prediction uncertainty propagation** — use the quantile uncertainty from the minutes model to widen the std used in stat prediction. Currently the stat-level uncertainty model doesn't incorporate minutes uncertainty.

### 3.2 Ensemble Architecture Limitations
**Current**: XGBoost + LightGBM → Ridge meta-learner. The meta-learner sees only 2 features (xgb_pred, lgbm_pred).
**Problems**:
- Ridge with 2 features is essentially a weighted average — it can't capture interaction effects.
- No CatBoost in the ensemble (it handles categorical features natively, which would simplify position/team encoding).
- The meta-learner is fit on the same OOF predictions used for stacking — there's a subtle information leak when the OOF fold sizes are small.

**Recommendations**:
1. **Add context features to meta-learner** — include `pre_minutes_avg5`, `is_b2b`, `player_game_num` in the Ridge input. This lets it learn "trust XGBoost more on B2B games" or "trust LightGBM more for low-minutes players."
2. **Add CatBoost as third base model** — CatBoost handles missingness and categoricals natively, which avoids the SimpleImputer information loss.
3. **Nested OOF for meta-learner** — use inner-loop OOF within each outer fold to generate leak-free meta-learner training data.

### 3.3 Quantile Regression Improvements
**Current**: Trains Q10/Q25/Q75/Q90 XGBoost models for uncertainty estimation. Derives `quantile_std = IQR / 1.35`.
**Problems**:
- The 1.35 divisor assumes normality — but the whole point of quantile regression is to capture non-normal distributions. For heavy-tailed stats like fg3m, this understates uncertainty.
- Q10/Q90 models are trained but not used for tail risk in signal gating (only IQR-based std feeds into the t-distribution).

**Recommendations**:
1. **Use Q10/Q90 directly for tail risk gating** — if `Q10 > prop_line`, that's a strong OVER signal (90% probability). If `Q90 < prop_line`, strong UNDER. This is a non-parametric confidence check that doesn't rely on distributional assumptions.
2. **Non-parametric P(over) from quantile position** — the function `predict_quantile_std()` already computes `p_over_nonparam` but it's returned as NaN when unavailable. Wire it into `compute_prediction_advantages()` as a blending input with the t-distribution P(over).
3. **Stat-specific tail behavior** — fg3m and rebounds have discrete distributions (0, 1, 2, 3...). Consider ordinal regression or Poisson regression for these targets instead of continuous quantile regression.

### 3.4 Residual Model (Stage 3) Scope
**Current**: Trains residual models for points, rebounds, assists, fg3m. Clips corrections to ±20% of base prediction.
**Problems**:
- No residual model for minutes — the highest-impact prediction.
- The 20% clip is applied uniformly regardless of confidence in the residual prediction. High-confidence residual corrections (where the residual model itself has low uncertainty) should be allowed to correct more.
- Residual interaction features are limited: `oof_pred_x_b2b`, `oof_pred_x_injury_pressure`. Missing: `oof_pred_x_opponent_defense`, `oof_pred_x_pace`, `oof_pred_x_rest`.

**Recommendations**:
1. **Add minutes residual model** — minutes prediction error is the largest single error source.
2. **Adaptive clip based on residual model confidence** — if the residual model's quantile spread is narrow, allow up to 30%; if wide, clip to 10%.
3. **Expand residual interaction features** — add `oof_pred_x_opp_def_rating`, `oof_pred_x_matchup_pace`, `oof_pred_x_rest_days`.

### 3.5 Combo Props (PRA, P+R, P+A)
**Current**: Not modeled.
**Recommendation**: Model as sum of individual predictions with correlation-adjusted uncertainty:
```
pred_PRA = pred_points + pred_rebounds + pred_assists
std_PRA = sqrt(std_pts^2 + std_reb^2 + std_ast^2 + 2*cov(pts,reb) + 2*cov(pts,ast) + 2*cov(reb,ast))
```
Estimate covariance from OOF residuals. This is a high-volume market with relatively simple modeling requirements.

---

## 4. SIGNAL GATING & EDGE COMPUTATION

### 4.1 Probability Model Assumptions
**Current**: Uses `t(df=7)` distribution for all stats (line 5221).
**Problem**: df=7 was chosen as a universal compromise. Points (roughly normal) don't need heavy tails. fg3m (discrete, zero-inflated) needs heavier tails or a different distribution entirely. Rebounds and assists are somewhere in between.
**Recommendation**: **Per-stat degrees of freedom** — fit df from historical OOF residuals per stat type. Likely: points→df=15, rebounds→df=8, assists→df=7, fg3m→df=5. Note: this is lower Accuracy than fixing probability saturation (§4.5) — saturated calibrated probs are a more urgent source of miscalibration than the t-distribution df choice.

### 4.2 Signal Threshold Asymmetry
**Current**: OVER requires 20% edge, UNDER requires 15% (line 173-176). `SUPPRESS_LEAN_OVER = True`.
**Problem**: The OVER suppression is a blanket policy. Some OVER signals (e.g., high-minutes starters with confirmed lineup against bad defenses) are legitimate. The current policy blocks all LEAN OVERs regardless of context.
**Recommendation**: **Context-dependent OVER gating** — instead of blanket suppression, create a lightweight classifier that predicts OVER signal reliability from features: `confirmed_starter`, `injury_unavailability_prob`, `opp_def_rating`, `minutes_avg5`. Allow LEAN OVER when the classifier predicts >60% reliability.

### 4.3 Line Movement Integration
**Current**: Line movement (open → current) is used to upgrade LEAN → HIGH CONFIDENCE if movement confirms model direction by ≥2% (line 5350). Also tracked for market efficiency score computation.
**Problem**: Line movement is only used for confidence upgrades, not for edge adjustment. If the line moved toward the model's prediction, the remaining edge is smaller but the signal is more trustworthy. If it moved away, the edge is larger but the market disagrees.
**Recommendation**: **Bayesian line-movement adjustment** — weight the model's prediction by line movement: `adjusted_pred = pred * 0.85 + closing_line * 0.15` when closing line is available. This incorporates closing-line efficiency without surrendering to it.

### 4.4 Correlated Signal Handling
**Current**: `correlated` flag is set when multiple props for the same player have the same direction (line 5421-5430). Portfolio caps limit per-player signals to 2.
**Problem**: Cross-player correlation isn't addressed. If 3 players on the same team all have OVER signals, there's implicit game-total correlation — if the game is low-scoring, all 3 lose.
**Recommendation**: **Game-level correlation penalty** — when >2 signals from the same team are all OVER (or all UNDER), apply a Allocation-like correlation discount to the confidence level. Alternatively, enforce a max of 2 same-direction signals per team.

### 4.5 Probability Saturation Control
**Current**: Calibrated probabilities (isotonic/Platt) can hit extremes — `p_under=1.000` or `p_over=1.000` — creating false certainty in edge computation. When calibrated_prob = 1.0, any finite line produces an "infinite" edge, and downstream signal gating treats it as the highest-confidence pick.
**Root cause**: Isotonic regression can map OOF predictions near the tails to 0.0 or 1.0 when training data is sparse in those regions. The t-distribution CDF itself approaches 0/1 for large z-scores but should never reach exactly 0 or 1.
**Recommendations**:
1. **Hard-clamp calibrated probabilities to [0.02, 0.98]** — no signal should ever claim >98% confidence. Apply the clamp after calibration but before edge computation.
2. **Store and display both raw (t-distribution) and calibrated probabilities** in the canonical results CSV. This enables post-hoc diagnosis of whether saturation came from the t-distribution or the calibrator.
3. **Per-stat probability distribution diagnostics** — add to the calibration report: histogram of calibrated P(over) by stat_type. Flag any stat where >5% of predictions are saturated (p < 0.02 or p > 0.98) as a calibration health issue.
4. **Calibrator training guard** — when fitting isotonic/Platt, ensure the training labels have at least 10 positives and 10 negatives per bin. If a bin is too sparse, fall back to the raw t-distribution probability for that region.

### 4.6 Hard Real-Time Lineup Fallback
**Current**: `fetch_confirmed_lineups()` pulls from ESPN. If ESPN returns 0/n confirmed starters close to tip-off, the pipeline proceeds with stale starter assumptions. The OVER minutes gate then blocks signals for players whose minutes estimate is anchored to an outdated lineup.
**Problems**:
- No second-source fallback when ESPN lineup data is empty.
- No auto-rerun trigger — if lineups confirm after the initial run, predictions aren't refreshed.
- Stale lineup data cascades: wrong minutes → wrong per-minute rates → wrong edges → blocked OVER signals that should have fired (or UNDER signals that shouldn't have).

**Recommendations**:
1. **Second-source lineup check** — when ESPN returns 0 confirmed starters for a game within 60 minutes of tip, query a fallback source (e.g., NBA CDN `/pregame/` endpoint or RotoWire). Accept the source with higher confirmation count.
2. **Auto-rerun gate** — add a `--rerun-on-lineup` mode: the initial run generates predictions with `lineup_confirmed=False`. A cron job re-runs 30 minutes before tip; if new lineups are found, regenerate affected games' signals. Flag rerun signals with `is_lineup_refresh=True` in canonical results.
3. **Lineup freshness metric** — track and report `pct_games_with_confirmed_lineup` in the daily report. If <50% of games have confirmed lineups at prediction time, warn in the deploy gate.

---

## 5. TESTING & VALIDATION

### 5.1 Walk-Forward Backtest Limitations
**Current**: `run_walk_forward_backtest()` uses season-based folds (train on prior seasons, test on next). The market-line backtest uses an 80/20 chronological split with up to 30 test dates.
**Problems**:
- Walk-forward only tests model accuracy (MAE/RMSE/R²), not signal profitability. It doesn't compute hit rates, P/L, or market efficiency score.
- The 80/20 split for market-line backtest doesn't match the walk-forward structure — models are trained on the full train set, not retrained per fold.
- No statistical significance testing — a 55% hit rate on 100 bets has a wide confidence interval (44%-66% at 95% CI).

**Recommendations**:
1. **Unify walk-forward and market-line backtest** — in each walk-forward fold, train models, generate predictions, match to cached market lines, compute signals, grade against actuals. This gives a realistic end-to-end profitability estimate.
2. **Bootstrap confidence intervals** — for hit rate, Accuracy, and market efficiency score, report 95% CIs via bootstrap resampling (1000 iterations). This tells you whether a 53% hit rate is real or noise.
3. **Permutation test for signal value** — shuffle signal assignments and compute null P/L distribution. If actual P/L isn't significantly above the null, the model isn't adding value over random.

### 5.2 Missing Unit Tests
**Current**: No unit tests for any component of the props pipeline.
**Recommendations** (ordered by impact):
1. **Feature leakage test** — assert that all `pre_` features are strictly shift(1) from actuals. Feed a known sequence, verify rolling calculations.
2. **OOF prediction leakage test** — verify that OOF predictions for fold K never use data from fold K.
3. **Signal gate tests** — verify that SUPPRESS_LEAN_OVER, minutes gate, calibration drift gate all suppress correctly.
4. **Canonical results idempotency** — verify that running `save_canonical_results()` twice with the same run_id doesn't duplicate rows.
5. **Quantile crossing test** — verify Q10 ≤ Q25 ≤ Q75 ≤ Q90 for all predictions.

### 5.3 Overfitting Detection
**Current**: No systematic overfitting monitoring. The residual model + market residual model + bias correction + star-out boost stack multiple correction layers, each adding overfit risk.
**Recommendations**:
1. **Track train vs test MAE ratio per model layer** — if train MAE is 50% lower than test MAE, the model is overfit.
2. **Feature importance stability** — across walk-forward folds, check whether the top-10 features by importance are stable. Unstable importance = feature is fitting noise.
3. **Correction magnitude monitoring** — log the average absolute correction from residual model, market residual, bias correction, and star-out boost. If corrections are growing over time, that's a sign of drift.

---

## 6. FEEDBACK LOOP & MONITORING

### 6.1 Deploy Gates Weakness
**Current**: 5 gates (min_graded, market efficiency score, calibration, model_freshness, missingness). DEPLOY_GATES_ENFORCE = False (advisory only).
**Problems**:
- Market efficiency gate (Gate 2) requires `DEPLOY_MES_MIN_SAMPLE = 50` rows with nonzero line movement. Early in the pipeline, this is nearly impossible to satisfy.
- Gate 2 uses line-point market efficiency score, not odds-based market efficiency score. Line movement of 0.5 points on a 25.5 line is very different from 0.5 on a 2.5 line.
- No per-stat gate enforcement — if points is calibrated but fg3m is miscalibrated, the entire system should still deploy points signals.
- No "degradation velocity" gate — calibration can drift slowly over weeks without triggering the threshold. A rapid drop (e.g., Brier jumps 0.05 in 3 days) should be an immediate alert.

**Recommendations**:
1. **Per-stat deploy gates** — allow signals for stats that pass all gates, suppress only the failing stats. This is partially implemented via `calibration_degraded_stats` but not wired to market efficiency score or min_graded gates.
2. **Odds-based market efficiency score as primary metric** — compute market efficiency score from implied probability difference (closing - opening) in the signal direction. This normalizes across line values.
3. **Degradation velocity alert** — track daily Brier score. If the 3-day rolling Brier increases by >0.05 from the 14-day rolling Brier, trigger an immediate alert regardless of absolute threshold.
4. **Graduated enforcement** — instead of binary enforce/advisory, implement: (a) advisory → (b) suppress LEAN signals for failing stats → (c) suppress all signals for failing stats → (d) full system pause.

### 6.2 Calibration Report Gaps
**Current**: `compute_calibration_report()` slices by stat_type, confidence, edge_bucket.
**Missing slices**:
- **By home/away** — home OVER signals may have different reliability than away OVER signals.
- **By rest (B2B vs normal)** — B2B games may systematically miscalibrate.
- **By model confidence vs market confidence** — signals where the model strongly disagrees with the market may have different accuracy.
- **By player minutes tier** — high-minutes starters vs low-minutes bench players.
- **By signal age** — signals made at 9 AM vs signals made at game time (with lineup confirmation) may differ.

### 6.3 Points Bias Correction Scope
**Current**: Only points get rolling bias correction (lines 5004-5093). The Bayesian shrinkage uses `k=120`, cap `±1.5`.
**Problem**: If the model systematically under-predicts rebounds by 0.5 (e.g., due to pace changes), there's no correction mechanism.
**Recommendation**: **Generalize bias correction to all stat types** — apply the same Bayesian shrinkage framework to rebounds, assists, and fg3m. Gate each stat independently (require min 75 graded rows, lookback 30 days). This is a ~30-line change.

### 6.4 Weekly Retrain Coverage
**Current**: `run_weekly_retrain()` retrains core models, runs market-line backtest, runs calibration report.
**Missing**:
- No feature importance comparison vs prior week.
- No OOS accuracy comparison vs prior week (is the new model better or worse?).
- No automatic threshold adjustment based on calibration results.
- Market residual models and probability calibrators are NOT retrained during weekly retrain — they depend on cached market line history that may not be regenerated.

**Recommendations**:
1. **Save pre/post retrain metrics** — before training new models, run a quick OOS eval. After training, run the same eval. Log both to `prop_logs/retrain_comparison.csv`.
2. **Retrain market models during weekly retrain** — call `train_market_residual_models()` as part of the retrain pipeline if sufficient prop history exists.
3. **Auto-adjust signal thresholds** — if calibration shows OVER signals at 48% hit rate over 90 days, automatically tighten `MIN_ADVANTAGE_PCT_BY_SIDE["OVER"]` by 2 points.

---

## 7. CODE QUALITY & PERFORMANCE

### 7.1 Monolith Dependency
**Current**: The 9235-line props script imports ~20 functions from the 4082-line monolith. Changes to the monolith can silently break props.
**Recommendation**: Add an import health check at the top of `predict_player_props.py` that validates all imported functions exist and have expected signatures. Log a warning (not crash) if signatures change.

### 7.2 Feature Cache Invalidation
**Current**: `PLAYER_FEATURE_CACHE_VERSION = "v8"`. Cache is a pickle file. When features change, the version must be bumped manually.
**Problem**: Forgetting to bump the version means stale features are silently used.
**Recommendation**: **Auto-version from feature list hash** — compute a hash of the `get_feature_list()` output for all targets. If the hash changes, invalidate the cache automatically.

### 7.3 Training Time
**Current**: Full pipeline (data load + feature engineering + multi-stage training + uncertainty models + residual models) takes 15-30 minutes depending on data size.
**Bottleneck**: Rolling window computations in `build_player_features()` use `groupby().transform(lambda: ...)` which is inherently slow. Each rolling column does a full groupby pass.
**Recommendation**: **Vectorized rolling** — replace `groupby().transform(lambda s: s.shift(1).rolling(...).mean())` with `groupby().shift(1).rolling(...).mean()` (pandas ≥2.0 supports this natively). For EWM, similarly use `groupby().shift(1).ewm(...)`. This alone could cut feature engineering time by 40-60%.

### 7.4 Memory Pressure
**Current**: `build_player_features()` calls `.copy()` 4 times for "defragmentation" and creates many intermediate columns.
**Recommendation**: Use `pd.concat()` to batch-add columns instead of repeated `pg[new_col] = ...` inserts. Remove intermediate `.copy()` calls — they double memory usage. The defragmentation pattern is a workaround for a pandas performance warning that can be suppressed.

---

## 8. PRIORITIZED IMPLEMENTATION ROADMAP

### Tier 1 — Highest Expected Value (implement in this order)
| # | Item | Section | Est. Complexity | Expected Impact |
|---|------|---------|-----------------|-----------------|
| 1 | Team-aware market line keying & source dedup | §1.1.5 | Low-Medium | High — eliminates duplicate/mismatched lines that corrupt edge computation |
| 2 | Forward-injury + starter reconstruction in training | §2.3.1, §2.4 | Medium | High — unlocks `injury_pressure_delta` and `confirmed_starter` as real features |
| 3 | Calibration saturation fix + per-stat probability diagnostics | §4.5 | Low | High — eliminates p=1.000 artifacts, adds visibility into calibrator health |
| 4 | Unified walk-forward market backtest with bootstrap CIs | §5.1.1, §5.1.2 | Medium | High — gives realistic profitability estimate with statistical significance |
| 5 | Per-stat deploy gate enforcement (graduated) | §6.1.1, §6.1.4 | Low-Medium | High — allows partial deployment, graduated severity |

This sequence gives the best reliability gain per unit of work.

### Tier 2 — Strong Accuracy
| # | Item | Section | Est. Complexity | Expected Impact |
|---|------|---------|-----------------|-----------------|
| 6 | Hard real-time lineup fallback | §4.6 | Medium | Medium-High — prevents stale lineup cascades |
| 7 | Blend two-stage and direct predictions | §3.1.1 | Medium | Medium — reduces minutes error propagation |
| 8 | Generalize bias correction to all stats | §6.3 | Low | Medium — fixes systematic per-stat drift |
| 9 | Feature leakage unit tests | §5.2.1 | Low | High — prevents silent bugs |
| 10 | Per-stat df for t-distribution | §4.1 | Low | Low-Medium — less impactful than fixing saturation (§4.5) |

### Tier 3 — Incremental Gains
| # | Item | Section | Est. Complexity | Expected Impact |
|---|------|---------|-----------------|-----------------|
| 11 | Combo props (PRA) | §3.5 | Medium | Medium — opens high-volume market; requires stable calibration first |
| 12 | The Odds API as second line source | §1.1.1 | Medium | Medium |
| 13 | Opponent injury features | §2.3.2 | Medium | Medium |
| 14 | Context-dependent OVER gating | §4.2 | Medium | Medium |
| 15 | Quantile-based tail risk gating | §3.3.1 | Low | Low-Medium |
| 16 | Auto-version feature cache | §7.2 | Low | Low (quality of life) |
| 17 | Vectorized rolling computation | §7.3 | Medium | Low (performance, not a prediction-edge lever while feature cache works) |
| 18 | Add meta-learner context features | §3.2.1 | Low | Low-Medium |
| 19 | Degradation velocity alert | §6.1.3 | Low | Low-Medium |

---

## 9. RISKS & ANTI-PATTERNS TO WATCH

1. **Correction stacking overfit** — The pipeline applies 4+ sequential corrections (two-stage → residual model → market residual → bias correction → star-out boost). Each layer's training error can compound. Monitor the total magnitude of corrections relative to base prediction — if corrections average >15% of the base prediction, something is wrong.

2. **Threshold proliferation** — There are 40+ configurable constants in lines 1-280. Each constant is a potential overfitting lever. Any threshold change should be validated via walk-forward backtest, not just intuition.

3. **Survivorship bias in calibration** — `calibration_degraded_stats` only checks stats with `min_sample ≥ 50` graded signal rows. New or rare stats (fg3m early in the season) won't be checked at all, allowing them to signal without validation.

4. **Train/prediction feature drift** — Multiple features are NaN at training time but populated at prediction time (injury fields, lineup fields). The model learns different patterns than it sees in production. The §2.3 and §2.4 recommendations address this directly.
