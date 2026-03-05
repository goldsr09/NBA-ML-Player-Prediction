# NBA Prediction System — Improvement Spec

Comprehensive audit of the codebase across data quality, feature engineering, model architecture, validation, and feedback loops. Organized by priority within each section.

---

## 1. Data Quality & Ingestion

### 1A. No Data Validation Layer (HIGH)
**Current state:** Raw JSON from NBA CDN and ESPN APIs is parsed directly with `_to_float()` helper but zero schema validation. If an API changes field names, adds nulls in new positions, or returns partial payloads, failures are silent (defaulting to `NaN` everywhere).

**Gaps:**
- No schema contracts on API responses — a field rename (e.g., `fieldGoalsAttempted` → `fga`) silently produces all-NaN features
- No row-count sanity checks after parsing (e.g., "expect 2 team rows per game, ~26 player rows")
- No anomaly detection on parsed values (e.g., a team scoring 300 points, negative possessions, `off_rating > 200`)
- Boxscore cache files are never invalidated — if a game is updated (stat corrections), stale data persists forever
- ESPN odds endpoint returns `items[0]` without checking which provider it is; provider priority shifts can silently change the odds source

**Recommendations:**
1. Add a lightweight `validate_boxscore(game_dict)` that asserts expected fields exist and values are in reasonable ranges
2. Add a `validate_team_game_row(row)` post-parse check: `0 <= pts <= 200`, `0 <= poss <= 130`, `fga > 0`, etc.
3. Add cache staleness: re-fetch boxscores for games completed within the last 48 hours (stat corrections window)
4. Log and alert when NaN rates exceed thresholds per feature column (e.g., >5% NaN in `off_rating` is suspicious)
5. Pin the odds provider preference (check `provider.id` or `provider.name` against a priority list rather than assuming `items[0]`)

### 1B. Historical Data Integrity (MEDIUM)
**Current state:** `fetch_historical_seasons.py` caches boxscores per season. `build_team_games_and_players(include_historical=True)` loads 4 seasons. No deduplication guard across season boundaries.

**Gaps:**
- No check for duplicate `game_id` across seasons (e.g., if a game's ID appears in both the current-season fetch and the historical cache)
- No validation that historical odds match the correct game (join on `[game_date_est, home_team, away_team]` can false-match doubleheaders or rescheduled games)
- Playoff games could leak in — the `gid.startswith("002")` filter catches regular season but some pipelines may not enforce this consistently
- No data versioning — if the parsing logic changes, old cached JSON produces different features than new JSON, creating training/serving skew

**Recommendations:**
1. Add explicit deduplication on `game_id` after concatenating historical + current data
2. Add a secondary join key (game_time_utc within ±3 hours) for odds matching
3. Add a cache version stamp: store `{"cache_version": 2, "data": ...}` so old caches are auto-refreshed when parsing changes
4. Add a one-time integrity scan script that validates all cached boxscores parse correctly

### 1C. ESPN Injury Report Fragility (MEDIUM)
**Current state:** `fetch_espn_injury_report()` scrapes the ESPN injury page. Players are fuzzy-matched to the roster via `match_injury_report_to_players()`.

**Gaps:**
- No coverage metric: what percentage of actual DNPs were captured by the injury report?
- Fuzzy matching can false-positive (e.g., "Marcus Morris" matching "Markieff Morris")
- No handling for load management (listed as healthy, sits out) — this is invisible to the system
- Injury report timing: fetched once at prediction time but injuries can be updated hours before tipoff
- The `evaluate_injury_feed_coverage()` function exists in player props but isn't used systematically for game predictions

**Recommendations:**
1. Implement injury report coverage tracking: compare predicted-out vs actual-DNP post-game
2. Add a re-fetch window: if prediction time is >4 hours before tipoff, log a warning that injury data may be stale
3. Use player-id-based matching instead of name matching where ESPN provides IDs
4. Track load management patterns: flag players with >20% DNP rate on B2B games

### 1D. Missing Data Sources (LOW — Future Enhancement)
**Current sources:** NBA CDN (boxscores, schedule), ESPN (odds, injury reports, player props).

**Missing:**
- **Play-by-play data**: clutch performance, garbage time identification, lineup-specific net ratings
- **Tracking/shot data**: shot quality (expected eFG), contested vs open shots, pace in half-court vs transition
- **Referee assignments**: `parse_officials()` exists in the monolith but referee features are only partially integrated
- **Vegas consensus lines** (multiple books): currently single-provider odds; consensus lines are more stable
- **Weather data for travel**: not relevant to indoor games but altitude/timezone effects are partially captured
- **Social media / news sentiment**: load management signals, team chemistry
- **Real-time lineup confirmations**: the `fetch_confirmed_starters_for_event()` exists in props but isn't used for game predictions

---

## 2. Feature Engineering

### 2A. Opponent-Adjustment Depth (HIGH)
**Current state:** Simple opponent adjustment exists: `adj_off = raw_off - (opp_def - league_avg_def)`. This is a single-iteration adjustment.

**Gaps:**
- No iterative opponent adjustment (RAPM-style). The current method doesn't account for schedule strength transitively
- No opponent-strength-weighted rolling averages (e.g., a team's last-5 against top-10 defenses vs bottom-10)
- The opponent adjustment uses `expanding` mean of the opponent's season stats, which over-weights early-season data
- No distinction between home-court and neutral-site games in the adjustment

**Recommendations:**
1. Implement a simple iterative SRS (Simple Rating System): solve `team_rating = avg_margin + avg(opp_rating)` iteratively until convergence (5-10 iterations). This is cheap and materially better than single-pass
2. Add opponent-quality-bucketed rolling stats: split recent games into "vs top-10" and "vs bottom-10" opponent rating
3. Use EWM-weighted opponent stats for the adjustment (more recent opponent form matters more)

### 2B. Player-Level Features for Game Predictions (HIGH)
**Current state:** Player availability is proxied via "absent rotation count" and "missing minutes" from the previous game's roster. No individual player impact modeling.

**Gaps:**
- No player impact metrics (e.g., RPM, BPM, VORP equivalents computed from boxscore data)
- The injury proxy carries forward the *previous game's* absence data, not real-time. If a star was out last game but is back, the proxy still says "missing"
- No modeling of *who* is out — losing a bench player vs a star has vastly different effects, but `missing_minutes5` treats them equally
- `lineup_continuity` and `active_roster_plus_minus` features exist but are simple aggregates
- No starter vs bench decomposition of team strength

**Recommendations:**
1. Compute a simple per-player "impact score" from boxscore data: `(minutes_share * (plus_minus_per36 + 0.5 * usage_proxy))`. Weight absences by this score
2. Add "star player absent" as a separate, high-signal feature (top-1 and top-2 player by impact score). Already partially exists for injury report but not for the boxscore-derived proxy
3. Replace carry-forward proxy with real-time injury report data for prediction time (already done in `_attach_injury_report_features` but only for upcoming predictions, not for historical training data)
4. Add lineup-combination features: how many of the usual starting 5 are available, weighted by their co-play net rating

### 2C. Pace & Context Interactions (MEDIUM)
**Current state:** Pace (possessions) is a feature. B2B, travel, and altitude are independent features. No interactions.

**Gaps:**
- No pace-matchup interaction: two fast-paced teams meeting should produce higher totals than the sum of their individual pace signals
- No B2B × pace interaction (tired teams in fast-paced games score differently)
- No altitude × travel interaction beyond the simple binary `altitude_short_rest`
- No conference/division context (Western Conference road trips are structurally different than Eastern)
- No "revenge game" or "rivalry" features (recently traded player, divisional matchup)
- No season-phase interaction (teams behave differently in Oct-Nov vs Jan-Feb vs March-April)

**Recommendations:**
1. Add `pace_matchup = home_pace_avg5 * away_pace_avg5 / league_avg_pace^2` as a multiplicative interaction
2. Add `b2b_x_pace = b2b * opp_pace_avg5` for totals modeling
3. Add month/season-phase ordinal (0-5) as a feature — early-season noise vs late-season intensity
4. Add `division_game` binary and `conference_game` binary

### 2D. Market-Derived Feature Engineering (MEDIUM)
**Current state:** Market lines are used as direct features (spread, total, implied prob) plus line movement (close - open). Residual models correct market-based predictions.

**Gaps:**
- No "sharp money" signal: large line movements in low-volume periods indicate sharp action
- No steam move detection (rapid line changes across multiple books)
- No market-model disagreement bucketing (model says home by 5, market says home by 1 → high-conviction signal)
- Line movement features are NaN when open lines are unavailable (common for upcoming games captured early)
- No historical closing line value (CLV) tracking for the game prediction pipeline (only exists for Kalshi Q4 and player props)
- The `market_home_implied_prob_open` is derived from spread via logistic approximation (`1/(1+exp(spread/6.5))`) rather than actual opening moneyline odds

**Recommendations:**
1. Track and feature-engineer CLV for game predictions: how much did the model's pick move the line in its direction by close?
2. Add market disagreement magnitude as a feature: `abs(model_margin - market_margin)` bucketed into low/med/high
3. Use actual opening moneyline when available instead of spread-derived approximation
4. Add "reverse line movement" detector: line moves against public betting percentages

### 2E. Time-Decay & Recency Weighting (LOW)
**Current state:** EWM with span=10 is used for some metrics. Season-long expanding averages are also computed.

**Gaps:**
- No exponential decay on training sample weights (recent games should matter more in model training, not just features)
- No separate short-term (3-game) and medium-term (15-game) EWM windows
- Rolling windows are fixed (5, 10, season) — no adaptive windowing based on schedule density
- No momentum/velocity features (is the EWM accelerating or decelerating?)

**Recommendations:**
1. Add sample weights to XGBoost training: `weight = exp(-lambda * days_since_game)` with `lambda` tuned via CV
2. Add 3-game EWM for very recent form capture
3. Add "acceleration": `ewm3 - ewm10` as a momentum signal

---

## 3. Model Architecture & Multi-Layer Design

### 3A. Ensemble Architecture (HIGH)
**Current state:** Final prediction is a 3-model ensemble for win probability (XGB calibrated + LightGBM + margin-consistent) blended 50/50 with a market-residual correction model. Total and margin have separate XGB regressors.

**Gaps:**
- Ensemble weights are hardcoded (50/50 base vs market, equal weight across 3 base models). No learned stacking
- No model diversity beyond XGB and LightGBM (missing: logistic regression baseline, random forest, neural net)
- The ensemble averages probabilities, not log-odds — this is suboptimal for calibration
- No per-game adaptive weighting (e.g., weight the market model more when line movement is high, weight the stats model more early in season)
- LightGBM hyperparameters are hardcoded (`n_estimators=200, max_depth=4`) while XGB gets Optuna tuning
- The calibrated classifier uses `CalibratedClassifierCV(method="sigmoid", cv=5)` with random CV folds, violating chronological ordering

**Recommendations:**
1. Replace fixed ensemble weights with a learned stacking layer: train a logistic regression on out-of-fold predictions from each base model (using time-series CV)
2. Tune LightGBM with Optuna just like XGB
3. Fix calibration to use time-series CV folds (the current `cv=5` uses random K-fold which leaks future data)
4. Average in log-odds space: `logit(p_ensemble) = mean(logit(p_i))` then convert back
5. Add a simple baseline model (regularized logistic regression on top-10 features) to the ensemble — provides stability
6. Consider adding a shallow neural net (2-3 layers) for non-linear interaction capture

### 3B. Margin-Total-Win Consistency (HIGH)
**Current state:** Win probability, margin, and total are predicted by separate models. A post-hoc `margin_consistent_win_prob()` derives win prob from margin via `Phi(margin/std)`.

**Gaps:**
- The three outputs (win, margin, total) are not jointly consistent. The predicted margin may imply a different win probability than the win model outputs
- Home score and away score are derived as `0.5*(total ± margin)` which doesn't account for home/away scoring asymmetry
- No explicit home-score and away-score models (these would naturally produce consistent margin + total)
- The margin residual std is computed on the entire training set, not time-varying — it should be larger early in the season

**Recommendations:**
1. Train home-score and away-score regressors directly. Derive `margin = home - away`, `total = home + away`, `p(win) = Phi(margin / residual_std)`. This ensures all outputs are mechanically consistent
2. Alternatively, use a multi-output model (XGBoost supports multi-target) that jointly predicts margin and total
3. Make residual std time-varying: compute it on a rolling window (e.g., last 200 games) rather than the full training set

### 3C. Market Residual Model Design (MEDIUM)
**Current state:** A separate XGBoost regressor predicts `actual_margin - market_implied_margin` and `actual_total - market_total`. The predicted residual is added to the market line.

**Gaps:**
- The residual model uses the same features as the base model plus market features — this can lead to overfitting to the market signal
- No separate feature selection for the residual model (SHAP or ablation to find which features have genuine residual predictive power beyond the market)
- No confidence gating: the residual correction is applied uniformly regardless of the model's certainty
- The correction is uncapped — a large predicted residual (+15 points) on a 220 total is probably noise
- No temporal stability check: does the residual model's edge persist out-of-sample month-over-month?

**Recommendations:**
1. Add residual clipping: cap corrections at ±1.5 * historical residual std
2. Feature-select for the residual model independently: only keep features whose SHAP contribution to the residual model exceeds a minimum threshold
3. Add a confidence gate: only apply residual correction when `abs(predicted_residual) > min_threshold` and the model's uncertainty (from quantile regression or std estimate) is below a maximum
4. Track residual model edge over time and auto-disable when it degrades below breakeven

### 3D. Player Props Model Architecture (MEDIUM)
**Current state:** Two-stage model (XGBoost base + OOF residual correction). Market residual models. Quantile uncertainty estimation. Signal gating via calibration monitoring.

**Gaps (from reading the 9235-line file):**
- The two-stage model's residual correction is clipped to ±20% of base prediction but this is a fixed threshold
- No player embedding or clustering — each player's model sees the same features regardless of play style
- No game-context interaction in the props model (e.g., a player's scoring should be conditioned on the opponent's defense rating *at their position*)
- Position-specific opponent features are missing (e.g., opponent's C defense vs opponent's PG defense)
- The `compute_forward_injury_pressure()` is a good idea but doesn't account for *which* teammates are out (a PG being out affects the team's assists more than a C being out)
- Market line features are used but no "vig-adjusted true line" computation (the market line includes bookmaker margin)

**Recommendations:**
1. Add player-role clustering (k-means on usage%, assist%, rebound%, etc.) and use cluster ID as a categorical feature
2. Add opponent positional defense features: opponent's defensive rating against the player's position
3. Vig-strip the market lines before using as features: compute true line from over/under odds
4. Add teammate-absence interaction: when Player X is out, Player Y's usage goes up by Z% historically

### 3E. Q4 Model Integration (LOW)
**Current state:** Separate `train_q4_model.py` trains an XGBoost regressor on end-of-Q3 state to predict final totals. Used by `kalshi_q4_edge.py` for live Kalshi markets.

**Gaps:**
- No integration with the main game prediction pipeline — the Q4 model could improve live total predictions
- No half-time model (Q1+Q2 → final total) which would be useful for live betting earlier in the game
- The Q4 model's linear baseline (`1.0604 * thru_3q + 44.34`) is never compared against the ML model in production
- No Q4 model for margin (only totals)

**Recommendations:**
1. Build a half-time model for total and margin predictions
2. Integrate Q4 model outputs into the live adjustment logic in `apply_live_adjustments()`
3. Add a Q4 margin model for live spread betting

---

## 4. Testing & Validation

### 4A. Zero Test Coverage (CRITICAL)
**Current state:** No test files exist anywhere in the repository. No `tests/` directory, no `pytest.ini`, no CI/CD.

**Gaps:**
- No unit tests for any parsing functions (boxscore parsing, odds parsing, feature engineering)
- No integration tests for the training pipeline
- No regression tests to catch when code changes affect prediction accuracy
- No smoke tests that verify the pipeline runs end-to-end
- No property tests for mathematical functions (e.g., `haversine_miles` symmetry, `american_to_prob` invertibility, `normalize_prob_pair` summing to 1)

**Recommendations (prioritized):**
1. **Immediate:** Add unit tests for parsing functions with known-good fixture data:
   - `parse_team_box_rows()` — provide a sample boxscore JSON, assert expected output
   - `parse_player_box_rows()` — same
   - `american_to_prob()` — test boundary cases (±100, ±200, None, NaN)
   - `haversine_miles()` — known distance pairs
   - `normalize_prob_pair()` — assert output sums to ~1.0
2. **Short-term:** Add integration tests:
   - Feature engineering pipeline: given a small DataFrame, verify expected columns and no NaN leakage
   - Chronological split: verify no future data in training set
   - Model training: verify output shapes and probability calibration on synthetic data
3. **Medium-term:** Add regression tests:
   - Store expected accuracy metrics for a fixed test set snapshot
   - Flag when a code change degrades accuracy by >1%
4. **Infrastructure:** Add `pytest` to dependencies, create `tests/` directory, add a CI pipeline (GitHub Actions) that runs tests on every push

### 4B. Backtesting Rigor (HIGH)
**Current state:** A `--backtest` mode exists that splits the last 20% of current-season games as test. `time_series_cv_folds()` exists for cross-validation.

**Gaps:**
- The 80/20 chronological split is a single point-in-time estimate — no walk-forward validation for game predictions (only exists for player props)
- No purge gap between train and test to prevent information leakage from rolling features (a game on day N uses features from day N-1; if day N-1 is in train and day N is in test, there's indirect leakage)
- No multi-season out-of-sample test: train on 2021-24, test on 2024-25 as a whole
- CV folds use `min_train=200` games — this doesn't account for the fact that early-season games have less reliable features
- No statistical significance testing on model comparisons (e.g., "is the enhanced model's 0.3% AUC improvement real or noise?")
- The backtest doesn't track ATS/O-U P&L over time — it only reports aggregate metrics

**Recommendations:**
1. Implement walk-forward backtesting for game predictions: retrain monthly, predict the next month, track cumulative P&L
2. Add a 3-game purge gap between train and test in CV folds
3. Add paired bootstrap confidence intervals for all metric comparisons
4. Add a season-holdout test: train on all seasons except the most recent, test on the holdout
5. Track daily/weekly P&L curves, not just aggregate metrics — to detect streaky behavior

### 4C. Calibration Monitoring (MEDIUM)
**Current state:** `calibration_by_decile()` and `calibration_error()` exist in `nba_evaluate.py`. The props pipeline has extensive calibration monitoring.

**Gaps:**
- Game prediction calibration is only checked at backtest time, not monitored in production
- No reliability diagram visualization (plotting predicted vs actual by bin)
- No Brier skill score comparison against market baseline
- No calibration drift detection: is the model becoming miscalibrated over time?
- The `CalibratedClassifierCV` in the ensemble uses random folds (see 3A)

**Recommendations:**
1. Add a daily calibration tracker: after each day's games, update a rolling calibration curve
2. Compute Brier skill score relative to market: `BSS = 1 - (BS_model / BS_market)`
3. Alert when rolling ECE exceeds 0.05 for any 50-game window
4. Add a reliability diagram output (can be ASCII/CSV — doesn't need matplotlib)

### 4D. Feature Leakage Audit (MEDIUM)
**Current state:** Features use `shift(1)` for rolling averages. `pre_` prefix convention enforced.

**Gaps:**
- The opponent adjustment (`opp_def_rating_season`) groups by `[team, season, opp]` with `shift(1).expanding()` — this includes data from the specific team-vs-opponent history, which could create subtle leakage if there are few matchups
- `lineup_continuity` and `active_roster_plus_minus` are computed from the current game's player data, not pre-game data (potential leakage)
- Injury proxy features carry forward from the *previous game's actual roster*, not from a pregame prediction of who will play
- Season-level expanding means include the full season up to (but not including) the current game — but for the first game of a new season, they're NaN, creating a feature cliff
- The Elo update uses the current game's margin, then the pre_elo is stored before update — this is correct, but there's no explicit test verifying this

**Recommendations:**
1. Audit every feature for temporal correctness: write a test that verifies `corr(feature[t], target[t+1])` is non-trivial (feature has predictive power) but `corr(feature[t], target[t])` is not suspiciously high (which would indicate leakage)
2. Add an explicit leakage test: train on random features + one leaked feature, verify the leaked feature dominates
3. Fix lineup_continuity to use only pre-game information
4. Add season-start smoothing: for the first 5 games of a season, blend with previous season's end-of-season stats

---

## 5. Feedback Loop & Production Pipeline

### 5A. Game Prediction Feedback Loop (HIGH — Missing Entirely)
**Current state:** Player props have a comprehensive feedback loop (results history, calibration monitoring, auto-suppression, weekly retrain). Game predictions have NO equivalent.

**Gaps:**
- No canonical results history for game predictions (equivalent to `prop_results_history.csv`)
- No post-game grading: predictions are generated but never compared to actual outcomes systematically
- No CLV tracking for game prediction signals (spread, ML, total)
- No P&L tracking over time — impossible to know if the system is profitable
- No automatic signal suppression when the model degrades
- No weekly retrain cadence — models are only retrained when manually invoking the script

**Recommendations:**
1. Add `--grade-results` flag to `predict_upcoming_nba.py`:
   - Load yesterday's predictions CSV
   - Fetch actual results
   - Compute: ATS result, ML result, O/U result, P&L at each signal
   - Append to `game_results_history.csv`
2. Add `--calibration-report` flag:
   - Compute rolling metrics by market type (spread, ML, total)
   - Alert on degraded windows (similar to props pipeline)
3. Add CLV tracking: compare model's pick direction to closing line movement
4. Add automated P&L dashboard output (daily and cumulative)
5. Add `--weekly-retrain` with deploy gates (similar to props pipeline)

### 5B. Model Versioning & Reproducibility (HIGH)
**Current state:** Models are saved as `.joblib` in `analysis/output/models/` (gitignored). Tuned params are stored inside the model artifact. No versioning.

**Gaps:**
- No model versioning — overwriting `advanced_models.joblib` loses the previous model
- No training metadata (which data was used, which features, what hyperparameters, when trained)
- No reproducibility: re-running training can produce different results due to API data changes
- No A/B testing framework: can't compare two model versions on the same upcoming games
- No rollback capability: if a new model performs poorly, there's no way to revert
- Models directory is gitignored — model artifacts aren't tracked anywhere

**Recommendations:**
1. Add model versioning: save as `advanced_models_v{timestamp}.joblib` with a symlink `advanced_models_latest.joblib`
2. Store training metadata alongside the model: `{"trained_at": ..., "n_games": ..., "features": [...], "params": {...}, "cv_metrics": {...}}`
3. Keep the last 3 model versions for rollback
4. Add a `--compare-models` flag that loads two model versions and evaluates both on the same test set

### 5C. Dependency & Environment Management (MEDIUM)
**Current state:** No `requirements.txt`, no `pyproject.toml`, no `setup.py`, no virtual environment specification.

**Gaps:**
- Package versions are unspecified — a `pip install xgboost` could install a breaking version
- No lockfile for reproducible installs
- `shap` is optional but no conditional import guard everywhere it's used
- No Python version specification

**Recommendations:**
1. Add `pyproject.toml` with pinned dependency versions
2. Add `python-requires = ">=3.10"` (type hints use `X | Y` syntax)
3. Add a `Makefile` or `justfile` with common commands: `make train`, `make predict`, `make backtest`, `make test`

### 5D. Operational Automation (MEDIUM)
**Current state:** CLAUDE.md mentions a crontab example for weekly prop retrain. No actual cron/scheduling setup exists.

**Gaps:**
- No automated daily prediction generation
- No automated post-game grading
- No automated model retraining schedule
- No alerting when a pipeline fails (e.g., API down, parsing error)
- No health check for stale data (e.g., "last boxscore fetched was 3 days ago")

**Recommendations:**
1. Create a `scripts/daily_pipeline.sh` that orchestrates: fetch data → predict → grade yesterday → log results
2. Add health checks: verify cache freshness, API availability, model freshness
3. Add error notification (even a simple "write to error log file" is better than silent failures)

### 5E. Code Architecture & Maintainability (LOW)
**Current state:** The monolith is 4082 lines. Player props is 9235 lines. Both files contain data fetching, feature engineering, model training, evaluation, and prediction in a single file.

**Gaps:**
- The monolith contains everything from HTTP fetching to model evaluation — a single change anywhere risks breaking something elsewhere
- No separation of concerns: data layer, feature layer, model layer, and serving layer are interleaved
- The "thin re-export modules" (`nba_data.py`, `nba_features.py`, `nba_models.py`) were intended to decompose the monolith but contain no logic — they're vestigial
- Import cycles: `predict_upcoming_nba.py` imports from the monolith at the top AND has delayed imports from it inside functions
- Player props at 9235 lines is extremely difficult to navigate or modify safely

**Recommendations:**
1. Split the monolith into actual modules:
   - `nba_data.py`: API fetching, caching, parsing (no feature engineering)
   - `nba_features.py`: all feature engineering functions
   - `nba_models.py`: model training, tuning, evaluation
   - `nba_predict.py`: prediction serving logic
2. Use a proper configuration object instead of scattered constants at module top
3. Add type hints to all public functions (many are already typed, but internal helpers aren't)

---

## 6. Quick Wins (Can Implement Immediately)

| # | Item | Impact | Effort |
|---|------|--------|--------|
| 1 | Fix `CalibratedClassifierCV` to use time-series CV folds | Correctness | Low |
| 2 | Add residual clipping to market correction models (±1.5σ) | Robustness | Low |
| 3 | Add `requirements.txt` with pinned versions | Reproducibility | Low |
| 4 | Add unit tests for `american_to_prob`, `haversine_miles`, `normalize_prob_pair` | Correctness | Low |
| 5 | Add walk-forward backtest to game predictions | Validation | Medium |
| 6 | Add game prediction results grading (`--grade-results`) | Feedback loop | Medium |
| 7 | Implement iterative SRS for opponent adjustment | Feature quality | Medium |
| 8 | Add home-score / away-score regressors for consistency | Model quality | Medium |
| 9 | Add sample weighting by recency in XGBoost training | Feature quality | Low |
| 10 | Tune LightGBM with Optuna | Model quality | Low |

---

## 7. Summary Priority Matrix

| Priority | Category | Key Gap |
|----------|----------|---------|
| CRITICAL | Testing | Zero test coverage — any code change is a blind gamble |
| HIGH | Feedback | No game prediction grading/P&L tracking |
| HIGH | Validation | No walk-forward backtesting for games |
| HIGH | Model | Fixed ensemble weights, no learned stacking |
| HIGH | Model | Win/margin/total inconsistency |
| HIGH | Features | Shallow opponent adjustment (single-pass) |
| HIGH | Features | No player impact modeling for game predictions |
| HIGH | Ops | No model versioning or rollback |
| MEDIUM | Data | No data validation layer |
| MEDIUM | Model | Market residual overcorrection (no clipping) |
| MEDIUM | Model | Calibration CV uses random folds (data leakage) |
| MEDIUM | Features | No pace/context interaction features |
| MEDIUM | Ops | No dependency management |
| LOW | Features | Missing data sources (play-by-play, tracking) |
| LOW | Model | Q4 model not integrated with live predictions |
| LOW | Code | Monolith architecture (4K + 9K line files) |
