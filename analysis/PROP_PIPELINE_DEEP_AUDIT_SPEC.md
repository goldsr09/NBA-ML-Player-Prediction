# Player Props Pipeline Deep Audit Spec

## Objective

Perform an exhaustive audit of the entire player props prediction pipeline (`predict_player_props.py` + dependencies) to identify:

1. **Data pipeline holes** — missing data, silent failures, stale caches, join losses
2. **Feature engineering bugs** — leakage, math errors, missing signals, stale anchoring
3. **Model logic gaps** — training/inference asymmetry, miscalibration, missing targets
4. **Signal generation flaws** — threshold logic errors, missing gates, P&L leaks
5. **Missing signals** — high-value features not yet exploited

---

## Audit Scope

### Files to Analyze

| File | Role | Lines |
|------|------|-------|
| `scripts/predict_player_props.py` | Core pipeline (all props logic) | ~8573 |
| `scripts/analyze_nba_2025_26_advanced.py` | Monolith: data fetch, team features, player parsing | ~1200 |
| `scripts/nba_evaluate.py` | Evaluation metrics (Brier, calibration, ATS, P/L) | ~300 |
| `scripts/fetch_historical_seasons.py` | Historical data fetcher | ~200 |

### Data Sources to Trace

- NBA CDN (`cdn.nba.com/static/json/liveData/boxscore/...`) — boxscores
- NBA CDN (`cdn.nba.com/static/json/staticData/scheduleLeagueV2_1.json`) — schedule
- ESPN Scoreboard API — game metadata, event IDs
- ESPN Odds API — game-level spread/total/moneyline
- ESPN PropBets API — player prop lines (over/under)
- ESPN Injuries API — injury report
- ESPN Summary API — confirmed starters
- stats.nba.com BoxScoreAdvancedV3 — player advanced metrics (usage, pace, TS%)
- The Odds API (optional, free tier) — alternative prop lines
- Manual CSV prop lines — user-supplied overrides
- Historical cache (`analysis/output/historical_cache/`) — seasons 2021-22 through 2024-25

---

## Phase 1: Data Ingestion Audit

### 1.1 Boxscore Parsing Completeness

**Examine:** `parse_team_box_rows()`, `parse_player_box_rows()`, `_parse_extended_player_stats()`

- [ ] **Are all counting stats extracted?** Verify every field from the NBA CDN boxscore JSON is parsed. Cross-reference the raw JSON schema against what's extracted. Look for stats available but not used (e.g., `plusMinusPoints`, `secondChancePoints`, `pointsOffTurnovers`, `benchPoints`).
- [ ] **Are there silent parsing failures?** The `except Exception: continue` pattern in `load_extended_player_stats()` and `load_historical_season()` swallows all errors. Quantify how many games/players are silently dropped.
- [ ] **Is `_minutes_to_float()` handling all formats?** Check for edge cases: `"PT00M00.00S"` (DNP), negative values, null strings, very large values (triple OT).
- [ ] **OT period detection accuracy:** `n_ot_periods = max(0, game_period - 4)` — verify `game.get("period", 4)` is reliably populated. Check what happens for games suspended/postponed mid-game.
- [ ] **Player ID consistency across seasons:** Are `personId` values stable across seasons? Could a trade mid-season cause duplicate player entries with different team codes?

### 1.2 Extended Stats Coverage

**Examine:** `_parse_extended_player_stats()` at line ~1503

- [ ] **Which boxscore fields exist but are NOT parsed?** Download a raw boxscore JSON and diff against parsed fields. Likely missing: `pointsSecondChance`, `pointsOffTurnovers`, `fastBreakPointsAttempted`, `benchPoints`, `biggestLead`, `timeLeading`.
- [ ] **Fouls drawn accuracy:** `stats.get("foulsDrawn")` — is this field reliably present in all seasons? Pre-2023 boxscores may not have it.
- [ ] **Paint/fastbreak points:** Available for all players or only team aggregates?

### 1.3 BoxScoreAdvancedV3 Ingestion

**Examine:** `load_boxscore_advanced_stats()`, `_parse_boxscore_advanced_payload()`, `_stats_nba_fetch_json()`

- [ ] **Coverage gap analysis:** How many games have advanced stats cached vs. total games? Is there a systematic gap (e.g., all pre-2023 games missing)?
- [ ] **Rate limiting:** `BOX_ADV_REQUEST_SLEEP_SECS = 1.0` — is this sufficient to avoid stats.nba.com IP bans? The session headers mimic a browser but the referer is `nba.com`, not `stats.nba.com`.
- [ ] **Parsing robustness:** Shape A (resultSets) vs Shape B (homeTeam/awayTeam) — are there games that match neither? Add logging for unrecognized payload shapes.
- [ ] **Usage percentage interpretation:** `adv_usage_pct` from different payload versions may have different scales (0-1 vs 0-100). Verify normalization.

### 1.4 Schedule and Odds Fetch

**Examine:** `fetch_schedule_df()`, `join_espn_odds()`, `load_historical_espn_odds()`

- [ ] **Games with no odds:** How many training games have `NaN` for `market_total_close` and `market_home_spread_close`? This affects `implied_total`, `implied_team_total`, and all Vegas-derived features.
- [ ] **ESPN abbreviation mapping completeness:** `ESPN_ABBR_MAP` handles 8 aliases. Are there others (e.g., playoff-specific codes, All-Star game codes)?
- [ ] **Historical odds join loss:** `load_game_odds_lookup()` joins on `["game_date_est", "home_team", "away_team"]`. If either the schedule or odds have different date formats or team codes, rows silently drop.

### 1.5 Prop Line Fetching

**Examine:** `fetch_espn_player_props()`, `fetch_odds_api_player_props()`, `load_manual_prop_lines()`

- [ ] **ESPN prop parsing fragility:** The over/under pairing assumes `entries[0]` = over, `entries[1]` = under. If ESPN changes ordering or adds a third entry type, this silently produces wrong data.
- [ ] **Athlete name resolution:** `_resolve_espn_athlete()` does network calls to resolve athlete IDs. If an athlete isn't found, the prop is silently skipped. How many props are lost this way?
- [ ] **Prop line caching:** `fetch_player_prop_lines()` writes a `.none.json` marker when no lines are found. The 45-minute retry (`NO_LINES_RETRY_SECS_SAME_DAY`) may be too aggressive for early-morning runs where lines appear later.
- [ ] **No team on ESPN props:** `"team": ""` is set for ESPN-fetched props. Team matching then relies solely on player name normalization, which can cause mismatches for players with identical names (rare but possible).
- [ ] **Steals lines fetched but not modeled:** `_ESPN_PROP_TYPE_MAP` includes `"Total Steals": "steals"` but `PROP_TARGETS = ["points", "rebounds", "assists", "minutes"]` excludes steals. Lines are fetched but never used for edge computation.

### 1.6 Injury Report

**Examine:** `fetch_injury_status_map()`, `fetch_espn_injury_report()`, `filter_out_inactive()`

- [ ] **Injury report staleness:** The injury report is fetched once per run. For games later in the day, the report may be stale (player upgraded to probable after initial fetch).
- [ ] **`availability_prob` default:** When `status_prob` is NaN, it defaults to 0.5. Is this appropriate for "day-to-day" vs. "questionable" vs. "probable"?
- [ ] **Doubtful vs. Out treatment:** Doubtful players are removed by default (`remove_doubtful=True`). If a doubtful player ends up playing, the model had no prediction for them. Should we model them with a minutes discount instead?

---

## Phase 2: Feature Engineering Audit

### 2.1 Rolling Window Integrity

**Examine:** `build_player_features()` rolling computations (lines ~1852-1900)

- [ ] **Shift correctness:** All rolling features use `.shift(1)` to prevent leakage. Verify this is applied uniformly — search for any `transform(lambda s: s.rolling(...)` without `.shift(1)`.
- [ ] **Cross-team contamination:** `player_group = ["team", "player_id", "season"]`. If a player is traded mid-season, they appear under two team codes. The rolling windows restart for the new team, losing context from the old team. Should `player_group` be `["player_id", "season"]` instead?
- [ ] **Season boundary handling:** Rolling windows don't cross season boundaries (grouped by season). Is this correct? A player's last 10 games of 2024-25 should inform their early 2025-26 predictions.
- [ ] **min_periods=1 everywhere:** `rolling(5, min_periods=1)` means a single game produces an avg5. This is extremely noisy for early-season predictions. Should `min_periods` be higher?

### 2.2 OT Regulation Adjustment

**Examine:** Lines ~1802-1900

- [ ] **Regulation factor math:** `reg_factor = 48.0 / (48.0 + 5.0 * n_ot_periods)` — this assumes each OT is exactly 5 minutes. NBA OT is 5 minutes, but actual time played varies. Is this a meaningful approximation?
- [ ] **Per-minute rates vs. regulation-adjusted counting stats:** Per-minute rates (`pts_per_min`) are computed from raw stats, while rolling averages use `_reg` adjusted stats. There's an implicit assumption that per-minute rates are OT-invariant — is this true? Players may play garbage time in OT with different efficiency.
- [ ] **Rename collision risk:** The `_reg_avg` → `_avg` rename map (`c.replace("_reg_", "_")`) could collide if there's ever a feature that naturally contains `_reg_` in its name.

### 2.3 Venue Split Features

**Examine:** Lines ~2042-2072

- [ ] **Venue split leakage:** `s.shift(1).expanding(min_periods=3).mean()` on home/away subsets, then `ffill()`. The ffill crosses home/away boundaries — a home game fills forward to the next away game. Is this intentional?
- [ ] **Home/away imbalance:** Early-season, a player may have 8 home games and 2 away games. The away average from 2 games is unreliable.
- [ ] **Venue splits not used in feature list:** Are `pre_{stat}_venue_diff` features actually in `get_feature_list()`? If they're computed but never used by the model, they waste processing time. Check if venue features appear in any target-specific feature lists.

### 2.4 Opponent Positional Defense

**Examine:** Lines ~2330-2365

- [ ] **Position mapping accuracy:** `POSITION_GROUPS` maps positions to G/F/C. What is the `position` column source? If it's from boxscores, is it accurate (some APIs return "G-F" or blank)?
- [ ] **Defense stats computation uses current-game data:** `defense_stats = pg.groupby(["game_id", "game_time_utc", "opp", "pos_group"]).agg(pts_allowed=("points", "sum"), ...)`. This aggregates points *scored by* each position group *against* the opponent. But the shift(1) rolling average is applied *after* aggregation. Is the groupby leaking current-game data into the aggregation? The player's own stats are included in the aggregation that feeds back to them.
- [ ] **Small sample positions:** Centers with injury may leave only 1 player in the "C" group. The `pre_opp_reb_allowed_to_pos_avg10` for centers would be very noisy.

### 2.5 Team Context Features

**Examine:** Lines ~2112-2195

- [ ] **Missing team features:** The merge `pg.merge(tg_subset, on=["game_id", "team"], how="left")` can produce NaN rows if `team_games` doesn't have all the features for certain games (e.g., very early season). How many training rows have NaN team context?
- [ ] **Opponent features missing for some games:** The opponent merge requires `"opp"` column in `pg`. Is `opp` reliably populated for all player-game rows?
- [ ] **Injury proxy features:** `team_injury_proxy_missing_minutes5` and `team_star_player_absent_flag` come from the monolith's `add_player_availability_proxy()`. How are these computed? Do they account for players who are truly absent vs. just resting?

### 2.6 Vegas-Derived Features

**Examine:** Lines ~2400-2430

- [ ] **`implied_team_total` formula:** `total / 2 - spread / 2` for home, `total / 2 + spread / 2` for away. This is the standard formula, but verify the sign convention of `market_home_spread_close` (negative = home is favorite).
- [ ] **NaN prevalence:** For historical games without odds data, all Vegas features (`implied_total`, `implied_spread`, `implied_team_total`, `abs_spread`, `spread_x_starter`, `is_big_favorite`) are NaN. XGBoost handles NaN natively, but the imputer fills with median — what median is used? If most training data has Vegas features, the median is meaningful; if most are NaN, it's meaningless.

### 2.7 Bayesian Shrinkage (Phase 12)

**Examine:** Lines ~1967-2023

- [ ] **Shrinkage weight ramp:** `games_w = (_game_num / 40.0).clip(upper=0.85).clip(lower=0.6)`. At game 1, weight on avg5 = 0.6; at game 40+, weight = 0.85. This means avg5 is always dominant (60-85%). Is the shrinkage actually strong enough to matter?
- [ ] **Volatility adjustment direction:** High CV reduces weight on avg5. This means volatile players get MORE shrinkage toward anchor. But if a player is genuinely volatile (e.g., Steph Curry's 3s), shouldn't we trust their recent form more?
- [ ] **Role change override:** During role changes, shrinkage shifts 50% back toward avg5. This interacts multiplicatively with the volatility adjustment. The combined effect may be unintuitive.

### 2.8 Recency and Trend Features

**Examine:** Lines ~1928-1965

- [ ] **`_window_slope` numerical stability:** When all 5 values are identical, `denom = dot(x_centered, x_centered) = 0`, and the function returns NaN. This is correct but may affect many bench players.
- [ ] **`recent_vs_season` ratio:** `avg3 / season_avg`. When season_avg is clipped to 0.1, a player averaging 0.1 rebounds with avg3=3.0 gets `recent_vs_season = 30.0`. This extreme ratio could dominate tree splits.
- [ ] **EWM variance:** `s.shift(1).ewm(span=5, min_periods=3).var()` — this is the *exponentially weighted* variance. The span=5 gives heavy weight to last 2-3 games. For players with one outlier game, this spikes dramatically.

### 2.9 Missing Features (Not Yet Implemented)

- [ ] **Head-to-head history:** Player performance against specific opponents (not just opponent position defense). Some players consistently perform well/poorly against specific teams.
- [ ] **Game pace prediction:** Current `matchup_pace_avg` is a simple average of team paces. A proper pace model would account for pace-of-play tendencies.
- [ ] **Home court advantage by team:** Different arenas have different effects (altitude in Denver, travel fatigue for West Coast road trips).
- [ ] **Lineup-adjusted features:** When key players are out, the remaining players' roles shift. The current `team_injury_pressure` is a rough proxy; explicit lineup encoding would be better.
- [ ] **Time-of-game features:** Early afternoon games vs. prime time may affect player performance.
- [ ] **Scoring environment trends:** League-wide scoring trends within a season (e.g., post-All-Star break).
- [ ] **Prop line movement as a feature:** Open line → current line movement signals sharp money. This is partially implemented (`line_move`, `line_move_pct`) but not used as a model feature.
- [ ] **Player matchup history:** How does this specific player perform against this specific opponent historically?
- [ ] **Blocks/steals as prop targets:** Lines are fetched for steals but never modeled. Blocks lines aren't even fetched.
- [ ] **Combos (PRA, PR, PA):** Points+Rebounds+Assists combo props are popular but not modeled.
- [ ] **Defensive matchup quality beyond position:** PG vs. elite defensive PG vs. poor defensive PG. Current features only look at team-level positional defense.
- [ ] **Ejection/early foul-out risk:** Players with high foul rates risk early exits. This affects minutes projection asymmetrically.

---

## Phase 3: Model Training Audit

### 3.1 Training/Inference Asymmetry

**Examine:** `train_two_stage_models()` vs `predict_two_stage()`

- [ ] **`pred_starter_prob` training vs. inference:** At training time, `pred_starter_prob = pre_starter_rate` (historical). At inference, it can be overridden by `confirmed_starter`. The model never sees confirmed starters during training — does this cause distribution shift?
- [ ] **Recency weighting:** Training uses `recency_weight = 0.4 + 0.6 * (order / max_order)` — recent games get up to 1.0 weight, old games get 0.4. But the weight also includes `1.0 + 0.2 * starter` — starters always get 20% more weight. Does this bias the model toward starters?
- [ ] **OOF predictions at training vs. live predictions:** During training, `pred_minutes` comes from OOF predictions. At inference, `pred_minutes` comes from the trained model applied to the new data. These have different error distributions.

### 3.2 Two-Stage Model Design

**Examine:** `train_two_stage_models()`, Stage 1 → Stage 2 → Stage 3

- [ ] **Minutes model circularity:** The minutes model uses features like `pre_points_avg3` and `pre_points_avg5`. Points depend on minutes. If a player's minutes increase, points increase, which feeds back into the minutes model. Is this circular dependency a problem?
- [ ] **Stage 3 residual clipping:** Residual corrections are clipped to ±20% of base prediction. For a player predicted at 5 rebounds, max correction is ±1.0. For 30 points, it's ±6.0. Is the percentage-based clip appropriate for low-line stats?
- [ ] **Ensemble stacking:** Ridge meta-learner on (XGB_pred, LGBM_pred). With only 2 features, Ridge can overfit to the training split. Should this be cross-validated?

### 3.3 Hyperparameter Selection

**Examine:** `train_prop_model()`, `train_prop_model_lgbm()` defaults

- [ ] **XGBoost defaults differ by target:** Minutes model gets `n_estimators=500, max_depth=4, lr=0.03, reg_lambda=3.0`. Other stats get `n_estimators=400, max_depth=5, lr=0.025, reg_lambda=2.0`. Were these tuned or hand-set?
- [ ] **No early stopping:** Both XGBoost and LightGBM train for a fixed number of estimators. Without early stopping, the models may overfit. `eval_metric="mae"` is set but no eval set is provided.
- [ ] **No hyperparameter tuning:** Unlike the game-level model (which uses Optuna), the prop models use fixed hyperparameters. The `--tune` flag exists for the game model but not for props.

### 3.4 Feature Selection

**Examine:** `get_feature_list()`, `filter_features()`

- [ ] **Feature list explosion:** Common features alone are ~90+ features. With target-specific features, some models have 120+ features. For training sets of ~15K rows, this is a high feature-to-sample ratio for tree models.
- [ ] **Duplicate/redundant features:** `pre_minutes_avg5`, `pre_minutes_ewm5`, `pre_minutes_avg3`, and `pre_minutes_shrunk` all measure roughly the same thing. The model may split on them interchangeably, adding noise without information.
- [ ] **Features that are always NaN at training time:** `injury_availability_prob`, `injury_unavailability_prob`, `injury_is_out`, `injury_is_doubtful`, `injury_is_questionable`, `injury_is_probable`, `lineup_confirmed`, `confirmed_starter`, `pred_starter_prob` — these are all set to NaN/0 during training (line ~2388-2398). The model can never learn from them. They only have signal at inference time. **This means the model has never seen the effect of injuries on player stats.**
- [ ] **Referee features mostly NaN:** `ref_crew_avg_total`, etc. come from `build_referee_game_features()`. How many training rows actually have non-NaN referee data? If coverage is <10%, these features add noise.
- [ ] **Season averages deliberately excluded:** Comment says "season averages excluded to prevent stale early-season production." But `pre_{stat}_season` is computed and available — it's just not in the feature list. Is this the right call for mid/late-season predictions where season average is stable?

### 3.5 Quantile Uncertainty Models

**Examine:** `train_quantile_uncertainty_models()`

- [ ] **Quantile crossing:** XGBoost `reg:quantileerror` does not guarantee `q10 < q25 < q75 < q90`. If quantiles cross, the IQR can be negative. The `max(iqr / 1.35, 0.5)` floor handles this, but it masks a deeper calibration issue.
- [ ] **Uncertainty model trained on same features as point model:** The uncertainty model sees the OOF prediction as a feature. If the OOF prediction is systematically biased for certain player types, the uncertainty model may learn to correct the bias rather than estimate uncertainty.

### 3.6 Sample Weighting

**Examine:** Recency weighting in `train_two_stage_models()`

- [ ] **Starter bonus in weighting:** `recency_weight * (1.0 + 0.2 * starter)`. This means the model upweights starters by 20% regardless of recency. This biases the model toward starter-type players and may degrade predictions for bench players.
- [ ] **Weight floor of 0.05:** `sw = np.clip(sw, 0.05, None)`. With 4+ seasons of data, the oldest game gets weight 0.4, the newest gets 1.0. This is a mild recency effect — is it strong enough?

---

## Phase 4: Signal Generation Audit

### 4.1 Edge and EV Computation

**Examine:** `compute_prop_edges()` (line ~4746)

- [ ] **t-distribution df=7 choice:** `sp_stats.t.cdf(z, df=7)` — why df=7? This should be empirically validated against the actual residual distribution. If the true tails are lighter than t(7), probabilities are too extreme; if heavier, too conservative.
- [ ] **P(over) calibration:** After computing raw P(over) from the t-distribution, it's passed through Platt/isotonic calibration. But the calibrators are trained on a 80/20 split of `_build_market_training_frame()`, which uses only the last 180 days of cached prop lines. If market data is sparse, the calibrator may be poorly trained.
- [ ] **EV computation:** `ev_over = p_over * over_payout - (1.0 - p_over)`. This assumes a $1 wager. The payout is computed from American odds. Verify the decimal odds conversion: for -110, payout should be $0.909 (profit per $1 risked), not $1.909.
- [ ] **Side-specific thresholds:** OVER requires 20% edge and 0.30 EV; UNDER requires 15% edge and 0.20 EV. This asymmetry was introduced to counter "observed over-bias." Has this been validated? Is the bias still present?

### 4.2 Signal Gating

**Examine:** Lines ~4909-4976

- [ ] **LEAN OVER suppression:** `SUPPRESS_LEAN_OVER = True` kills ALL lean over signals. This eliminates potentially profitable lower-confidence overs. Is the data showing that lean overs are net negative?
- [ ] **Lineup confirmation gate for OVER:** `OVER_REQUIRE_LINEUP_CONFIRMED = True` — OVER signals require confirmed starters. But confirmed starters are only available 30 minutes before tip. This means OVER signals are only generated very close to game time, when lines may have already moved.
- [ ] **Injury gate for OVER:** `OVER_MAX_INJURY_PROB = 0.15` — if injury_unavailability_prob > 15%, OVER is blocked. But at training time, `injury_unavailability_prob = NaN` always. The model has no concept of this gate.
- [ ] **Minutes gate:** `pred_minutes >= 20.0 AND pre_minutes_avg10 >= 18.0`. This filters out all role players, even if the model predicts a strong edge. A 6th man averaging 22 minutes who the model projects at 19 minutes would be filtered.

### 4.3 Portfolio Caps

**Examine:** `apply_portfolio_caps()` (not fully read — need to verify)

- [ ] **Max signals per day = 10:** Is this too few? Too many? Is it applied before or after the EV sort?
- [ ] **Max signals per player = 2:** If a player has strong edges on points, rebounds, and assists, the 3rd is dropped. Which one? Is it the weakest by EV?
- [ ] **Correlated signal handling:** The `correlated` flag is set but is it actually used in portfolio filtering? A player with OVER points and OVER rebounds is heavily correlated (both driven by minutes).

### 4.4 Points Bias Correction

**Examine:** `compute_points_bias()`, `get_active_points_bias()`

- [ ] **Bias applied only to points:** If there's a systematic bias in rebounds or assists, it's never corrected.
- [ ] **Sign flip auto-disable:** If the bias flips sign from the prior correction, it's disabled. This means the correction only works for persistent, unidirectional bias.
- [ ] **Lookback window:** 30 days of graded data. Early in the deployment, there may be <75 graded rows, making the bias inactive. Once active, the 30-day window may be too short to capture seasonal patterns.

### 4.5 Market Residual Adjustment

**Examine:** `_predict_market_residual_adjustment()`, `train_market_residual_models()`

- [ ] **Holdout validation:** The market residual model is only retained if `holdout_mae < baseline_holdout_mae` (i.e., the model beats "predict zero residual"). This is a good gate, but the holdout is a single time-ordered split — it may be noisy.
- [ ] **Feature availability:** Market residual model uses `pred_value`, `line`, `edge`, `edge_pct`, `over_implied_prob`, etc. At inference time, these are computed from the base model's prediction. At training time, they're from the 80% training split predictions. If the base model improves between training and inference, the residual model is calibrated to the old model.

---

## Phase 5: Grading and Feedback Loop Audit

### 5.1 Canonical Results

**Examine:** `save_canonical_results()`, `grade_canonical_results()`

- [ ] **Deduplication:** `generate_prediction_id("{date}_{name_norm}_{stat_type}")`. If a player's name normalizes differently across runs (e.g., accent handling changes), predictions can duplicate.
- [ ] **Grading actuals source:** Actuals come from the same boxscores used for training. If a boxscore is updated after the initial fetch (stat corrections), the graded actual may differ from the training data.
- [ ] **Push handling:** `abs(actual - line) < 1e-9` defines a push. For half-point lines (e.g., 24.5), pushes are impossible. For whole-number lines, exact matches are pushes. But some books use "at least" rules. Is this assumption correct?

### 5.2 Calibration Report

**Examine:** `compute_calibration_report()`, `get_calibration_degraded_stats()`

- [ ] **Min sample for drift detection:** `CALIB_MIN_SAMPLE = 50` — 50 graded signal bets per stat type per slice. Early in deployment, this is rarely met, making calibration monitoring inactive.
- [ ] **Degraded stats gating:** If a stat type is flagged as degraded (Brier > 0.30 or gap > 8%), ALL signals for that stat are suppressed. This is a blunt instrument — it could suppress profitable subsegments within the degraded stat.
- [ ] **Calibration vs. accuracy:** Brier score measures calibration, not sharpness. A model that always predicts 50% has perfect calibration but no edge. The drift detection should also check if the model has positive CLV or ROI.

### 5.3 Deploy Gates

**Examine:** `check_deploy_gates()`, `DEPLOY_GATES_ENFORCE`

- [ ] **Gates currently advisory:** `DEPLOY_GATES_ENFORCE = False`. All deploy gates are currently informational only. When should this be switched to enforcement?
- [ ] **Model age gate:** `DEPLOY_GATE_MAX_MODEL_AGE_DAYS = 14`. How is model age determined? If models are retrained weekly (Sunday cron), by Friday they're 5 days old, which is fine. But if the cron fails, models could be 21+ days old before anyone notices.
- [ ] **CLV gate:** `DEPLOY_GATE_MIN_CLV = 0.0` requires non-negative CLV. But `DEPLOY_CLV_MIN_SAMPLE = 50` means CLV isn't checked until 50 bets with line movement data. This may take weeks.

---

## Phase 6: Numerical and Statistical Audit

### 6.1 Distribution Assumptions

- [ ] **Gaussian/t-distribution for counting stats:** Points, rebounds, assists are bounded below by 0 and have right-skewed distributions. The normal/t-distribution assumption for P(over) may be inappropriate. Consider Poisson or negative binomial for low-count stats (rebounds, assists, 3PM).
- [ ] **Residual std estimation:** `compute_prop_residual_stds()` uses the test-set residual std from a simple chronological split. This is a single estimate — it doesn't vary by player, stat magnitude, or game context. A player projected for 30 points has different variance than one projected for 10.

### 6.2 Potential Mathematical Errors

- [ ] **`_american_odds_to_decimal` vs. `_american_odds_to_prob`:** Three different functions convert American odds: `_american_odds_to_prob`, `_american_odds_to_decimal`, `_american_odds_to_implied_prob`. The first and third are identical. `_american_odds_to_decimal` returns *profit per $1 wagered* (not total payout). Verify this is consistently used in EV computation.
- [ ] **EV formula verification:** `ev_over = p_over * over_payout - (1.0 - p_over)`. If `over_payout = _american_odds_to_decimal(-110) = 100/110 = 0.909`, then EV = p * 0.909 - (1-p). At p=0.55, EV = 0.55*0.909 - 0.45 = 0.50 - 0.45 = 0.05. This is correct for profit per $1 risked.
- [ ] **`VIG_FACTOR = 0.9524`:** This is `100/105`, representing -105 juice. But many props are at -110 (`100/110 = 0.909`). If actual odds are -110 but the VIG_FACTOR is -105, the EV is overestimated by ~4 cents per dollar.

### 6.3 Imputation Strategy

- [ ] **Median imputation for all features:** `SimpleImputer(strategy="median")` is used everywhere. For binary features (e.g., `is_b2b`, `is_big_favorite`), median imputation may produce 0 (if most games are not B2B), which is semantically correct. But for features like `adv_usage_pct` where NaN means "data not available," median imputation may introduce phantom signal.
- [ ] **Imputer fit on training data applied to inference:** The imputer learned medians from training data. If inference data has a different feature distribution (e.g., all NaN for a feature), the imputed values may be inappropriate.

---

## Phase 7: Performance and Reliability Audit

### 7.1 Caching Issues

- [ ] **Feature cache invalidation:** `PLAYER_FEATURE_CACHE_VERSION = "v7"`. The cache key includes `n_player_rows`, `n_team_rows`, `n_odds_rows`, and `last_game_time_utc`. If new boxscores are fetched but the count doesn't change (e.g., a correction), the cache won't invalidate.
- [ ] **In-memory caches:** `_EXTENDED_STATS_CACHE`, `_BOX_ADV_CACHE` are module-level globals. These persist for the entire process lifetime. If the script is run as a long-lived service, stale caches could cause issues.
- [ ] **ESPN athlete cache:** `espn_athlete_cache.json` grows indefinitely. With 500+ athletes, this file could grow large and slow down reads.

### 7.2 Error Handling Gaps

- [ ] **Silent `except Exception: continue`:** This pattern appears ~20+ times in data loading functions. A single malformed JSON file could cause an entire season's data to be silently dropped.
- [ ] **Network failure handling:** `fetch_json()` retries 3 times with 0.4s delay. For stats.nba.com, retries are 5 times with exponential backoff. But there's no circuit breaker — if the API is down, the script hangs for minutes per game.

### 7.3 Prediction Pipeline End-to-End Test

- [ ] **Training data leakage check:** Run the pipeline with `--walk-forward` and compare per-fold MAE to the simple `--backtest` MAE. If walk-forward MAE is significantly worse, there may be leakage in the simple backtest.
- [ ] **Prediction consistency:** Run predictions for the same date twice. Are results identical? (They should be if caches are stable.)
- [ ] **Edge case handling:** What happens when a game has 0 eligible players (all filtered by min_games)? When all prop lines are NaN? When a player has exactly `DEFAULT_MIN_GAMES` = 20 games?

---

## Deliverables

For each item checked, the auditing agent should produce:

1. **Finding:** One sentence describing what was found
2. **Severity:** Critical / High / Medium / Low / Info
3. **Evidence:** Code reference (file:line) or data sample
4. **Recommendation:** Specific fix or investigation needed
5. **Priority:** P0 (fix immediately) / P1 (fix before next deployment) / P2 (backlog)

### Output Format

Write findings to `analysis/output/prop_pipeline_audit_report.md` as a structured markdown document with the above categories. Group findings by phase (Data, Features, Model, Signal, Feedback, Numerical, Reliability).

### Success Criteria

- Every checkbox in this spec has a finding (even if "no issue found")
- At least 5 critical/high findings with concrete code fixes
- At least 3 "missing signal" recommendations with expected impact estimate
- All mathematical formulas verified with worked examples
