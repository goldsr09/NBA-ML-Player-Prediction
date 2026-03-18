# Data Expansion & Model Architecture Plan

## Current State

### Architecture
```
Stage 1: Minutes model (XGB + LGBM → Ridge meta-learner)
Stage 2: Per-stat rate models (points/rebounds/assists/fg3m), each XGB+LGBM stacked
          pred_stat ≈ f(features, pred_minutes)
Stage 3: OOF residual correction (XGB on out-of-fold errors, clipped ±20%)
Stage 4: Quantile uncertainty (XGB quantile regression at q10/q25/q75/q90)
Stage 5: Market residual model (adjusts prediction using market line + odds context)
Stage 6: Probability calibration (isotonic on t-distribution z-scores, per-side)
```

### Feature Count
- Common: ~70 features (all targets share)
- Per-target specific: 10-20 additional
- Total per model: ~80-90 features after filter_features drops all-NaN columns
- Training rows: ~61,000 (4 seasons of player-games with 20+ game minimum)

### Data Sources
- NBA CDN: boxscores, schedule
- ESPN: odds, injuries, lineups, props
- BallDontLie V2: tracking + hustle (5,813 games)
- stats.nba.com: advanced box score (currently down)

---

## Part 1: New Data to Pull In

### Tier 1 — Use data we already have but don't exploit

**1.1 Travel distance**
- Source: `TEAM_COORDS` dict + `haversine_miles()` already in the monolith
- Features to engineer:
  - `travel_miles_last_game`: distance from previous game venue to current
  - `travel_miles_last_3_games`: cumulative travel over 3-game window
  - `is_long_road_trip`: >3 consecutive away games
  - `cross_timezone_game`: team crossed 2+ time zones
  - `travel_x_b2b`: interaction (long travel + B2B is worse than either alone)
- Why it matters: A team that played in Portland last night and flies to Miami for tonight's game will underperform across the board. Current B2B features don't capture the *severity* of the travel.

**1.2 Opponent injury features**
- Source: ESPN injury report (already fetched daily)
- Features to engineer:
  - `opp_missing_minutes_top5`: total avg minutes of opponent's top 5 players who are OUT
  - `opp_missing_starter_count`: how many starters the opponent is missing
  - `opp_rim_protector_out`: binary — is the opponent's primary rim protector out?
  - `opp_primary_ball_handler_out`: is their PG/primary playmaker out?
- Why it matters: If the opposing center is out, your center gets easier rebounds. If their best perimeter defender is out, your wing gets more open looks. This is a first-order effect that the model currently ignores entirely.

**1.3 BDL fields we fetch but don't use**
- `trk_dist_miles` (distance run in game): fatigue proxy — a player who ran 3+ miles last game on a B2B will be more tired
- `trk_avg_speed`: athleticism proxy — declining speed trend could signal fatigue or injury
- `contested_shots_2pt` vs `contested_shots_3pt`: split tells you if a defender is primarily contesting at the rim vs perimeter — relevant for opponent defense features
- Features: rolling averages of distance/speed, distance_x_b2b interaction, speed trend (declining = fatigue)

### Tier 2 — New data from existing API (BallDontLie)

**2.1 BDL season averages with tracking subcategories**
- Endpoint: `GET /v1/season_averages?season=2025&player_id=X`
- Categories available: `catchshoot`, `pullupshot`, `drives`, `passing`, `rebounding`, `defense`, `speeddistance`, `possessions`
- These give us the shot-type breakdowns that stats.nba.com can't serve right now
- Features: catch_shoot_fg3_pct (season), pullup_fg3_pct, drives_per_game, drive_pts_per_game
- Why: catch-and-shoot percentage is one of the stickiest shooting metrics — more predictive than raw fg3_pct for 3PM props

**2.2 BDL player info for physical attributes**
- Endpoint: `GET /v1/players?search=X`
- Data: height, weight, draft position, years of experience
- Features: `player_height_inches`, `player_weight`, `years_experience`, `is_rookie`
- Why: height matters for rebounds (taller players have structural advantage), weight correlates with durability on B2Bs, rookies have different fatigue curves

### Tier 3 — New external data sources

**3.1 Market line movement / steam**
- Source: The Odds API (key already exists in `.odds_api_key`)
- Fetch opening and current lines from market data provider
- Features: `line_move_direction` (+1 if line went up, -1 if down), `line_move_magnitude`, `time_since_line_set`
- Why: Informed participants move lines. A line that drops from 24.5 to 22.5 means informed money expects the under. This is the single strongest short-term signal in sports prediction. It's essentially a free feature that captures information the model doesn't have (injury intel, lineup decisions, weather, etc.)

**3.2 DFS consensus projections**
- Source: FantasyPros, NumberFire, or RotoGrinders (scrape or API)
- Feature: `consensus_projection_{stat}` — average of 3-5 expert projections
- Why: Expert consensus embeds qualitative factors (matchup scouting, coach tendencies, game narrative) that purely statistical models miss. Using it as a feature lets the model learn when to trust vs discount consensus.

**3.3 Coach tendencies / rotation patterns**
- Source: Derive from existing boxscore data (no new API needed)
- Features:
  - `coach_avg_starters_minutes`: how many minutes does this coach give starters?
  - `coach_blowout_pull_threshold`: at what margin does the coach pull starters? (derive from historical game score differential vs minutes played)
  - `coach_b2b_rest_tendency`: does this coach rest players on B2Bs more than average?
- Why: Coach effects are real and persistent. Some coaches play 9-man rotations, others play 11. Some pull starters at +20, others ride them. This isn't captured by any current feature.

---

## Part 2: Architecture Evaluation

### Current architecture strengths
- **Two-stage decomposition (minutes × rate)** correctly separates the two biggest sources of variance: *will the player play?* and *how productive will they be per minute?*
- **XGB + LGBM stacking** is the gold standard for tabular data. These models handle missing values, feature interactions, and nonlinearity naturally.
- **OOF residual model** catches systematic biases the base model misses.
- **Quantile regression** provides distribution-aware uncertainty without assuming a parametric shape.

### Current architecture weaknesses

**Problem 1: Multiplicative error propagation**

The two-stage prediction is: `pred_stat = f(features, pred_minutes)` where `pred_minutes` comes from Stage 1.

If the minutes model is off by 15% (predicts 32 when actual is 28), then ALL stat predictions for that player are biased high by roughly 15%. This is the single biggest source of correlated errors across stat types.

The residual model (Stage 3) partially corrects this, but it's trained on *average* residuals — it can't know that tonight's specific minutes prediction is too high.

**Problem 2: Rate isn't constant across minutes**

The model assumes `stat ≈ rate × minutes`, but:
- A player who plays 38 minutes (starter in a close game) has a different per-minute rate than one who plays 22 minutes (blowout, bench time)
- Fatigue: per-minute rate declines in minutes 35-40
- Garbage time: a player's last 5 minutes in a blowout have inflated stats against backups
- Game script: a team that's losing plays differently than one that's winning

The current model does include `pred_minutes` as a feature, so XGBoost CAN learn nonlinear minutes→stat relationships. But the Stage 3 residual model doesn't have access to the *uncertainty* in minutes — it doesn't know whether the minutes prediction is confident or shaky.

**Problem 3: No direct prediction baseline**

There's no "direct" model that predicts stats without the minutes decomposition. In some cases (especially assists for non-point-guards, or fg3m for low-volume shooters), the direct statistical relationship to minutes is weak, and a direct model might outperform the two-stage approach.

**Problem 4: Fixed distribution assumption for probabilities**

The t(df=7) distribution is applied uniformly to all stat types. But:
- Points: roughly symmetric, heavy-tailed (t-distribution is decent)
- Rebounds: right-skewed for most players (a center can grab 18 but not -2)
- Assists: zero-inflated for non-playmakers, left-skewed for primary ball handlers
- FG3M: highly discrete (0, 1, 2, 3...), zero-inflated

A single t-distribution doesn't capture these different shapes. The quantile models (q10/q25/q75/q90) already implicitly model the shape — but they aren't used for probability computation. The probabilities come from the point prediction + assumed t-distribution, not from the quantile models.

### Proposed architecture changes

**Change 1: Ensemble direct + two-stage predictions**

Add a direct prediction model for each stat that doesn't use predicted minutes:

```
Direct model:   pred_stat_direct = g(features_without_pred_minutes)
Two-stage model: pred_stat_2stage = f(features, pred_minutes)
Final: pred_stat = α × pred_stat_direct + (1-α) × pred_stat_2stage
```

Where α is learned per-stat from OOF performance. For stats where the minutes decomposition helps (points, rebounds), α will be small. For stats where it doesn't (assists for bench players), α will be larger.

This costs almost nothing — same XGB infrastructure, just a second model per stat and a simple blending weight.

**Change 2: Quantile-based probabilities instead of parametric**

Replace the t-distribution probability calculation with direct quantile interpolation:

```
Current:  z = (line - pred) / std  →  p_over = 1 - t.cdf(z, df=7)
Proposed: interpolate between q10/q25/q50/q75/q90 to estimate P(stat > line)
```

The quantile models already exist and are already trained. They capture the actual conditional distribution shape (including asymmetry, fat tails, zero-inflation) without assuming t-distribution.

Implementation: fit a monotone interpolating spline through the 5 quantile points (q10, q25, q50=median prediction, q75, q90), then evaluate at the market line value. This gives a nonparametric CDF estimate.

Benefits:
- No more CDF mismatch bugs (no parametric assumption at all)
- Naturally handles skewed distributions (rebounds, fg3m)
- Calibration should improve because the raw probabilities are already better

**Change 3: Minutes uncertainty propagation**

Pass the minutes model's quantile predictions (q25_minutes, q75_minutes) as features to the stat models:

```
Features for points model: [..., pred_minutes, minutes_q25, minutes_q75, minutes_iqr]
```

This lets the stat model learn: "when the minutes prediction is uncertain (wide IQR), discount the stat prediction." Currently the stat model only sees a single minutes point estimate and has no way to know how reliable it is.

Implementation: train a quantile minutes model (same infrastructure as existing stat quantile models), then feed the IQR as an additional feature to Stage 2.

**Change 4: Per-stat residual std (heteroscedastic uncertainty)**

Instead of a single residual_std per stat type, train a model that predicts the *expected absolute error* for each prediction:

```
Current:  std = constant per stat (e.g., 6.08 for points)
Proposed: std = h(features, pred_value, line_value) — varies per prediction
```

Some predictions are inherently more uncertain than others:
- A starter in a close game → low uncertainty
- A questionable player on a B2B → high uncertainty
- A player whose recent stats are highly variable → high uncertainty

The quantile IQR (q75 - q25) already gives us this, but an explicit heteroscedastic model would be more precise. XGBoost with `reg:absoluteerror` on |residual| as target.

### Changes I would NOT make

**Not: Neural networks / transformers**
- With 61K training rows and ~90 features, GBMs dominate. Neural nets need 10-100x more data to match, and we'd lose interpretability.
- The sequential nature of player game logs *could* suit an LSTM/transformer, but the feature engineering already captures the temporal patterns through rolling averages/EWM. A transformer over raw game-by-game sequences would need much more data per player.

**Not: Separate models per player archetype**
- Tempting to train separate models for guards/wings/bigs, but:
  - Splits the training data 3 ways (20K each) — too thin
  - Position boundaries are blurry (is Giannis a wing or a big?)
  - GBMs can already learn position-conditional splits from the feature space
- Position IS implicitly captured through features (usage, touches, box_outs, etc.)

**Not: Bayesian updating / online learning**
- Would help with in-season adaptation, but:
  - XGBoost doesn't support incremental updates natively
  - Would need to switch to a fundamentally different model class
  - The weekly retrain cycle already adapts to recent data via recency weighting

---

## Part 3: Implementation Phases

### Phase 1: Low-hanging data (1-2 sessions)
1. Wire travel distance from monolith into props pipeline
2. Build opponent injury features from existing injury data
3. Add BDL unused fields (distance run, speed) to rolling averages
4. Fix the duplicate `pre_trk_passes_avg5` bug
5. Remove hardcoded BDL API key

### Phase 2: BDL season averages + player attributes (1 session)
1. Fetch BDL season averages with tracking subcategories (catch-shoot, pull-up, drives)
2. Fetch player physical attributes (height, weight, experience)
3. Engineer features and wire into feature lists
4. Retrain and evaluate

### Phase 3: Architecture improvements (2-3 sessions)
1. Add direct prediction models (no minutes decomposition) and blend with two-stage
2. Replace t-distribution probabilities with quantile-interpolated probabilities
3. Add minutes uncertainty (IQR) as feature to stat models
4. Evaluate each change independently via walk-forward backtest

### Phase 4: External data (1-2 sessions)
1. Fetch market line movement from The Odds API (opening vs current)
2. Derive coach tendency features from historical boxscore data
3. Wire into feature lists and retrain

### Phase 5: Validation (1 session)
1. Full walk-forward backtest comparing old vs new architecture
2. Feature ablation on each new feature group
3. Calibration analysis on new quantile-based probabilities
4. Profit/loss simulation with historical market lines

---

## Summary of expected impact

| Change | Expected Impact | Confidence | Effort |
|--------|----------------|------------|--------|
| Travel distance | +0.5-1% edge on B2B games | Medium | Low |
| Opponent injuries | +1-2% edge on affected games | High | Low |
| BDL unused fields | +0.3% general | Low | Low |
| Direct + two-stage ensemble | +0.5-1% across all stats | Medium | Medium |
| Quantile-based probabilities | Better calibration, fewer p=0/1 | High | Medium |
| Minutes uncertainty propagation | +0.5% on uncertain games | Medium | Low |
| Line movement | +1-3% on moved lines | High | Medium |
| Coach tendencies | +0.5-1% on blowout/rest games | Medium | Medium |
| BDL season averages (catch-shoot etc.) | +0.5% on fg3m/points | Medium | Low |

The highest-Accuracy changes: **opponent injuries**, **quantile-based probabilities**, and **line movement**. These address real blind spots in the current system. The architecture changes (direct+two-stage blend, minutes uncertainty) are structural improvements that compound with better data.
